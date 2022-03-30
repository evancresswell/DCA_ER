import os.path, sys

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder

import Bio.PDB, warnings
pdb_list = Bio.PDB.PDBList()
pdb_parser = Bio.PDB.PDBParser()
from scipy.spatial import distance_matrix
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

import expectation_reflection as ER
from direct_info import direct_info

from joblib import Parallel, delayed
import timeit

import matplotlib.pyplot as plt

# # --- Import our Code ---# #
#import emachine as EM
from direct_info import direct_info

# import data processing and general DCA_ER tools
from data_processing import data_processing_msa2pdb
import ecc_tools as tools
from pathlib import Path
np.random.seed(1)

from Bio import SeqIO
from Bio.PDB import *
from scipy.spatial import distance_matrix
from Bio import pairwise2
#from Bio.SubsMat.MatrixInfo import blosum62
pdb_parser = Bio.PDB.PDBParser()

from prody import *

from ecc_tools import scores_matrix2dict
from ecc_tools import roc_curve, roc_curve_new, precision_curve


create_new = True
printing = True
removing_cols = True
remove_diagonals = False


data_path = Path('/data/cresswellclayec/DCA_ER/Pfam-A.full')
data_path = Path('/data/cresswellclayec/Pfam-A.full')

# Define data directories
DCA_ER_dir = '/data/cresswellclayec/DCA_ER' # Set DCA_ER directory
biowulf_dir = '%s/biowulf_full' % DCA_ER_dir

out_dir = '%s/protein_data/di/' % biowulf_dir
processed_data_dir = "%s/protein_data/data_processing_output" % biowulf_dir
pdb_dir = '%s/protein_data/pdb_data/' % biowulf_dir
metric_dir = '%s/protein_data/metrics/' % biowulf_dir


pfam_dir = "/fdb/fastadb/pfam"

from data_processing import pdb2msa, data_processing_pdb2msa

number2letter = {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K', 9: 'L', \
                     10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'V', 18: 'W', 19: 'Y', 20: '-',
                     21: 'U'}


# pdb_path = "/pdb/pdb/zd/pdb1zdr.ent.gz"
pdb_path = sys.argv[1]
pdb_id = os.path.basename(pdb_path)[3:7]
n_cpus = int(sys.argv[2])



prody_df = pd.read_csv('%s/%s_pdb_df.csv' % (pdb_dir, pdb_id))
pdb2msa_row = prody_df.iloc[0]

pdb_id = pdb2msa_row['PDB ID']
pfam_id = pdb2msa_row['Pfam']

s0 = np.load("%s/%s_%s_preproc_msa.npy" % (processed_data_dir, pfam_id, pdb_id))
removed_cols = np.load("%s/%s_%s_removed_cols.npy" % (processed_data_dir, pfam_id, pdb_id))
s_index = np.load("%s/%s_%s_preproc_sindex.npy" % (processed_data_dir, pfam_id, pdb_id))
#pdb_s_index = np.load("%s/%s_%s_preproc_pdb_sindex.npy" % (processed_data_dir, pfam_id, pdb_id))
ref_seq = np.load("%s/%s_%s_preproc_refseq.npy" % (processed_data_dir, pfam_id, pdb_id))
ref_seq_str = [number2letter[num] for num in ref_seq]
print('ref_seq = %s' % ''.join(ref_seq_str))
for i, seq in enumerate(s0):
    seq_str = [number2letter[num] for num in seq]
    if ''.join(seq_str) == ''.join(ref_seq_str):
        tpdb=i
        break

s0_pydca = np.load("%s/%s_%s_allCols_msa.npy" % (processed_data_dir, pfam_id, pdb_id))
s_index_pydcs = np.load("%s/%s_%s_allCols_sindex.npy" % (processed_data_dir, pfam_id, pdb_id))
ref_seq_pydca = np.load("%s/%s_%s_allCols_refseq.npy" % (processed_data_dir, pfam_id, pdb_id))
print('ref_seq = %s' % ''.join(ref_seq_pydca))
for i, seq in enumerate(s0_pydca):
    if ''.join(seq) == ''.join(ref_seq_pydca):
        tpdb_pydca=i
        break

print('found tpdb for s0 and s0_pydca: %d %d ' % (tpdb,tpdb_pydca))


# number of positions
n_var = s0.shape[1]
n_seq = s0.shape[0]
if n_seq < 500:
    print('not enough sequences')
    sys.exit()

print("Number of residue positions:",n_var)
print("Number of sequences:",n_seq)

bootstrap_seq_num = int(.6 * n_seq)
print('bootstrapping we will using .6 of sequences or %d' % bootstrap_seq_num )

# Expectation Reflection
#=========================================================================================
def predict_w(s,i0,i1i2,niter_max,l2):
    #print('i0:',i0)
    i1,i2 = i1i2[i0,0],i1i2[i0,1]

    x = np.hstack([s[:,:i1],s[:,i2:]])
    y = s[:,i1:i2]

    h01,w1 = ER.fit(x,y,niter_max,l2)

    return h01,w1


print('Generating contact map...')

unzipped_pdb_filename = os.path.basename(pdb_path).replace(".gz", "")
pdb_out_path = "%s%s" % (pdb_dir, unzipped_pdb_filename)
### ISSUE THIS SHOULDNT WORK BUT IT DOES>>> TO WHAT EXTENT CAN PDB_S_INDEX BE REPLACED BY S_INDEX
pdb_s_index = s_index
ct, ct_full = tools.contact_map_pdb2msa_new(pdb2msa_row, pdb_out_path, removed_cols, pdb_s_index, pdb_out_dir=pdb_dir, printing=True)
print('contact dimentions: ', ct.shape)


start_time = timeit.default_timer()
n_iter = 100
bootstrap_aucs = []
print('\n\nRunning Bootstrap Simulations...')
for i in range(n_iter):
    bootstrap_s0 = s0[np.random.choice(len(s0), size=bootstrap_seq_num,replace=False)]
    # number of aminoacids at each position
    mx = np.array([len(np.unique(bootstrap_s0[:,i])) for i in range(n_var)])
    #mx = np.array([m for i in range(n_var)])
    print("Number of different amino acids at each position",mx)

    mx_cumsum = np.insert(mx.cumsum(),0,0)
    i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T 
    # print("(Sanity Check) Column indices of first and (",i1i2[0],") and last (",i1i2[-1],") positions")
    # print("(Sanity Check) Column indices of second and (",i1i2[1],") and second to last (",i1i2[-2],") positions")


    # number of variables
    mx_sum = mx.sum()
    print("Total number of variables",mx_sum)

    # number of bias term
    n_linear = mx_sum - n_var

    onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
    # s is OneHot encoder format, bootstrap_s0 is original sequnce matrix
    s = onehot_encoder.fit_transform(bootstrap_s0)
    # print("Amino Acid sequence Matrix\n",bootstrap_s0)
    # print("OneHot sequence Matrix\n",s)
    # print("An individual element of the OneHot sequence Matrix (size:",
    #      s.shape,") --> ",s[0], " has length ",s[0].shape)

    # Define wight matrix with variable for each possible amino acid at each sequence position
    w = np.zeros((mx.sum(),mx.sum())) 
    h0 = np.zeros(mx.sum())


    #-------------------------------
    # parallel
    res = Parallel(n_jobs = n_cpus-2)(delayed(predict_w)\
	    (s,i0,i1i2,niter_max=10,l2=100.0)\
	    for i0 in range(n_var))

    #-------------------------------
    for i0 in range(n_var):
        i1,i2 = i1i2[i0,0],i1i2[i0,1]

        h01 = res[i0][0]
        w1 = res[i0][1]

        h0[i1:i2] = h01
        w[:i1,i1:i2] = w1[:i1,:]
        w[i2:,i1:i2] = w1[i1:,:]

    # make w symmetric
    w = (w + w.T)/2.


    di = direct_info(bootstrap_s0,w)
    print(di)
    print(di.shape)
    print(len(s_index))


    # get contact map and auc.
    #print("Direct Information from Expectation reflection:\n",di)
    print('s_index (%d): ' %len(s_index), s_index)
    er_di = di
    print('ER DI shape (before removing cols): ' , di.shape)
    #if not removing_cols:
    #    er_di = np.delete(di, removed_cols_range,0)
    #    er_di = np.delete(er_di, removed_cols_range,1)
    #else:
    #    er_di = di
    
    print('Final ER DI shape (cols removed): ', er_di.shape)
    
    if remove_diagonals: 
        ER_di = no_diag(er_di, 4, s_index)
    else:
        ER_di = er_di
    
    #ER_di_dict = scores_matrix2dict(ER_di, s_index, removed_cols_range, removing_cols=removing_cols)
    #print(ER_di_dict[:20])
    
    
    # find optimal threshold of distance
    ct_thres = np.linspace(4.,6.,18,endpoint=True)
    n = ct_thres.shape[0]
    
    ct_mat = ct
    
    # ER ROC
    auc_ER = np.zeros(n)
    for i in range(n):
        try:
            fpr, tpr, thresholds, auc = roc_curve_new(ct_mat, ER_di, ct_thres[i])
            auc_ER[i] = auc
        except:
            auc_ER[i] = 0
    
    # Get ER method's best contact prediction
    i0_ER = np.argmax(auc_ER)
    print('ER auc max:',ct_thres[i0_ER],auc_ER[i0_ER])
    fpr0_ER, tpr0_ER, thresholds_ER, auc = roc_curve_new(ct_mat, ER_di, ct_thres[i0_ER])

    bootstrap_aucs.append(auc)    

run_time = timeit.default_timer() - start_time
print('Bootstrapping run time:',run_time)
np.save("%s/%s_%s_bootstrap_aucs.npy" % (metric_dir, pfam_id, pdb_id), bootstrap_aucs)
   
# confidence intervals
alpha = 0.95
p = ((1.0-alpha)/2.0) * 100
lower = max(0.0, np.percentile(bootstrap_aucs, p))
p = (alpha+((1.0-alpha)/2.0)) * 100
upper = min(1.0, np.percentile(bootstrap_aucs, p))
print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
print("Number of residue positions:",n_var)
print("Number of sequences:",n_seq)


