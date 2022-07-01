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



create_new = True
printing = True
removing_cols = True


data_path = Path('/data/cresswellclayec/DCA_ER/Pfam-A.full')
data_path = Path('/data/cresswellclayec/Pfam-A.full')

# Define data directories
DCA_ER_dir = '/data/cresswellclayec/DCA_ER' # Set DCA_ER directory
biowulf_dir = '%s/biowulf_full' % DCA_ER_dir

out_dir = '%s/protein_data/di/' % biowulf_dir
processed_data_dir = "%s/protein_data/data_processing_output" % biowulf_dir
pdb_dir = '%s/protein_data/pdb_data/' % biowulf_dir


pfam_dir = "/fdb/fastadb/pfam"

from data_processing import pdb2msa, data_processing_pdb2msa


import gzip, shutil
def gunzip(file_path, output_path):
    print('Unzipping %s to %s' % (file_path, output_path))
    with gzip.open(file_path,"rb") as f_in, open(output_path,"wb") as f_out:
        shutil.copyfileobj(f_in, f_out)



# pdb_path = "/pdb/pdb/zd/pdb1zdr.ent.gz"
pdb_path = sys.argv[1]
n_cpus = int(sys.argv[2])
print('\n\nUnzipping %s' % pdb_path)

unzipped_pdb_filename = os.path.basename(pdb_path).replace(".gz", "")

pdb_out_path = "%s%s" % (pdb_dir, unzipped_pdb_filename)
print('Unzipping %s to %s' % (pdb_path, pdb_out_path))

gunzip(pdb_path, pdb_out_path)
print(pdb_out_path)
print(pdb_dir)
pdb2msa_results = pdb2msa(pdb_out_path, pdb_dir, create_new=False)
print(pdb2msa_results)

if len(pdb2msa_results) > 1:
    fasta_file = pdb2msa_results[0]
    prody_df = pdb2msa_results[1]
else:
    prody_df = pdb2msa_results[0]


print('\nPDB DF with associated Protein Families\n', prody_df.loc[:,  [column for column in prody_df.columns if column not in ['locations', 'PDB Sequence']]].head())
print("\n\nLooping through Prody Search DataFrame:", prody_df.head())
rows_to_drop = []
for ir, pdb2msa_row in enumerate(prody_df.iterrows()):
    print('\n\nGetting msa with following pdb2msa entry:\n', pdb2msa_row)
    try:
        dp_result =  data_processing_pdb2msa(data_path, prody_df.iloc[pdb2msa_row[0]], gap_seqs=0.2, gap_cols=0.2, prob_low=0.004,
                               conserved_cols=0.8, printing=True, out_dir=processed_data_dir, pdb_dir=pdb_dir, letter_format=False,
                               remove_cols=True, create_new=True)
        if dp_result is not None:
            [s0, removed_cols, s_index, tpdb, pdb_s_index] = dp_result
            break
        else: 
            rows_to_drop.append(ir) 
            continue
    except Exception as e:
        print('row %d got exception: ' % ir , e)
        print('moving on.. ')
        pass


pdb_id = pdb2msa_row[1]['PDB ID']
pfam_id = pdb2msa_row[1]['Pfam']
# update Prody search DF (use same filename as pdb2msa() in data_processing
prody_df = prody_df.drop(rows_to_drop)
print("\nSaving updated Prody Search DataFrame:", prody_df.head())
prody_df.to_csv('%s/%s_pdb_df.csv' % (pdb_dir, pdb_id))

if dp_result is None:
    print('None of the available prody pdb search found matching alignments... Exiting..')
    sys.exit()
print('Done Preprocessing Data.....')



# number of positions
n_var = s0.shape[1]
n_seq = s0.shape[0]

print("Number of residue positions:",n_var)
print("Number of sequences:",n_seq)

# number of aminoacids at each position
mx = np.array([len(np.unique(s0[:,i])) for i in range(n_var)])
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
# s is OneHot encoder format, s0 is original sequnce matrix
s = onehot_encoder.fit_transform(s0)
# print("Amino Acid sequence Matrix\n",s0)
# print("OneHot sequence Matrix\n",s)
# print("An individual element of the OneHot sequence Matrix (size:",
#      s.shape,") --> ",s[0], " has length ",s[0].shape)

# Define wight matrix with variable for each possible amino acid at each sequence position
w = np.zeros((mx.sum(),mx.sum())) 
h0 = np.zeros(mx.sum())

import sys
import numpy as np
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
import expectation_reflection as ER
from direct_info import direct_info
from joblib import Parallel, delayed


# Expectation Reflection
#=========================================================================================
def predict_w(s,i0,i1i2,niter_max,l2):
    #print('i0:',i0)
    i1,i2 = i1i2[i0,0],i1i2[i0,1]

    x = np.hstack([s[:,:i1],s[:,i2:]])
    y = s[:,i1:i2]

    h01,w1 = ER.fit(x,y,niter_max,l2)

    return h01,w1

w_file = "%s/%s_%s_w.npy" % (processed_data_dir, pdb_id, pfam_id)
if os.path.exists(w_file) and not create_new:
    w = np.load(w_file)
else:
    #-------------------------------
    # parallel
    start_time = timeit.default_timer()
    #res = Parallel(n_jobs = 4)(delayed(predict_w)\
    #res = Parallel(n_jobs = 8)(delayed(predict_w)\
    res = Parallel(n_jobs = n_cpus-2)(delayed(predict_w)\
            (s,i0,i1i2,niter_max=10,l2=100.0)\
            for i0 in range(n_var))

    run_time = timeit.default_timer() - start_time
    print('run time:',run_time)
    ## This above line seems wrong, seems like the following for loop should be moved up?? not sure if this is some 
    ## python implementation or just wrong
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

# Verify that w is symmetric (sanity test)
print("Dimensions of w: ",w.shape)
np.save(w_file, w)

if not create_new and os.path.exists("%s/%s_%s_ER_di.npy" % (out_dir, pdb_id, pfam_id)):
    di = np.load("%s/%s_%s_ER_di.npy" % (out_dir, pdb_id, pfam_id))
else:
    di = direct_info(s0,w)
    np.save("%s/%s_%s_ER_di.npy" % (out_dir, pdb_id, pfam_id), di)
print(di)
print(di.shape)
print(len(s_index))
