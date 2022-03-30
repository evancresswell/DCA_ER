import os.path, sys

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import distance

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
from Bio.SubsMat.MatrixInfo import blosum62
pdb_parser = Bio.PDB.PDBParser()

from prody import *

print('Done with initial import')


create_new = False
printing = True
removing_cols = True
remove_diagonals = False

pdb_path = "/pdb/pdb/"
data_path = Path('/data/cresswellclayec/DCA_ER/Pfam-A.full')
data_path = Path('/data/cresswellclayec/Pfam-A.full')

# Define data directories
DCA_ER_dir = '/data/cresswellclayec/DCA_ER' # Set DCA_ER directory
biowulf_dir = '%s/biowulf_full' % DCA_ER_dir


out_dir = '%s/protein_data/di/' % biowulf_dir
out_metric_dir = '%s/protein_data/metrics/' % biowulf_dir

processed_data_dir = "%s/protein_data/data_processing_output/" % biowulf_dir
pdb_dir = '%s/protein_data/pdb_data/' % biowulf_dir

# pdb_path = "/pdb/pdb/zd/pdb1zdr.ent.gz"
pdb_id = sys.argv[1]
n_cpus = int(sys.argv[2])

prody_df = pd.read_csv('%s/%s_pdb_df.csv' % (pdb_dir, pdb_id))

from data_processing import pdb2msa, data_processing_pdb2msa

pdb2msa_row = prody_df.iloc[0]
pfam_id = pdb2msa_row['Pfam']

if removing_cols:
    s0 = np.load("%s/%s_%s_preproc_msa.npy" % (processed_data_dir, pfam_id, pdb_id))
    s_index = np.load("%s/%s_%s_preproc_sindex.npy" % (processed_data_dir, pfam_id, pdb_id))
    #pdb_s_index = np.load("%s/%s_%s_preproc_pdb_sindex.npy" % (processed_data_dir, pfam_id, pdb_id))
    removed_cols = np.load("%s/%s_%s_removed_cols.npy" % (processed_data_dir, pfam_id, pdb_id))
    ref_seq = np.load("%s/%s_%s_preproc_refseq.npy" % (processed_data_dir, pfam_id, pdb_id))
else: 
    s0 = np.load("%s/%s_%s_allCols_msa.npy" % (processed_data_dir, pfam_id, pdb_id))
    s_index = np.load("%s/%s_%s_allCols_sindex.npy" % (processed_data_dir, pfam_id, pdb_id)) 
    #pdb_s_index = np.load("%s/%s_%s_allCols_pdb_sindex.npy" % (processed_data_dir, pfam_id, pdb_id)) 
    removed_cols = np.load("%s/%s_%s_removed_cols.npy" % (processed_data_dir, pfam_id, pdb_id))
    ref_seq = np.load("%s/%s_%s_allCols_refseq.npy" % (processed_data_dir, pfam_id, pdb_id))
pdb_s_index = s_index # will need to rerun simulations with corrected data_processing (corrected on 3/15/22) to get pdb_s_index. not actually needed in contact mapping......


# number of positions
n_var = s0.shape[1]
n_seq = s0.shape[0]
# compute effective number of sequences
dst = distance.squareform(distance.pdist(s0, 'hamming'))
theta = .2 													# minimum necessary distance (theta = 1. - seq_identity_thresh)
seq_ints = (dst < theta).sum(axis=1).astype(float)
ma_inv = 1/((dst < theta).sum(axis=1).astype(float))  
meff = ma_inv.sum()

print("Number of residue positions:",n_var)
print("Number of sequences:",n_seq)
print('N_effective ', meff)


ER_di = np.load("%s/%s_%s_ER_di.npy" % (out_dir, pdb_id, pfam_id))
MF_di = np.load("%s/%s_%s_MF_di.npy" % (out_dir, pdb_id, pfam_id))
PMF_di_data = np.load("%s/%s_%s_PMF_di.npy" % (out_dir, pdb_id, pfam_id),allow_pickle=True)
PLM_di_data = np.load("%s/%s_%s_PLM_di.npy" % (out_dir, pdb_id, pfam_id),allow_pickle=True)

pdb_path = '%s%s/pdb%s.ent.gz' % (pdb_path, pdb_id[1:3], pdb_id) 
unzipped_pdb_filename = os.path.basename(pdb_path).replace(".gz", "")
pdb_out_path = "%s%s" % (pdb_dir, unzipped_pdb_filename)
print('Unzipping %s to %s' % (pdb_path, pdb_out_path))
# --------------------- Data Processing (should be saving correct row!!!!) --- #
import gzip, shutil
def gunzip(file_path,output_path):
    with gzip.open(file_path,"rb") as f_in, open(output_path,"wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
gunzip(pdb_path, pdb_out_path)
print('Done unzipping pdb file')




### ISSUE THIS SHOULDNT WORK BUT IT DOES>>> TO WHAT EXTENT CAN PDB_S_INDEX BE REPLACED BY S_INDEX
### ISSUE THIS SHOULDNT WORK BUT IT DOES>>> TO WHAT EXTENT CAN PDB_S_INDEX BE REPLACED BY S_INDEX
### ISSUE THIS SHOULDNT WORK BUT IT DOES>>> TO WHAT EXTENT CAN PDB_S_INDEX BE REPLACED BY S_INDEX
ct, ct_full = tools.contact_map_pdb2msa_new(pdb2msa_row, pdb_out_path, removed_cols, pdb_s_index, pdb_out_dir=pdb_dir, printing=True)
ct_file = "%s%s_%s_ct.npy" % (pdb_dir, pdb_id, pfam_id)
np.save(ct_file, ct)

#print("Direct Information from Expectation reflection:\n",di)
print('s_index (%d): ' %len(s_index), s_index)
print('ER DI shape (before removing cols): ' , ER_di.shape)
er_di = ER_di
#if not removing_cols:
#    er_di = np.delete(di, removed_cols_range,0)
#    er_di = np.delete(er_di, removed_cols_range,1)
#else:
#    er_di = di

print('contact dimentions: ', ct.shape)
print('Final ER DI shape (cols removed): ', er_di.shape)
if remove_diagonals: 
    ER_di = no_diag(er_di, 4, s_index)
else:
    ER_di = er_di

from ecc_tools import scores_matrix2dict
print(s_index)
#ER_di_dict = scores_matrix2dict(ER_di, s_index, removed_cols_range, removing_cols=removing_cols)
#print(ER_di_dict[:20])


from ecc_tools import roc_curve, roc_curve_new, precision_curve
# find optimal threshold of distance
ct_thres = np.linspace(4.,10.,18,endpoint=True)
n = ct_thres.shape[0]

ld_threshold = 5
if ld_threshold > 0:
    file_end = '_ld%d.npy' % ld_threshold
else:
    file_end = '.npy'
   

ct_mat = ct

if 1:
    # ER ROC
    auc_ER = np.zeros(n)
    for i in range(n):
        try:
            fpr, tpr, thresholds, auc, ct_pos_flat = roc_curve_new(ct_mat, ER_di, ct_thres[i] ,s_index, ld_thresh=ld_threshold)
            auc_ER[i] = auc
        except:
            auc_ER[i] = 0
    
    # Get ER method's best contact prediction
    i0_ER = np.argmax(auc_ER)
    print('ER auc max:',ct_thres[i0_ER],auc_ER[i0_ER])
    fpr0_ER, tpr0_ER, thresholds_ER, auc, tpr_uni, fpr_uni, auc_uni, uni_bins, ct_pos_flat = roc_curve_new(ct_mat, ER_di, ct_thres[i0_ER], s_index, ld_thresh=ld_threshold, get_uniform=True)
    fp_uni_file = "%s%s_%s_ER_fp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    tp_uni_file = "%s%s_%s_ER_tp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    np.save(fp_uni_file, fpr_uni)
    np.save(tp_uni_file, tpr_uni)   

    fp_file = "%s%s_%s_ER_fp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    tp_file = "%s%s_%s_ER_tp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    np.save(fp_file, fpr0_ER)
    np.save(tp_file, tpr0_ER)

    ct_pos_file = "%s%s_%s_ER_ct_flat%s" % (processed_data_dir, pdb_id, pfam_id, file_end)
    np.save(ct_pos_file, ct_pos_flat)


    #print('ER tp raw (len %d): ' % len(tpr0_ER), tpr0_ER[:10])
    #print('ER tp uni (len %d): ' % len(tpr_uni), tpr_uni[:10])   
    print('raw auc: %f, binned auc %f' % (auc, auc_uni))

    print('ER tp raw (len %d): ' % len(tpr0_ER), tpr0_ER[:10])
    print('ER tp uni (len %d): ' % len(tpr_uni), tpr_uni[:10])   


if 1:
    # MF ROC
    auc_MF = np.zeros(n)
    for i in range(n):
        try:
            fpr, tpr, thresholds, auc, ct_pos_flat = roc_curve_new(ct_mat, MF_di, ct_thres[i], s_index, ld_thresh=ld_threshold)
            auc_MF[i] = auc
        except:
            auc_MF[i] = 0
    
    # Get ER method's best contact prediction
    i0_MF = np.argmax(auc_MF)
    print('MF auc max:',ct_thres[i0_MF],auc_MF[i0_MF])
    fpr0_MF, tpr0_MF, thresholds_MF, auc, tpr_uni, fpr_uni, auc_uni, uni_bins, ct_pos_flat = roc_curve_new(ct_mat, MF_di, ct_thres[i0_MF], s_index, ld_thresh=ld_threshold, get_uniform=True)
    fp_uni_file = "%s%s_%s_MF_fp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    tp_uni_file = "%s%s_%s_MF_tp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    np.save(fp_uni_file, fpr_uni)
    np.save(tp_uni_file, tpr_uni)   

    fp_file = "%s%s_%s_MF_fp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    tp_file = "%s%s_%s_MF_tp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    np.save(fp_file, fpr0_MF)
    np.save(tp_file, tpr0_MF)
    print('MF tp raw (len %d): ' % len(tpr0_MF), tpr0_MF[:10])
    print('MF tp uni (len %d): ' % len(tpr_uni), tpr_uni[:10])   
    ct_pos_file = "%s%s_%s_MF_ct_flat%s" % (processed_data_dir, pdb_id, pfam_id, file_end)
    np.save(ct_pos_file, ct_pos_flat)

    print('raw auc: %f, binned auc %f' % (auc, auc_uni))
    
    
if 1:
    # PMF ROC
    # translate PMF di tuple to contact matrix
    PMF_di = np.zeros(ER_di.shape)
    PMF_di_dict = {}
    for score_set in PMF_di_data:
        PMF_di_dict[(score_set[0][0], score_set[0][1])] = score_set[1]
    for i, index_i in enumerate(s_index):
        for j, index_j in enumerate(s_index):
            if i==j:
                PMF_di[i,j] = 1.
                continue
            try:
                PMF_di[i,j] = PMF_di_dict[(index_i, index_j)]
                PMF_di[j,i] = PMF_di_dict[(index_i, index_j)] # symetric
            except(KeyError):
                continue
    
    
    auc_PMF = np.zeros(n)
    for i in range(n):
        try:
            fpr, tpr, thresholds, auc, ct_pos_flat = roc_curve_new(ct_mat, PMF_di, ct_thres[i], s_index, ld_thresh=ld_threshold)
            auc_PMF[i] = auc
        except:
            auc_PMF[i] = 0
    
    # Get ER method's best contact prediction
    i0_PMF = np.argmax(auc_PMF)
    print('PMF auc max:',ct_thres[i0_PMF],auc_PMF[i0_PMF])
    fpr0_PMF, tpr0_PMF, thresholds_PMF, auc, tpr_uni, fpr_uni, auc_uni, uni_bins, ct_pos_flat = roc_curve_new(ct_mat, PMF_di, ct_thres[i0_PMF], s_index, ld_thresh=ld_threshold, get_uniform=True)
    fp_uni_file = "%s%s_%s_PMF_fp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    tp_uni_file = "%s%s_%s_PMF_tp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    np.save(fp_uni_file, fpr_uni)
    np.save(tp_uni_file, tpr_uni)   

    fp_file = "%s%s_%s_PMF_fp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    tp_file = "%s%s_%s_PMF_tp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    np.save(fp_file, fpr0_PMF)
    np.save(tp_file, tpr0_PMF)
    print('PMF tp raw (len %d): ' % len(tpr0_PMF), tpr0_PMF[:10])
    print('PMF tp uni (len %d): ' % len(tpr_uni), tpr_uni[:10])   
    print('raw auc: %f, binned auc %f' % (auc, auc_uni))
    ct_pos_file = "%s%s_%s_PMF_ct_flat%s" % (processed_data_dir, pdb_id, pfam_id, file_end)
    np.save(ct_pos_file, ct_pos_flat)

   
    
if 1:
    # PLM ROC
    # translate PMF di tuple to contact matrix
    PLM_di = np.zeros(ER_di.shape)
    PLM_di_dict = {}
    for score_set in PLM_di_data:
        PLM_di_dict[(score_set[0][0], score_set[0][1])] = score_set[1]
    for i, index_i in enumerate(s_index):
        for j, index_j in enumerate(s_index):
            if i==j:
                PLM_di[i,j] = 1.
                continue
            try:
                PLM_di[i,j] = PLM_di_dict[(index_i, index_j)]
                PLM_di[j,i] = PLM_di_dict[(index_i, index_j)] # symetric
            except(KeyError):
                continue
    
    
    
    auc_PLM = np.zeros(n)
    for i in range(n):
        try:
            fpr, tpr, thresholds, auc, ct_pos_flat = roc_curve_new(ct_mat, PLM_di, ct_thres[i], s_index, ld_thresh=ld_threshold)
            auc_PLM[i] = auc
        except:
            auc_PLM[i] = 0
    
    # Get ER method's best contact prediction
    i0_PLM = np.argmax(auc_PLM)
    print('PLM auc max:',ct_thres[i0_PLM],auc_PLM[i0_PLM])
    fpr0_PLM, tpr0_PLM, thresholds_PLM, auc, tpr_uni, fpr_uni, auc_uni, uni_bins, ct_pos_flat = roc_curve_new(ct_mat, PLM_di, ct_thres[i0_PLM], s_index, ld_thresh=ld_threshold, get_uniform=True)
    fp_uni_file = "%s%s_%s_PLM_fp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    tp_uni_file = "%s%s_%s_PLM_tp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    np.save(fp_uni_file, fpr_uni)
    np.save(tp_uni_file, tpr_uni)   

    fp_file = "%s%s_%s_PLM_fp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    tp_file = "%s%s_%s_PLM_tp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    np.save(fp_file, fpr0_PLM)
    np.save(tp_file, tpr0_PLM)
    print('PLM tp raw (len %d): ' % len(tpr0_PLM), tpr0_PLM[:10])
    print('PLM tp uni (len %d): ' % len(tpr_uni), tpr_uni[:10])   
    print('raw auc: %f, binned auc %f' % (auc, auc_uni))
    ct_pos_file = "%s%s_%s_PLM_ct_flat%s" % (processed_data_dir, pdb_id, pfam_id, file_end)
    np.save(ct_pos_file, ct_pos_flat)

# save relevant data for categorizing contact prediction metrics
pfam_dimensions = [n_var, n_seq, meff, ct_thres[i0_ER],  ct_thres[i0_MF],  ct_thres[i0_PMF],  ct_thres[i0_PLM]]
pfam_dimensions_file = "%s%s_%s_pfam_dimensions%s" % (processed_data_dir, pdb_id, pfam_id, file_end)
np.save(pfam_dimensions_file, pfam_dimensions)



