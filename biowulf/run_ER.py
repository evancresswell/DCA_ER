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
from data_processing import data_processing_new
import ecc_tools as tools
from pathlib import Path
np.random.seed(1)

data_path = Path('/data/cresswellclayec/DCA_ER/Pfam-A.full/')


pfam_id = sys.argv[1]
n_jobs = int(sys.argv[2])
print('Finding ER contacts for %s', pfam_id)

create_new = True
removing_cols = True


# Define data directories
DCA_ER_dir = '/data/cresswellclayec/DCA_ER' # Set DCA_ER directory
biowulf_dir = '%s/biowulf' % DCA_ER_dir
msa_npy_file = '%s/%s/msa.npy' % (str(data_path), pfam_id)
msa_fa_file  = '%s/%s/msa.fa' %  (str(data_path), pfam_id)
pdb_ref_file = '%s/%s/pdb_refs.npy' %  (str(data_path), pfam_id)

out_dir = '%s/protein_data/di/' % biowulf_dir 
processed_data_dir = "%s/protein_data/data_processing_output" % biowulf_dir
pdb_dir = '%s/protein_data/pdb_data/' % biowulf_dir 


# ------------------------- Get PDB reference Data from Uniprot/Pfam ------------------------------------------------------------------------------------------------------- #

# Referencing the same dataframe may be useful so we dont always have to load individual ref files...
# however we also
individual_pdb_ref_file = Path(data_path, pfam_id, 'pdb_refs.npy')
pdb = np.load(individual_pdb_ref_file)

try: 
    # delete 'b' in front of letters (python 2 --> python 3)
    pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0]) \
         for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])
    
    
    # Print number of pdb structures in Protein ID folder
    npdb = pdb.shape[0]
    print('number of pdb structures:',npdb)
    
    # Create pandas dataframe for protein structure
    pdb_df = pd.DataFrame(pdb,columns = ['PF','seq','id','uniprot_start','uniprot_end',\
                                     'pdb_id','chain','pdb_start','pdb_end'])
    print(pdb_df.head())
    pdb_reference_ids = np.unique(pdb_df['pdb_id'].to_numpy())
    print('PDB reference IDs:\n', pdb_reference_ids)
    pdb_ref_ipdb = 0
    print('seq:',int(pdb[pdb_ref_ipdb,1]))



except(IndexError):
    print('Loaded pdb: ', pdb)
    print('\n\nEMPTY PDB REFERENCE!!!\n\n')
    pdb_reference_ids = None


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
s0, curated_cols, s_index, tpdb, pdb_select \
= data_processing_new(data_path, pfam_id, index_pdb=0,gap_seqs=0.2, gap_cols=0.2, prob_low=0.004, 
                        conserved_cols=0.9, printing=True, out_dir=processed_data_dir, pdb_dir=pdb_dir,  letter_format=False, 
                        remove_cols=removing_cols, create_new=create_new, n_cpu=n_jobs)


pdb_chain, ct, ct_full, n_amino_full, poly_seq_curated, poly_seq_range, poly_seq, pp_ca_coords_curated, pp_ca_coords_full_range \
= tools.contact_map_new(pdb_id=pdb_select['PDB ID'][:4], pdb_range=[pdb_select['Subject Beg'], pdb_select['Subject End']], \
                  removed_cols=curated_cols, queried_seq=pdb_select['Subject Aligned Seq'],  pdb_out_dir=pdb_dir)

# --------------------------------- Process Data and get Contact Map ------------------------------------------------------------------------------------------------------- #

print('\n\n\nPreprocessed reference Sequence: ', s0[tpdb])
# save processed data
np.save('%s/%s_ER_s0.npy' 		% (processed_data_dir, pfam_id), s0)
np.save('%s/%s_ER_curated_cols.npy' 	% (processed_data_dir, pfam_id), curated_cols) 
np.save('%s/%s_ER_s_index.npy' 		% (processed_data_dir, pfam_id), s_index)
np.save('%s/%s_ER_tpdb.npy' 		% (processed_data_dir, pfam_id), tpdb)

print(s0.shape, '\n\n\n')

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
print(pdb_chain)
print(pdb_select)
# -------------------------------- Run ER Method --------------------------------------------------------------------------------------------------------------------------- #

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
#      s.shape,") --> ",s[0], " has length ",s[0].shape)a



# Define wight matrix with variable for each possible amino acid at each sequence position
w = np.zeros((mx.sum(),mx.sum())) 
h0 = np.zeros(mx.sum())


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

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
#-------------------------------
# parallel
start_time = timeit.default_timer()
#res = Parallel(n_jobs = 4)(delayed(predict_w)\
#res = Parallel(n_jobs = 8)(delayed(predict_w)\
res = Parallel(n_jobs = n_jobs)(delayed(predict_w)\
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



#print w is symmetric (sanity test)
print("Dimensions of w: ",w.shape)
w_file = "%s/%s_w.npy" % (processed_data_dir, pfam_id)
np.save(w_file, w)



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

di = direct_info(s0,w)
# print(di)
print('s_index length: ', len(s_index))
print('di shape: ', di.shape)

if 0: # we can just do this in postprocessing
    if not removing_cols:
        ER_di = np.delete(di, curated_cols,0)
        ER_di = np.delete(ER_di, curated_cols,1)

print(di.shape)

er_file = "%s/%s_ER_di.npy" % (out_dir, pfam_id)
np.save(er_file, di)

print('Complete...')
