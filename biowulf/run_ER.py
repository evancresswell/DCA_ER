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
from data_processing import data_processing
import ecc_tools as tools
from pathlib import Path
np.random.seed(1)

data_path = Path('/data/cresswellclayec/DCA_ER/Pfam-A.full/')


pfam_id = sys.argv[1]
n_jobs = int(sys.argv[2])
print('Finding MF contacts for %s', pfam_id)


DCA_ER_dir = '/data/cresswellclayec/DCA_ER/'
msa_npy_file = '/data/cresswellclayec/DCA_ER/Pfam-A.full/%s/msa.npy' % pfam_id
msa_fa_file  = '/data/cresswellclayec/DCA_ER/Pfam-A.full/%s/msa.fa' % pfam_id
pdb_ref_file = '/data/cresswellclayec/DCA_ER/Pfam-A.full/%s/pdb_refs.npy' % pfam_id
out_dir = '%sprotein_data/di/' % DCA_ER_dir
processed_data_dir = "%s/protein_data/data_processing_output" % DCA_ER_dir


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Set DCA_ER directory

# Define data directories
# Need to think on best way to do this..
# Referencing the same dataframe may be useful so we dont always have to load individual ref files...
# however we also
individual_pdb_ref_file = Path(data_path, pfam_id, 'pdb_refs.npy')
pdb = np.load(individual_pdb_ref_file)

# delete 'b' in front of letters (python 2 --> python 3)
pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0]) \
         for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])


# Print number of pdb structures in Protein ID folder
npdb = pdb.shape[0]
print('number of pdb structures:',npdb)

# Create pandas dataframe for protein structure
pdb_df = pd.DataFrame(pdb,columns = ['PF','seq','id','uniprot_start','uniprot_end',\
                                 'pdb_id','chain','pdb_start','pdb_end'])
pdb_df.head()

ipdb = 0
printing = True
print('seq:',int(pdb[ipdb,1]))



#s0,cols_removed, s_index, tpdb, orig_seq_len = data_processing(data_path, pfam_id, ipdb,\
#                gap_seqs=0.2, gap_cols=0.2, prob_low=0.004, conserved_cols=0.9, printing=printing, out_dir=processed_data_dir)
#print('\n\n\nPreprocessed reference Sequence: ', s0[tpdb])
#
# npy2fa does not remove cols this way we are as close to original as possible
msa_outfile, ref_outfile, s0, cols_removed, s_index, tpdb, orig_seq_len  = tools.npy2fa(pfam_id, msa_npy_file, pdb_ref_file=pdb_ref_file, ipdb=ipdb, preprocess=True,gap_seqs=.2, gap_cols=.2, prob_low=.004, conserved_cols=.9, letter_format=False)

# save processed data
np.save('%s/%s_ER_s0.npy' 		% (processed_data_dir, pfam_id), s0)
np.save('%s/%s_ER_cols_removed.npy' 	% (processed_data_dir, pfam_id), cols_removed) 
np.save('%s/%s_ER_s_index.npy' 		% (processed_data_dir, pfam_id), s_index)
np.save('%s/%s_ER_tpdb.npy' 		% (processed_data_dir, pfam_id), tpdb)
np.save('%s/%s_ER_original_seq_len.npy' % (processed_data_dir, pfam_id), orig_seq_len)

# data processing
#print(s0[1934])
#print(s0[3522])
print(s0.shape, '\n\n\n')

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# # Load pydca trimmed datrimmed_data_outfile = '%sprotein_data/data_processing_output/MSA_%s_Trimmed.fa' % (DCA_ER_dir, pfam_id)
# trimmed_msa_file = Path(DCA_ER_dir, 'protein_data/data_processing_output/MSA_%s_Trimmed.fa' % pfam_id)
# s0_pydca = tools.read_FASTA(str(trimmed_msa_file), ref_index = int(pdb[ipdb,1]))


# We don't need to generate contact map while calculating DI on Biowulf
if 0:
    # Load csv of PDB-PFAM mapping.
    #    downloaded from 
    #    ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_pfam_mapping.csv.gz
    pdb_id = pdb_df.iloc[ipdb]['pdb_id']
    pdb_chain = pdb_df.iloc[ipdb]['chain']

    pdb_pfam_map_file = Path('%s/protein_data/pdb_data/pdb_pfam_mapping.csv' % DCA_ER_dir)
    pdb_map_df = pd.read_csv(pdb_pfam_map_file, sep=',', header=1)
    print(pdb_map_df.head())

    pdb_id_map_df = pdb_map_df.loc[pdb_map_df['PDB']==pdb_id.lower()]
    pdb_pfam_map = pdb_id_map_df.loc[pdb_id_map_df['CHAIN']==pdb_chain]

    ## Generate and Plot Contact Map from PDB coordinates!
    # Check that pdb--pfam mapping is unique
    if pdb_pfam_map.shape[0] > 1:
        print('Unable to get unique PDB-->Pfam mapping')
        print(pdb_pfam_map)

    # pp_range = [pdb_info['PDB_START'], pdb_info['PDB_END']]
    pp_range = [pdb_pfam_map.iloc[0]['PDB_START'], pdb_pfam_map.iloc[0]['PDB_END']]
    print('Polypeptide range for contact map: ', pp_range)


    pdb_out = "%s/protein_data/pdb_data" % DCA_ER_dir
    # Directory for storing PDB data locally
    # returns contact map with the appropriate columns removed..
    # For list of retained columns us s_index
    ct, ct_full, n_amino_full, poly_seq_curated = tools.contact_map(pdb, ipdb, pp_range, cols_removed, s_index, pdb_out_dir=pdb_out, printing=printing)
    print(ct.shape)
    print(ct_full.shape)


    ct_mat = ct

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

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

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
# s is OneHot encoder format, s0 is original sequnce matrix
s = onehot_encoder.fit_transform(s0)
# print("Amino Acid sequence Matrix\n",s0)
# print("OneHot sequence Matrix\n",s)
# print("An individual element of the OneHot sequence Matrix (size:",
#      s.shape,") --> ",s[0], " has length ",s[0].shape)a


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# Define wight matrix with variable for each possible amino acid at each sequence position
w = np.zeros((mx.sum(),mx.sum())) 
h0 = np.zeros(mx.sum())


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
print(di)
print(di.shape)
print(len(s_index))

print(di.shape)
ER_di = np.delete(di, cols_removed,0)
ER_di = np.delete(ER_di, cols_removed,1)

print(ER_di.shape)

er_file = "%s/%s_ER_di.npy" % (out_dir, pfam_id)
np.save(er_file, ER_di)

print('Complete...')
