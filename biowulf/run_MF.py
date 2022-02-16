import os.path, sys

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder

import Bio.PDB, warnings
from Bio import SeqIO

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
from prody import *
from data_processing import data_processing_new, pdb2msa 
import ecc_tools as tools
from pathlib import Path
np.random.seed(1)

data_path = Path('/data/cresswellclayec/DCA_ER/Pfam-A.full')


pfam_id = sys.argv[1]
n_jobs = sys.argv[2]
print('Finding MF contacts for %s', pfam_id)


create_new = False
printing = True
removing_cols = True
preprocessing = False



# Define data directories
DCA_ER_dir = '/data/cresswellclayec/DCA_ER' # Set DCA_ER directory
biowulf_dir = '%s/biowulf' % DCA_ER_dir
msa_npy_file = '%s/%s/msa.npy' % (str(data_path), pfam_id)
msa_fa_file  = '%s/%s/msa.fa' %  (str(data_path), pfam_id)
pdb_ref_file = '%s/%s/pdb_refs.npy' %  (str(data_path), pfam_id)

out_dir = '%s/protein_data/di/' % biowulf_dir 
processed_data_dir = "%s/protein_data/data_processing_output" % biowulf_dir
pdb_dir = '%s/protein_data/pdb_data/' % biowulf_dir 


print ('\n\n\ntesting pdb2msa')
pf00186_path = "/pdb/pdb/zd/pdb1zdr.ent.gz"
import gzip, shutil
def gunzip(file_path,output_path):
    with gzip.open(file_path,"rb") as f_in, open(output_path,"wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
unzipped_pdb_path = os.path.basename(pf00186_path).replace(".gz", "")
pdb_path = "%s%s" % (pdb_dir, unzipped_pdb_path)
print(pdb_path)
gunzip(pf00186_path, pdb_path)

pdb2msa(pdb_path)
print ('\n\n\n')

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
                        remove_cols=removing_cols, create_new=create_new)


pdb_chain, ct, ct_full, n_amino_full, poly_seq_curated, poly_seq_range, poly_seq, pp_ca_coords_curated, pp_ca_coords_full_range \
= tools.contact_map_new(pdb_id=pdb_select['PDB ID'][:4], pdb_range=[pdb_select['Subject Beg'], pdb_select['Subject End']], \
                  removed_cols=curated_cols, queried_seq=pdb_select['Subject Aligned Seq'],  pdb_out_dir=pdb_dir)

# --------------------------------- Process Data and get Contact Map ------------------------------------------------------------------------------------------------------- #


print(s0.shape, '\n\n\n')

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
print(pdb_chain)
print(pdb_select)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
from inference_dca import direct_info_dca
print('Final MSA shape before MF calculation: ', s0.shape)
seq_wt_file = None
seq_wt_file = '%s/seq_weight_%s.npy' % (processed_data_dir, pfam_id)


mf_di, fi, fij, c, cinv, w, w2d, fi_pydca, fij_pydca, c_pydca, c_inv_pydca, \
w_pydca, w2d_pydcak, di_pydca, ma_inv,seq_ints\
= direct_info_dca(s0, seq_wt_outfile=seq_wt_file)

print(c_pydca[0])
print(c[0])
diff = c_pydca - c
print(diff)
print('difference between c_pydca and c: ', np.linalg.norm(diff))


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
if 0: # we can just do this in postprocessing
    if not removing_cols:
        MF_di = np.delete(mf_di, cols_removed,0)
        MF_di = np.delete(MF_di, cols_removed,1)

mf_file = "%s/%s_MF_di.npy" % (out_dir, pfam_id)
np.save(mf_file, mf_di)
print('Complete...')
