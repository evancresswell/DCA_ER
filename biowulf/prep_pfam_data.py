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

data_path = Path('/data/cresswellclayec/DCA_ER/Pfam-A.full')


pfam_id = sys.argv[1]

create_new = True
printing = True
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



# ------------------------------------------ Pre-Process Data for ER/MF ---------------------------------------------------------------------------------------------------- #

start_time = timeit.default_timer()

s0, curated_cols, s_index, tpdb, pdb_select \
= data_processing_new(data_path, pfam_id, index_pdb=0,gap_seqs=0.2, gap_cols=0.2, prob_low=0.004, 
                        conserved_cols=0.9, printing=True, out_dir=processed_data_dir, pdb_dir=pdb_dir,  letter_format=False, 
                        remove_cols=removing_cols, create_new=create_new)

print('selected PDB structure:\n', pdb_select, '\n\n')
if 1:
    sys.exit()

print('Getting PDB sequence and contact info from queried PDB strucuture')
pdb_chain, ct, ct_full, n_amino_full, poly_seq_curated, poly_seq_range, poly_seq, pp_ca_coords_curated, pp_ca_coords_full_range \
= tools.contact_map_new(pdb_id=pdb_select['PDB ID'][:4], pdb_range=[pdb_select['Subject Beg'], pdb_select['Subject End']], \
                  removed_cols=curated_cols, queried_seq=pdb_select['Subject Aligned Seq'],  mismatches=pdb_select['Mismatches'] , pdb_out_dir=pdb_dir)

run_time = timeit.default_timer() - start_time
print('Pre-Process run time:',run_time)

pdb_id = pdb_select['PDB ID'][:4]

print('\n\n\nPreprocessed reference Sequence: ', s0[tpdb])

print(s0.shape, '\n\n\n')
print(pdb_chain)
print(pdb_select)

pfam_reference_data = np.array([ tpdb, pdb_select['PDB ID'][:4], pdb_chain, pdb_select['Subject Beg'], pdb_select['Subject End'], pdb_select['Subject Aligned Seq']])
np.save('%s/%s_reference_data.npy' % (processed_data_dir, pfam_id), pfam_reference_data)
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------ Pre-Process Data for ER/MF ---------------------------------------------------------------------------------------------------- #
start_time = timeit.default_timer()
msa_outfile, ref_outfile, s, pdb_select, pdb_chain, tpdb = tools.npy2fa_new(data_path, pfam_id, pdb_data_dir=pdb_dir, index_pdb=0, create_new=create_new, processed_data_dir=processed_data_dir)
run_time = timeit.default_timer() - start_time
print('PYDCA-Prep run time:',run_time)


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
pdb_id_pydca = pdb_select['PDB ID'][:4]


