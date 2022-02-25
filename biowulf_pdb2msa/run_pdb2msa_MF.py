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
from Bio.SubsMat.MatrixInfo import blosum62
pdb_parser = Bio.PDB.PDBParser()

from prody import *



create_new = True
printing = True
removing_cols = True


data_path = Path('/data/cresswellclayec/DCA_ER/Pfam-A.full')
data_path = Path('/data/cresswellclayec/Pfam-A.full')

# Define data directories
DCA_ER_dir = '/data/cresswellclayec/DCA_ER' # Set DCA_ER directory
biowulf_dir = '%s/biowulf_pdb2msa' % DCA_ER_dir

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


for ir, pdb2msa_row in enumerate(prody_df.iterrows()):
    print('\n\nGetting msa with following pdb2msa entry:\n', pdb2msa_row)
    try:
        dp_result =  data_processing_pdb2msa(data_path, prody_df.iloc[pdb2msa_row[0]], gap_seqs=0.2, gap_cols=0.2, prob_low=0.004,
                               conserved_cols=0.8, printing=True, out_dir=processed_data_dir, pdb_dir=pdb_dir, letter_format=False,
                               remove_cols=True, create_new=True, n_cpu=min(2, n_cpus))
        if dp_result is not None:
            [s0, removed_cols, s_index, tpdb] = dp_result
            break
        else: 
            continue
    except Exception as e:
        print('row %d got exception: ' % ir , e)
        print('moving on.. ')
        pass

print('Done Preprocessing Data.....')

pfam_id = pdb2msa_row[1]['Pfam']
pdb_id = pdb2msa_row[1]['PDB ID']
pdb_chain = pdb2msa_row[1]['Chain']

# number of positions
n_var = s0.shape[1]
n_seq = s0.shape[0]
print("Number of residue positions:",n_var)
print("Number of sequences:",n_seq)

print(s0.shape, '\n\n\n')

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
print(pdb2msa_row[1])
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
from inference_dca import direct_info_dca
print('Final MSA shape before MF calculation: ', s0.shape)
seq_wt_file = None
seq_wt_file = '%s/seq_weight_%s_%s.npy' % (processed_data_dir, pdb_id, pfam_id)


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

mf_file = "%s/%s_%s_MF_di.npy" % (out_dir, pdb_id, pfam_id)
np.save(mf_file, mf_di)
print('Complete...')
