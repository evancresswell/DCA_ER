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


create_new = False
printing = True
removing_cols = True
remove_diagonals = False

data_path = Path('/data/cresswellclayec/DCA_ER/Pfam-A.full')
data_path = Path('/data/cresswellclayec/Pfam-A.full')

# Define data directories
DCA_ER_dir = '/data/cresswellclayec/DCA_ER' # Set DCA_ER directory
biowulf_dir = '%s/biowulf_pdb2msa' % DCA_ER_dir


out_dir = '%s/protein_data/di/' % biowulf_dir
out_metric_dir = '%s/protein_data/metrics/' % biowulf_dir

processed_data_dir = "%s/protein_data/data_processing_output" % biowulf_dir
pdb_dir = '%s/protein_data/pdb_data/' % biowulf_dir

# pdb_path = "/pdb/pdb/zd/pdb1zdr.ent.gz"
pdb_path = sys.argv[1]
n_cpus = int(sys.argv[2])
print('\n\nUnzipping %s' % pdb_path)

unzipped_pdb_filename = os.path.basename(pdb_path).replace(".gz", "")

pdb_out_path = "%s%s" % (pdb_dir, unzipped_pdb_filename)
print('Unzipping %s to %s' % (pdb_path, pdb_out_path))



from data_processing import pdb2msa, data_processing_pdb2msa

# --------------------- Data Processing (should be saving correct row!!!!) --- #
import gzip, shutil
def gunzip(file_path,output_path):
    with gzip.open(file_path,"rb") as f_in, open(output_path,"wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

        
unzipped_pdb_filename = os.path.basename(pdb_path).replace(".gz", "")

pdb_out_path = "%s%s" % (pdb_dir, unzipped_pdb_filename)
print('Unzipping %s to %s' % (pdb_path, pdb_out_path))

gunzip(pdb_path, pdb_out_path)


print(pdb_out_path)
print(pdb_dir)
pdb2msa_results = pdb2msa(pdb_out_path, pdb_dir, create_new=True)
print(pdb2msa_results)

if len(pdb2msa_results) > 1:
    fasta_file = pdb2msa_results[0]
    prody_df = pdb2msa_results[1]
else:
    prody_df = pdb2msa_results[0]


# --------------------- Data Processing (should be saving correct row!!!!) --- #
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
pfam_id = pdb2msa_row[1]['Pfam']
pdb_id = pdb2msa_row[1]['PDB ID']

print('Done found correct prody_df row for contact predictions ... NOTE\n this step should be saved during data_processing!!!...')
# ---------------------------------------------------------------------------- #




di = np.load("%s/%s_%s_ER_di.npy" % (out_dir, pdb_id, pfam_id))

ct, ct_full, n_amino_full, poly_seq_curated, poly_seq_range, poly_seq, pp_ca_coords_curated, pp_ca_coords_full_range =  \
tools.contact_map_pdb2msa(pdb2msa_row[1], pdb_out_path, removed_cols, pdb_out_dir=pdb_dir, printing=True)

#print("Direct Information from Expectation reflection:\n",di)
print('ER DI shape: ' , di.shape)
print(removed_cols)
if not removing_cols:
    er_di = np.delete(di, removed_cols,0)
    er_di = np.delete(er_di, removed_cols,1)
else:
    er_di = di

print('Final ER DI shape (cols removed): ', er_di.shape)
if remove_diagonals: 
    ER_di = no_diag(er_di, 4, s_index)
else:
    ER_di = er_di

from ecc_tools import scores_matrix2dict
print(s_index)
ER_di_dict = scores_matrix2dict(ER_di, s_index, removed_cols, removing_cols=removing_cols)
print(ER_di_dict[:20])


from ecc_tools import roc_curve, roc_curve_new, precision_curve
# find optimal threshold of distance
ct_thres = np.linspace(4.,6.,18,endpoint=True)
n = ct_thres.shape[0]

ct_mat = ct
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

fp_file = "%s%s_%s_ER_fp.npy" % (out_metric_dir, pdb_id, pfam_id)
tp_file = "%s%s_%s_MF_tp.npy" % (out_metric_dir, pdb_id, pfam_id)
np.save(fp_file, fpr0_ER)
np.save(tp_file, tpr0_ER)
