import os.path, sys
import pydca
from pydca.plmdca import plmdca
from pydca import sequence_backmapper
from pydca.msa_trimmer import msa_trimmer
from pydca.contact_visualizer import contact_visualizer
from pydca import dca_utilities

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
if len(sys.argv[1]) > 4:
    pdb_path = sys.argv[1]
else:
    pdb_id = sys.argv[1]
    pdb_path = "/pdb/pdb/%s/pdb%s.ent.gz" % (pdb_id[1:3], pdb_id) 

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

if 0:
    for ir, pdb2msa_row in enumerate(prody_df.iterrows()):
        print('\n\nGetting msa with following pdb2msa entry:\n', pdb2msa_row)
        #try:
        dp_result =  data_processing_pdb2msa(data_path, prody_df.iloc[pdb2msa_row[0]], gap_seqs=0.2, gap_cols=0.2, prob_low=0.004,
                                   conserved_cols=0.8, printing=True, out_dir=processed_data_dir, pdb_dir=pdb_dir, letter_format=False,
                                   remove_cols=True, create_new=True, n_cpu=min(2, n_cpus))
        if dp_result is not None:
            [s0, removed_cols, s_index, tpdb, pdb_s_index] = dp_result
            break
        else: 
            continue
        #except Exception as e:
        #    print('row %d got exception: ' % ir , e)
        #    print('moving on.. ')
        #    pass
else:
    # since pdb2msa link was already found with ER run we just need to take it and process for PYDCA
    pdb2msa_row  = prody_df.iloc[0]
    print('\n\nGetting msa with following pdb2msa entry:\n', pdb2msa_row)
    #try:
    print(pdb2msa_row)
    pfam_id = pdb2msa_row['Pfam']
    pdb_id = pdb2msa_row['PDB ID']

    msa_outfile, ref_outfile, s, tpdb, removed_cols, s_index, pdb_s_index = tools.npy2fa_pdb2msa(data_path, pdb2msa_row, pdb_dir, index_pdb=0, n_cpu=4, create_new=True, processed_data_dir=processed_data_dir) # letter_format is True

print('Done Preprocessing Data.....')


# PYDCA PLM

# create MSATrimmer instance 
trimmer = msa_trimmer.MSATrimmer(
    str(msa_outfile), biomolecule='protein', 
    refseq_file=str(ref_outfile),
)

trimmed_data = trimmer.get_msa_trimmed_by_refseq(remove_all_gaps=True)

#write trimmed msa to file in FASTA format
trimmed_data_outfile = '%s/MSA_%s_trimmed.fa' % (processed_data_dir, pfam_id)
with open(trimmed_data_outfile, 'w') as fh:
    for seqid, seq in trimmed_data:
        fh.write('>{}\n{}\n'.format(seqid, seq))
 
# Compute DCA scores using Pseudolikelihood maximization algorithm

plmdca_inst = plmdca.PlmDCA(
    trimmed_data_outfile,
    'protein',
    seqid = 0.8,
    lambda_h = 1.0,
    lambda_J = 20.0,
    num_threads =n_cpus,
    max_iterations = 500,
)

# SAVE PYDCA MF
plm_out_file = "%s/%s_%s_PLM_di.npy" % (out_dir, pdb_id, pfam_id)


# compute MF DCA scores summarized by various methods:
# FN_APC: Frobenius norm and average product corrected
# DI_APC: raw DI with average product correction
# DI: raw DI score
# FN: Frobenius norm of raw DI
method = "DI"
if method == "DI_APC":
    plmdca_scores = plmdca_inst.compute_sorted_DI_APC()
if method == "DI":
    plmdca_scores = plmdca_inst.compute_sorted_DI()
if method == "FN_APC":
    plmdca_scores = plmdca_inst.compute_sorted_FN_APC()
if method == "FN":
    plmdca_scores = plmdca_inst.compute_sorted_FN()


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

for site_pair, score in plmdca_scores[:5]:
    print(site_pair, score)

   
# Save the scores for future use/comparison
np.save(plm_out_file, plmdca_scores)

# save processed data
np.save('%s/%s_pydca_PLM_tpdb.npy' 		% (processed_data_dir, pfam_id), tpdb)

print('Complete...')

