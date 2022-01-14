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
n_jobs = sys.argv[2]
print('Finding MF contacts for %s', pfam_id)

create_new = False


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

# npy2fa does not remove cols this way we are as close to original as possible
msa_outfile, ref_outfile, s0, cols_removed, s_index, tpdb, orig_seq_len  = tools.npy2fa(pfam_id, msa_npy_file, pdb_ref_file=pdb_ref_file, ipdb=ipdb, preprocess=True,gap_seqs=.2, gap_cols=.2, prob_low=.004, conserved_cols=.9, letter_format=False)


print('\n\n\nPreprocessed reference Sequence: ', s0[tpdb])
# save processed data
np.save('%s/%s_MF_s0.npy' 		% (processed_data_dir, pfam_id), s0)
np.save('%s/%s_MF_cols_removed.npy' 	% (processed_data_dir, pfam_id), cols_removed) 
np.save('%s/%s_MF_s_index.npy' 		% (processed_data_dir, pfam_id), s_index)
np.save('%s/%s_MF_tpdb.npy' 		% (processed_data_dir, pfam_id), tpdb)
np.save('%s/%s_MF_original_seq_len.npy' % (processed_data_dir, pfam_id), orig_seq_len)

print(s0.shape, '\n\n\n')

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# # Load pydca trimmed datrimmed_data_outfile = '%sprotein_data/data_processing_output/MSA_%s_Trimmed.fa' % (DCA_ER_dir, pfam_id)
# trimmed_msa_file = Path(DCA_ER_dir, 'protein_data/data_processing_output/MSA_%s_Trimmed.fa' % pfam_id)
# s0_pydca = tools.read_FASTA(str(trimmed_msa_file), ref_index = int(pdb[ipdb,1]))

# Load csv of PDB-PFAM mapping.
#    downloaded from 
#    ftp://ftp.ebi.ac.uk/pub/databases/msd/sifts/flatfiles/csv/pdb_pfam_mapping.csv.gz

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
from inference_dca import direct_info_dca
print('Final MSA shape before MF calculation: ', s0.shape)
seq_wt_file = None
seq_wt_file = '%s/seq_weight_%s.npy' % (processed_data_dir, pfam_id)


mf_di, fi, fij, c, cinv, w, w2d, fi_pydca, fij_pydca, c_pydca, c_inv_pydca, \
w_pydca, w2d_pydcak, di_pydca, ma_inv,seq_ints\
= direct_info_dca(s0, seq_wt_outfile=seq_wt_file)



# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
MF_di = np.delete(mf_di, cols_removed,0)
MF_di = np.delete(MF_di, cols_removed,1)

mf_file = "%s/%s_MF_di.npy" % (out_dir, pfam_id)
np.save(mf_file, MF_di)
print('Complete...')
