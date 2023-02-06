import os, os.path, sys

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
# %matplotlib inline

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
chunk = sys.argv[1]
chunksx = [l[i:i + n] for i in range(0, 10000, 10)]
pdb_ids = np.load('best_pdb.npy')
pfam_ids = np.load('best_pfam.npy')
silhouette_scores = np.zeros(10)
for i,indx in enumerate(chunks[chunk]):
    print(pdb_ids[indx], pfam_ids[indx])
    pdb_id = pdb_ids[indx]
    
    prody_df = pd.read_csv('%s/%s_pdb_df.csv' % (pdb_dir, pdb_id))                                           
    pdb2msa_row = prody_df.iloc[0]
    pfam_id = pdb2msa_row['Pfam']
    if pfam_ids[indx] != pfam_id:
        print('Pfam ID from AUC_summary.ipynb generated datafram does not match Pfam ID from prody df')
        sys.exit(0)
    
    n_cpus = sys.argv[2]
    
    s0 = np.load("%s/%s_%s_preproc_msa.npy" % (processed_data_dir, pfam_id, pdb_id))                         
    s_index = np.load("%s/%s_%s_preproc_sindex.npy" % (processed_data_dir, pfam_id, pdb_id))                 
    #pdb_s_index = np.load("%s/%s_%s_preproc_pdb_sindex.npy" % (processed_data_dir, pfam_id, pdb_id))        
    removed_cols = np.load("%s/%s_%s_removed_cols.npy" % (processed_data_dir, pfam_id, pdb_id))              
    ref_seq = np.load("%s/%s_%s_preproc_refseq.npy" % (processed_data_dir, pfam_id, pdb_id))      
    
    onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
    print('s0: ',s0.shape,'\n',s0)
    onehot_encoder.fit(s0)
    s = onehot_encoder.transform(s0)
    print('s: ',s.shape,'\n',s)
    
    from sklearn.decomposition import PCA
    pca_dim=3
    
    pca = PCA(n_components = pca_dim)
    s_pca = pca.fit_transform(s)
    
    # Spectral Clustering of OneHot representation of MSA
    
    from sklearn.cluster import SpectralClustering
    
    clustering = SpectralClustering(n_clusters=2, random_state=0).fit(s)
    # https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c
    
    # Get Silhouette Coefficient
    # https://towardsdatascience.com/silhouette-coefficient-validating-clustering-techniques-e976bb81d10c
    from sklearn.metrics import silhouette_score
    sc = silhouette_score(s, clustering.labels_)
    print('Silhouette Score(n=2):', sc)
    silhouette_scores[i] = sc

print(silhouette_scores)
np.save('cluster_silhouette_scores_chunk%d.npy' % chunk, silhouette_scores)

