# Import system packages
import os.path, sys
import timeit
from pathlib import Path
from joblib import Parallel, delayed
import warnings

# import scientific computing packages
import numpy as np
np.random.seed(1)
from scipy.spatial import distance
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# import biopython packages
from Bio.PDB import *
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

# import pydca modules
import pydca
from pydca.plmdca import plmdca
from pydca.meanfield_dca import meanfield_dca
from pydca import sequence_backmapper
from pydca.msa_trimmer import msa_trimmer
from pydca.contact_visualizer import contact_visualizer
from pydca import dca_utilities
import os
import pandas as pd
import numpy as np

# # --- Import our Code ---# #
#import emachine as EM
from direct_info import direct_info

# import data processing and general DCA_ER tools
from data_processing import pdb2msa, data_processing
import ecc_tools as tools
from pathlib import Path

import matplotlib.backends.backend_pdf

colors_hex = {"red": "#e41a1c", "blue": "#2258A5", "green": "#349C55", "purple": "#984ea3", "orange": "#FF8B00",
                      "yellow": "#ffff33", "grey": "#BBBBBB"}
colors_key = ["blue", "orange", "green"]
method2color = {"ER":"blue", "MF":"orange", "PLM":"green"}



def plot_contact_map_example(ax2, pdb_id, pfam_id, n_seq, n_col, pairs_flat, ct_flat, dist_flat, ER_di_flat, ct_ER, ld_val=5):
    

        
    x_true_positives = []
    y_true_positives = []
    
    x_false_positives = []
    y_false_positives = []
    
    x_pdb = []
    y_pdb = []
    
    
    # define (i,j) pairs as top-L ranked di scores (L is length of protein) i > j (ie bottom triagular)
    counter = 0
    rank = len(s_index)
    for i, pair in enumerate(pairs_flat):
#         if pair[0] > pair[1]:
#             x = pair[0]
#             y = pair[1]
#         else:
#             x = pair[1]
#             y = pair[0]
        x=pair[0]
        y=pair[1]
        if abs(x-y)<ld_val:
            pass
        if dist_flat[i]<=ct_ER and x not in x_true_positives:
            x_true_positives.append(x)
            y_true_positives.append(y)
            counter += 1
        elif dist_flat[i]<=ct_ER and x in x_true_positives:
            pass
        elif dist_flat[i]>ct_ER and x not in x_false_positives:
            x_false_positives.append(x)
            y_false_positives.append(y)
            counter += 1
        else:
            pass
        if counter >= rank:
            break
    
    for i, pair in enumerate(pairs_flat):
        x=pair[0]
        y=pair[1]
        if abs(x-y)<ld_val:
            pass
        if dist_flat[i]<=ct_ER:
            x_pdb.append(x)
            y_pdb.append(y)
    
    
    ax_title = '''
    PDB ID: {} Pfam ID: {}
    Number of Sequences: {}
    Number of Columns: {}
    Maximum PDB contact distance : {} Angstrom
    Minimum residue chain distance: {} residues
    Fraction of true positives : {:.3g}
    '''.format(pdb_id, pfam_id, n_seq, n_col, ct_ER, ld_val,
        len(x_true_positives)/(len(x_true_positives) + len(x_false_positives)),
    )
    
    ax2.scatter(y_true_positives, x_true_positives, s=6, color='green')
    ax2.scatter(y_false_positives, x_false_positives, s=6, color='red')
    ax2.scatter(x_pdb, y_pdb, s=6, color='grey')
    ax2.set_xlabel('Residue Position', fontsize=14)
    ax2.set_ylabel('Residue Position', fontsize=14)
    ax2.set_title(ax_title)
    #ax2.set_title(ax_title)
#     # plt.savefig('%s_%s_%s_pydca_contact_map.pdf' % (pdb_id, pfam_id, method) )
#     if pdf_obj is not None:
#         pdf_obj.savefig( fig )
    return ax2    
    
    
def plot_di_vs_ct(ax, ct_flat, dist_flat, dis_flat, ld_flat, labels):
    colors = 'brg'
    for ii, di_flat in enumerate(dis_flat):
        fmethod = False # plot label for method 
        for i, contact in enumerate(ct_flat):
            if fmethod:
                ax.scatter(dist_flat[i], di_flat[i], marker='.',  c=colors_hex[colors_key[ii]], label=labels[ii])
                fmethod=False
            else:
                ax.scatter(dist_flat[i], di_flat[i], marker='.',  c=colors_hex[colors_key[ii]])

    ax.set_xlabel('Distance ($\AA$)', fontsize=14)
    ax.set_ylabel('DI', fontsize=14)
    return ax


# --------------------------------- Start simulation --------------------------------- #

data_path = Path('/data/cresswellclayec/DCA_ER/Pfam-A.full')
data_path = Path('/data/cresswellclayec/Pfam-A.full')

# Define data directories
DCA_ER_dir = '/data/cresswellclayec/DCA_ER' # Set DCA_ER directory
biowulf_dir = '%s/biowulf_full' % DCA_ER_dir

out_dir = '%s/protein_data/di/' % biowulf_dir
processed_data_dir = "%s/protein_data/data_processing_output" % biowulf_dir
pdb_dir = '%s/protein_data/pdb_data/' % biowulf_dir
contact_map_example_dir = '%s/manuscript_contact_map_examples' % DCA_ER_dir


pfam_dir = "/fdb/fastadb/pfam"

# PDB ID: 1zdr, Pfam ID: PF00186
pdb_id = sys.argv[1]
pfam_id = sys.argv[2]
#pdb_id = '1zdr'
#pfam_id = 'PF00186'



ref_outfile = Path(processed_data_dir, '%s_ref.fa' % pfam_id)
prody_df = pd.read_csv('%s/%s_pdb_df.csv' % (pdb_dir, pdb_id))
pdb2msa_row  = prody_df.iloc[0]
print('\n\nGetting msa with following pdb2msa entry:\n', pdb2msa_row)
#try:
print(pdb2msa_row)
pfam_id = pdb2msa_row['Pfam']
pdb_id = pdb2msa_row['PDB ID']
pdb_chain = pdb2msa_row['Chain']


pfam_dimensions_file = "%s/%s_%s_pfam_dimensions.npy" % (processed_data_dir, pdb_id, pfam_id)
pfam_dimensions = np.load(pfam_dimensions_file)
if len(pfam_dimensions)==7:
    [n_col, n_seq, m_eff, ct_ER, ct_MF, ct_PMF, ct_PLM] = pfam_dimensions
elif len(pfam_dimensions)==6: # new pfam_dimensions created in run_method_comparison. we dont need MF..
    [n_col, n_seq, m_eff, ct_ER, ct_PMF, ct_PLM] = pfam_dimensions
elif len(pfam_dimensions)==3:
    [n_col, n_seq, m_eff] = pfam_dimensions
# create output pdf object



# --- Load scores --- #
er_out_file = "%s/%s_%s_ER_di.npy" % (out_dir, pdb_id, pfam_id)
# ER scores need to be translated
ER_di = np.load(er_out_file)
s_index = np.load("%s/%s_%s_preproc_sindex.npy" % (processed_data_dir, pfam_id, pdb_id))
from ecc_tools import scores_matrix2dict
ER_scores = scores_matrix2dict(ER_di, s_index)
# ------------------- #
# load DI data
ER_di = np.load("%s/%s_%s_ER_di.npy" % (out_dir, pdb_id, pfam_id))

er_scores_ordered = {}
for [pair, score] in ER_scores:
    er_scores_ordered[pair] = score
ER_scores = er_scores_ordered
ER_scores = sorted(ER_scores.items(), key =lambda k : k[1], reverse=True)
for site_pair, score in ER_scores[:25]:
    print(site_pair, score)

ld_thresh = 5.
contact_dist = 10.

er_visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
        refseq_file = str(ref_outfile),
        sorted_dca_scores = ER_scores,
        linear_dist = ld_thresh,
        contact_dist = contact_dist,
)


ct_file = "%s%s_%s_ct.npy" % (pdb_dir, pdb_id, pfam_id)
ct = np.load(ct_file)

# Set contact distance
ct1 = ct.copy()
ct_pos = ct < contact_dist
ct1[ct_pos] = 1
ct1[~ct_pos] = 0


mask = np.triu(np.ones(ER_di.shape[0], dtype=bool), k=1)
# argsort sorts from low to high. [::-1] reverses 
er_order = ER_di[mask].argsort()[::-1]

linear_distance = np.zeros((len(s_index),len(s_index)))                                                                                                   
for i, ii in enumerate(s_index): 
    for j, jj in enumerate(s_index):
        linear_distance[i,j] = abs(ii - jj)   

ld = linear_distance >= ld_thresh                                                                                                                         
ER_ld_flat = ld[mask][er_order]          
ER_di_flat = ER_di[mask][er_order]
ER_ct_flat = ct1[mask][er_order]
ER_dist_flat = ct[mask][er_order]


pairs_matrix = np.empty((len(s_index),len(s_index)), dtype=object)                                                               
for i, ii in enumerate(s_index):
    for j, jj in enumerate(s_index):
        pairs_matrix[i,j] =(i,j)
print(pairs_matrix)
print(pairs_matrix.shape)

ER_pairs_flat = pairs_matrix[mask][er_order]


fig = plt.figure(figsize=(5,6))
ax = plt.subplot2grid((1,1),(0,0))
ax = plot_contact_map_example(ax, pdb_id, pfam_id, n_col, n_seq, ER_pairs_flat, ER_ct_flat, ER_dist_flat, ER_di_flat, contact_dist, ld_val=ld_thresh)

ax.legend()
plt.tight_layout()
plt.savefig('%s/%dseq_%scol_%s_%s_ER_contact_map.pdf' % (contact_map_example_dir, int(n_seq), int(n_col), pdb_id, pfam_id))


