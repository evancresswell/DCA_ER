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



def plot_di_compare_methods(ax, ct_flat, di1_flat, di2_flat, ld_flat, labels):
    ec = 'b'
    f1 = True
    f2 = True
    f3 = True
    for j in range(len(ct_flat), -1, -1):
        i = j-1
        contact = ct_flat[i]
        if not ld_flat[i]:
            if f1:
                ax.scatter(di1_flat[i], di2_flat[i], marker='x',  c='k', alpha=.2, label='to close')
                f1 = False
            else:
                ax.scatter(di1_flat[i], di2_flat[i],  marker='x',  c='k', alpha=.2)
    
        elif contact==1.:
            if f2:
                ax.scatter(di1_flat[i], di2_flat[i], marker='o', facecolors='none', edgecolors='g', label='contact')
                f2=False
            else:
                ax.scatter(di1_flat[i], di2_flat[i],  marker='o', facecolors='none', edgecolors='g')
        else:
            if f3:
                ax.scatter(di1_flat[i], di2_flat[i], marker='_', c='r', label='no contact')
                f3 = False
            else:
                ax.scatter(di1_flat[i], di2_flat[i], marker='_', c='r')
    ax.set_xlabel('%s DI' % labels[0], fontsize=14)
    ax.set_ylabel('%s DI' % labels[1], fontsize=14)
    return ax


def pydca_tp_plot(method_visualizers, methods, pdf_obj, ld=4, contact_dist=10.):

    fig = plt.figure(figsize=(5,5))
    ax1 = plt.subplot2grid((1,1), (0,0))

   
    # Plot ER results
    if len(method_visualizers) > 1:
        for i, mv in enumerate(method_visualizers):
            true_positive_rates_dict = mv.compute_true_positive_rates()
            tpr = true_positive_rates_dict['dca']
            pdb_tpr = true_positive_rates_dict['pdb']
            max_rank = len(tpr)
            ranks = [i + 1 for i in range(max_rank)]
            
                        
            ax1.plot(ranks, tpr, label=methods[i], color=colors_hex[colors_key[i]])
            if i == 0:
                ax1.plot(ranks, pdb_tpr,color='k')
            ax1.set_xscale('log')
            ax_title = '''
            True Positive Rate Per Rank
            PDB cut-off distance : {} Angstrom
            Residue chain distance : {}
            '''
            #ax.set_title(ax_title.format(self.__contact_dist, self.__linear_dist,))
            ax1.set_xlabel('Rank', fontsize=14)
            ax1.set_ylabel('True Positives/Rank', fontsize=14)
            plt.legend()
            plt.grid()
            plt.tight_layout()
        #plt.savefig('%s_%s_pydca_tp_rate.pdf' % (pdb_id, pfam_id) )
        pdf_obj.savefig( fig )
        
    else:
        true_positive_rates_dict = method_visualizers[0].compute_true_positive_rates()
        tpr = true_positive_rates_dict['dca']
        pdb_tpr = true_positive_rates_dict['pdb']
        max_rank = len(tpr)
        ranks = [i + 1 for i in range(max_rank)]
        
                    
        ax1.plot(ranks, tpr, color=colors_hex[method2color[methods[0]]])
        ax1.plot(ranks, pdb_tpr, color='k')
        ax1.set_xscale('log')
        ax_title = '''
        True Positive Rate Per Rank
        PDB cut-off distance : {} Angstrom
        Residue chain distance : {}
        '''
        #ax.set_title(ax_title.format(self.__contact_dist, self.__linear_dist,))
        ax1.set_xlabel('Rank', fontsize=14)
        ax1.set_ylabel('True Positives/Rank', fontsize=14)
        plt.grid()
        plt.tight_layout()
        #plt.savefig('%s_%s_%s_pydca_tp_rate.pdf' % (pdb_id, pfam_id, methods[0]) )
        pdf_obj.savefig( fig )
    
def pydca_contact_plot(method_visualizer, method , pdf_obj, ld=4, contact_dist=10.):
    contact_categories_dict = method_visualizer.contact_categories()
    true_positives = contact_categories_dict['tp']
    false_positives = contact_categories_dict['fp']
    missing_pairs = contact_categories_dict['missing']
    pdb_contacts =  contact_categories_dict['pdb']
    
    filtered_pdb_contacts_list = [ 
       site_pair for site_pair, metadata in pdb_contacts.items() if abs(site_pair[1] - site_pair[0]) > ld  
    ]
    num_filtered_pdb_contacts = len(filtered_pdb_contacts_list)
    
    fig = plt.figure(figsize=(5,5))
    ax2 = plt.subplot2grid((1,1), (0,0))
    if missing_pairs:
        x_missing, y_missing = method_visualizer.split_and_shift_contact_pairs(missing_pairs)
        ax.scatter(x_missing, y_missing, s=6, color='blue')
    x_true_positives, y_true_positives = method_visualizer.split_and_shift_contact_pairs(
        true_positives,
    )
    x_false_positives, y_false_positives = method_visualizer.split_and_shift_contact_pairs(
        false_positives
    )
    x_pdb, y_pdb = method_visualizer.split_and_shift_contact_pairs(pdb_contacts)
    ax_title = '''
    Maximum PDB contact distance : {} Angstrom
    Minimum residue chain distance: {} residues
    Fraction of true positives : {:.3g}
    '''.format(contact_dist, ld,
        len(true_positives)/(len(true_positives) + len(false_positives)),
    )
    
    ax2.scatter(y_true_positives, x_true_positives, s=6, color='green')
    ax2.scatter(y_false_positives, x_false_positives, s=6, color='red')
    ax2.scatter(x_pdb, y_pdb, s=6, color='grey')
    ax2.set_xlabel('Residue Position', fontsize=14)
    ax2.set_ylabel('Residue Position', fontsize=14)
    #ax2.set_title(ax_title)
    plt.tight_layout()
    # plt.savefig('%s_%s_%s_pydca_contact_map.pdf' % (pdb_id, pfam_id, method) )
    pdf_obj.savefig( fig )

def plot_contact_map_ecc(ax2, method, pairs_flat, ct_flat, dist_flat, ER_di_flat, ct_ER, ld_val=5):
    

        
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
    Method: {}
    Maximum PDB contact distance : {} Angstrom
    Minimum residue chain distance: {} residues
    Fraction of true positives : {:.3g}
    '''.format(method, ct_ER, ld_val,
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




def plot_true_positive_rates_ecc(method_visualizers, tps, methods, contact_dist, linear_dist, ax):
    """Plotes the true positive rate per rank of DCA ranked site pairs.
    The x-axis is in log scale.

    Parameters
    ----------
        self : DCAVisualizer

    Returns
    -------
        true_positive_rates_dict : dict
            A dictionary whose keys are true positives types (pdb, or dca)
            and whose values are the corresponding true positive rates per
            rank.

    """
    mv = method_visualizers[0]
    true_positive_rates_dict = mv.compute_true_positive_rates()
    pdb_tpr = true_positive_rates_dict['pdb']
    max_rank = len(pdb_tpr)
    pdb_ranks = [i + 1 for i in range(max_rank)]
    ax.plot(pdb_ranks, pdb_tpr,color='k')
    
    for i, tp in enumerate(tps):
        max_rank = len(tp)
        ranks = [j + 1 for j in range(max_rank)]
        tpr = [t/ranks[j] for j,t in enumerate(tp)]
        ax.plot(ranks, tpr, color=colors_hex[method2color[methods[i]]],label=methods[i])

        ax.set_xscale('log')
        ax_title = '''
        True Positive Rate Per Rank
        PDB cut-off distance : {} Angstrom
        Residue chain distance : {}
        '''
    ax.set_title(ax_title.format(
            contact_dist, linear_dist,
        )
    )
    ax.set_xlabel('Rank', fontsize=14)
    ax.set_ylabel('True Positives/rank', fontsize=14)
    plt.legend(fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.show()
    return ax



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

# PDB ID: 1zdr, Pfam ID: PF00186
pdb_id = sys.argv[1]
pfam_id = sys.argv[2]
#pdb_id = '1zdr'
#pfam_id = 'PF00186'

# create output pdf object
pdf_output = matplotlib.backends.backend_pdf.PdfPages('single_protein_plots/%s_%s_plots.pdf' % (pdb_id, pfam_id) )



ref_outfile = Path(processed_data_dir, '%s_ref.fa' % pfam_id)
prody_df = pd.read_csv('%s/%s_pdb_df.csv' % (pdb_dir, pdb_id))
pdb2msa_row  = prody_df.iloc[0]
print('\n\nGetting msa with following pdb2msa entry:\n', pdb2msa_row)
#try:
print(pdb2msa_row)
pfam_id = pdb2msa_row['Pfam']
pdb_id = pdb2msa_row['PDB ID']
pdb_chain = pdb2msa_row['Chain']




# --- Load scores --- #
plm_out_file = "%s/%s_%s_PLM_di.npy" % (out_dir, pdb_id, pfam_id)
mf_out_file = "%s/%s_%s_PMF_di.npy" % (out_dir, pdb_id, pfam_id)
er_out_file = "%s/%s_%s_ER_di.npy" % (out_dir, pdb_id, pfam_id)
plmdca_scores = np.load(plm_out_file)
mfdca_scores = np.load(mf_out_file)
# ER scores need to be translated
ER_di = np.load(er_out_file)
s_index = np.load("%s/%s_%s_preproc_sindex.npy" % (processed_data_dir, pfam_id, pdb_id))
from ecc_tools import scores_matrix2dict
ER_scores = scores_matrix2dict(ER_di, s_index)
# ------------------- #
# load DI data
ER_di = np.load("%s/%s_%s_ER_di.npy" % (out_dir, pdb_id, pfam_id))

# --- Compare Methods --- #
PMF_di_data = np.load("%s/%s_%s_PMF_di.npy" % (out_dir, pdb_id, pfam_id),allow_pickle=True)
PLM_di_data = np.load("%s/%s_%s_PLM_di.npy" % (out_dir, pdb_id, pfam_id),allow_pickle=True)


# transform pydca DI dictionary to DI matrices.
PLM_di = np.zeros(ER_di.shape)
PLM_di_dict = {}
for score_set in PLM_di_data:
    PLM_di_dict[(score_set[0][0], score_set[0][1])] = score_set[1]
for i, index_i in enumerate(s_index):
    for j, index_j in enumerate(s_index):
        if i==j:
            PLM_di[i,j] = 1.
            continue
        try:
            PLM_di[i,j] = PLM_di_dict[(index_i, index_j)]
            PLM_di[j,i] = PLM_di_dict[(index_i, index_j)] # symetric
        except(KeyError):
            continue

PMF_di = np.zeros(ER_di.shape)
PMF_di_dict = {}
for score_set in PMF_di_data:
    PMF_di_dict[(score_set[0][0], score_set[0][1])] = score_set[1]
for i, index_i in enumerate(s_index):
    for j, index_j in enumerate(s_index):
        if i==j:
            PMF_di[i,j] = 1.
            continue
        try:
            PMF_di[i,j] = PMF_di_dict[(index_i, index_j)]
            PMF_di[j,i] = PMF_di_dict[(index_i, index_j)] # symetric
        except(KeyError):
            continue



er_scores_ordered = {}
for [pair, score] in ER_scores:
    er_scores_ordered[pair] = score
ER_scores = er_scores_ordered
ER_scores = sorted(ER_scores.items(), key =lambda k : k[1], reverse=True)
for site_pair, score in ER_scores[:25]:
    print(site_pair, score)

plm_scores_ordered = {}
for [pair, score] in PLM_di_data:
    plm_scores_ordered[pair] = score
PLM_scores = plm_scores_ordered
PLM_scores = sorted(PLM_scores.items(), key =lambda k : k[1], reverse=True)
for site_pair, score in PLM_scores[:5]:
    print(site_pair, score)

mf_scores_ordered = {}
for [pair, score] in PMF_di_data:
    mf_scores_ordered[pair] = score
PMF_scores = mf_scores_ordered
PMF_scores = sorted(PMF_scores.items(), key =lambda k : k[1], reverse=True)
for site_pair, score in PMF_scores[:5]:
    print(site_pair, score)
   

print('Ref Seq will not match pdb seq because of data_processing but thats ok.')
ld_thresh = 5.
contact_dist = 10.
plmdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
    refseq_file = str(ref_outfile),
    sorted_dca_scores = PLM_di_data,
    linear_dist = ld,
    contact_dist = contact_dist
)

mfdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
    refseq_file = str(ref_outfile),
    sorted_dca_scores = PMF_di_data,
    linear_dist = ld,
    contact_dist = contact_dist
)
    
er_visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
        refseq_file = str(ref_outfile),
        sorted_dca_scores = ER_scores,
        linear_dist = ld,
        contact_dist = contact_dist,
)


ct_file = "%s%s_%s_ct.npy" % (pdb_dir, pdb_id, pfam_id)
ct = np.load(ct_file)

# Set contact distance
ct1 = ct.copy()
ct_pos = ct < contact_dist
ct1[ct_pos] = 1
ct1[~ct_pos] = 0


ld_thresh = 0.
mask = np.triu(np.ones(ER_di.shape[0], dtype=bool), k=1)
# argsort sorts from low to high. [::-1] reverses 
er_order = ER_di[mask].argsort()[::-1]
plm_order = PLM_di[mask].argsort()[::-1]
pmf_order = PMF_di[mask].argsort()[::-1]

linear_distance = np.zeros((len(s_index),len(s_index)))                                                                                                   
for i, ii in enumerate(s_index):                                                                                                                          
    for j, jj in enumerate(s_index):                                                                                                                      
        linear_distance[i,j] = abs(ii - jj)   


ld = linear_distance >= ld_thresh                                                                                                                         

ER_ld_flat = ld[mask][er_order]          
ER_di_flat = ER_di[mask][er_order]
ER_ct_flat = ct1[mask][er_order]
ER_dist_flat = ct[mask][er_order]

PLM_ld_flat = ld[mask][plm_order]          
PLM_di_flat = PLM_di[mask][plm_order]
PLM_ct_flat = ct1[mask][plm_order]
PLM_dist_flat = ct[mask][plm_order]

PMF_ld_flat = ld[mask][pmf_order]          
PMF_di_flat = PMF_di[mask][pmf_order]
PMF_ct_flat = ct1[mask][pmf_order]
PMF_dist_flat = ct[mask][pmf_order]

pairs_matrix = np.empty((len(s_index),len(s_index)), dtype=object)                                                               
for i, ii in enumerate(s_index):
    for j, jj in enumerate(s_index):
        pairs_matrix[i,j] =(i,j)
print(pairs_matrix)
print(pairs_matrix.shape)

ER_pairs_flat = pairs_matrix[mask][er_order]
PLM_pairs_flat = pairs_matrix[mask][plm_order]
PMF_pairs_flat = pairs_matrix[mask][pmf_order]


#pydca_contact_plot(plmdca_visualizer, 'PLM', pdf_output, ld=5, contact_dist=10.)
#pydca_contact_plot(er_visualizer, 'ER', pdf_output, ld=5, contact_dist=10.)
#pydca_contact_plot(mfdca_visualizer, 'MF', pdf_output, ld=5, contact_dist=10.)

fig = plt.figure(figsize=(5,5.75))
ax = plt.subplot2grid((1,1),(0,0))
ax = plot_contact_map_ecc(ax, 'ER', ER_pairs_flat, ER_ct_flat, ER_dist_flat, ER_di_flat, contact_dist, ld_val=ld_thresh)

ax.legend()
plt.tight_layout()
#plt.savefig('%s_%s_di_contact_%sv%s.pdf' % (pdb_id, pfam_id, labels[0],labels[1]) )
pdf_output.savefig( fig )


fig = plt.figure(figsize=(5,5.75))
ax = plt.subplot2grid((1,1),(0,0))
ax = plot_contact_map_ecc(ax, 'MF', PMF_pairs_flat, PMF_ct_flat, PMF_dist_flat, PMF_di_flat, contact_dist, ld_val=ld_thresh)
ax.legend()
plt.tight_layout()
#plt.savefig('%s_%s_di_contact_%sv%s.pdf' % (pdb_id, pfam_id, labels[0],labels[1]) )
pdf_output.savefig( fig )

fig = plt.figure(figsize=(5,5.75))
ax = plt.subplot2grid((1,1),(0,0))
ax = plot_contact_map_ecc(ax, 'PLM', PLM_pairs_flat, PLM_ct_flat, PLM_dist_flat, PLM_di_flat, contact_dist, ld_val=ld_thresh)
ax.legend()
plt.tight_layout()
#plt.savefig('%s_%s_di_contact_%sv%s.pdf' % (pdb_id, pfam_id, labels[0],labels[1]) )
pdf_output.savefig( fig )





colors = ['b', 'r', 'g']
#pydca_tp_plot( [er_visualizer, mfdca_visualizer, plmdca_visualizer], [ 'ER', 'MF','PLM'],pdf_output, ld=4, contact_dist=5. )
#pydca_tp_plot( [plmdca_visualizer ],['PLM'],pdf_output, ld=5, contact_dist=10. )
#pydca_tp_plot( [ mfdca_visualizer], ['MF'],pdf_output, ld=5, contact_dist=10. )
#pydca_tp_plot( [er_visualizer],  [ 'ER'],pdf_output, ld=5, contact_dist=10. )
ct_flats = [ct_flat, PLM_ct_flat, MF_ct_flat]
tps = []
for ct_f in ct_flats:
    tp = np.cumsum(ct_f, dtype=float)
    print(len(tp))
#     if tp[-1] !=0:
#         tp /= tp[-1]
    tps.append(tp)
methods = ['ER', 'MF', 'PLM']

fig = plt.figure(figsize=(5,5.75))
ax = plt.subplot2grid((1,1),(0,0))
ax = plot_true_positive_rates_ecc([plmdca_visualizer ], tps, methods, contact_dist, ld_thresh, ax)
ax.legend()
plt.tight_layout()
#plt.savefig('%s_%s_di_contact_%sv%s.pdf' % (pdb_id, pfam_id, labels[0],labels[1]) )
pdf_output.savefig( fig )


# We dont need individual TP rates...
#plot_true_positive_rates_ecc([plmdca_visualizer ], [tps[0]], ['ER'], contact_dist, ld_thresh, ax)
#plot_true_positive_rates_ecc([plmdca_visualizer ], [tps[0]], ['PLM'], contact_dist, ld_thresh, ax)
#plot_true_positive_rates_ecc([plmdca_visualizer ], [tps[0]], ['MF'], contact_dist, ld_thresh, ax)


# --- Plot DI vs Distance ER --- %
# Define data directories
DCA_ER_dir = '/data/cresswellclayec/DCA_ER' # Set DCA_ER directory
biowulf_dir = '%s/biowulf_full' % DCA_ER_dir


out_dir = '%s/protein_data/di/' % biowulf_dir
out_metric_dir = '%s/protein_data/metrics/' % biowulf_dir

processed_data_dir = "%s/protein_data/data_processing_output/" % biowulf_dir
pdb_dir = '%s/protein_data/pdb_data/' % biowulf_dir

ct_file = "%s%s_%s_ct.npy" % (pdb_dir, pdb_id, pfam_id)
ct = np.load(ct_file)



file_end = ".npy"
fp_file = "%s%s_%s_ER_fp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
tp_file = "%s%s_%s_ER_tp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
er_fp = np.load(fp_file)
er_tp = np.load(tp_file)
file_end = ".npy"
fp_file = "%s%s_%s_PMF_fp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
tp_file = "%s%s_%s_PMF_tp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
pmf_fp = np.load(fp_file)
pmf_tp = np.load(tp_file)
file_end = ".npy"
fp_file = "%s%s_%s_PLM_fp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
tp_file = "%s%s_%s_PLM_tp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
plm_fp = np.load(fp_file)
plm_tp = np.load(tp_file)




#fp_uni_file = "%s%s_%s_ER_fp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
#tp_uni_file = "%s%s_%s_ER_tp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
#er_fp_uni = np.load(fp_uni_file)
#er_tp_uni = np.load(tp_uni_file)


ct1 = ct.copy()
ct_pos = ct < 6
ct1[ct_pos] = 1
ct1[~ct_pos] = 0
mask = np.triu(np.ones(ER_di.shape[0], dtype=bool), k=1)
# argsort sorts from low to high. [::-1] reverses 
order = ER_di[mask].argsort()[::-1]

ld_thresh = 0.
linear_distance = np.zeros((len(s_index),len(s_index)))                                                                                                   
for i, ii in enumerate(s_index):                                                                                                                          
    for j, jj in enumerate(s_index):                                                                                                                      
        linear_distance[i,j] = abs(ii - jj)   


ld = linear_distance >= ld_thresh                                                                                                                         
ld_flat = ld[mask][order]          

ER_di_flat = ER_di[mask][order]
ct_flat = ct1[mask][order]
dist_flat = ct[mask][order]


labels = ['ER']
flat_dis =  [ER_di_flat]

fig = plt.figure(figsize=(5,5))
ax = plt.subplot2grid((1,1),(0,0))
ax = plot_di_vs_ct(ax, ct_flat, dist_flat, flat_dis, ld_flat, labels)
plt.tight_layout()
# ax.legend()
#plt.savefig('%s_%s_%s_di_dist.pdf' % (pdb_id, pfam_id, 'ER') )
pdf_output.savefig( fig )

fp_file = "%s%s_%s_PLM_fp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
tp_file = "%s%s_%s_PLM_tp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
plm_fp = np.load(fp_file)
plm_tp = np.load(tp_file)

#fp_uni_file = "%s%s_%s_PLM_fp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
#tp_uni_file = "%s%s_%s_PLM_tp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
#plm_fp_uni = np.load(fp_uni_file)
#plm_tp_uni = np.load(tp_uni_file)

ct_pos_file = "%s%s_%s_PLM_ct_flat%s" % (processed_data_dir, pdb_id, pfam_id, file_end)
plm_ct_flat = np.load(ct_pos_file)

PLM_di_flat = PLM_di[mask][order] # get array of plm di in the order of ER di to plot together

fp_file = "%s%s_%s_PMF_fp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
tp_file = "%s%s_%s_PMF_tp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
pmf_fp = np.load(fp_file)
pmf_tp = np.load(tp_file)

#fp_uni_file = "%s%s_%s_PMF_fp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
#tp_uni_file = "%s%s_%s_PMF_tp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
#pmf_fp_uni = np.load(fp_uni_file)
#pmf_tp_uni = np.load(tp_uni_file)

ct_pos_file = "%s%s_%s_PMF_ct_flat%s" % (processed_data_dir, pdb_id, pfam_id, file_end)
pmf_ct_flat = np.load(ct_pos_file)

PMF_di_flat = PMF_di[mask][order] # get array of pmf di in the order of ER di to plot together




flat_dis =  [ER_di_flat, PLM_di_flat]
labels = ['ER', 'PLM']
fig = plt.figure(figsize=(5,5))
ax = plt.subplot2grid((1,1),(0,0))
ax = plot_di_compare_methods(ax, ct_flat, flat_dis[0], flat_dis[1], ld_flat, labels)
ax.legend()

plt.tight_layout()
#plt.savefig('%s_%s_di_contact_%sv%s.pdf' % (pdb_id, pfam_id, labels[0],labels[1]) )
pdf_output.savefig( fig )


flat_dis =  [ER_di_flat, PMF_di_flat]
labels = ['ER', 'MF']
fig = plt.figure(figsize=(5,5))
ax2 = plt.subplot2grid((1,1),(0,0))
ax2 = plot_di_compare_methods(ax2, ct_flat, flat_dis[0], flat_dis[1], ld_flat, labels)
ax2.legend()
plt.tight_layout()
#plt.savefig('%s_%s_di_contact_%sv%s.pdf' % (pdb_id, pfam_id, labels[0],labels[1]) )
pdf_output.savefig( fig )

fig = plt.figure(figsize=(5,5))
ax = plt.subplot2grid((1,1),(0,0))
ax.plot(er_fp, er_tp, label='ER', color=colors_hex[colors_key[0]])
ax.plot(pmf_fp, pmf_tp, label='MF', color=colors_hex[colors_key[1]])
ax.plot(plm_fp, plm_tp, label='PLM', color=colors_hex[colors_key[2]])
ax.set_ylabel('True Positive Rate', fontsize=14)
ax.set_xlabel('False Positive Rate', fontsize=14)
ax.legend()
plt.tight_layout()
#plt.savefig('%s_%s_roc_comparison.pdf' % ( pdb_id, pfam_id) )
pdf_output.savefig( fig )

pdf_output.close()


