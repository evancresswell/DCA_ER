import glob
import numpy as np
from pathlib import Path
from math import floor
import timeit
import os
import random
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import auc, precision_recall_curve                                                  

colors_hex = {"red": "#e41a1c", "blue": "#2258A5", "green": "#349C55", "purple": "#984ea3", "orange": "#FF8B00",
                      "yellow": "#ffff33", "grey": "#BBBBBB"}
colors_key = ["blue", "orange", "green"]


# Define data directories
DCA_ER_dir = '/data/cresswellclayec/DCA_ER' # Set DCA_ER directory
biowulf_dir = '%s/biowulf_full' % DCA_ER_dir
processed_data_dir = "%s/protein_data/data_processing_output/" % biowulf_dir

pdb_path = "/data/cresswellclayec/DCA_ER/biowulf_full/protein_data/metrics"
data_path = "/data/cresswellclayec/DCA_ER/biowulf_full/protein_data/data_processing_output"
metric_dir = "/data/cresswellclayec/DCA_ER/biowulf_full/protein_data/metrics"


ER_tprs = []
ER_fprs = []
PMF_tprs = []
PMF_fprs = []
PLM_tprs = []
PLM_fprs = []


ks_compares = []
mc = []
ER_bootstrap_aucs = []

MSA_sizes = []
# # Get list of files from completed TP files
tp_files = list(Path(pdb_path).rglob("*PLM_tp.npy"))
tp_files_str = [str(os.path.basename(path)) for path in tp_files]
pfam_ids = [tp_str[5:12] for tp_str in tp_files_str] 
pdb_ids = [tp_str[:4] for tp_str in tp_files_str] 

file_end = '_uni.npy'
file_end = '_uni_ld5.npy'
file_end = '.npy'

# Get list of files from completed auc-bootstrap files

effective_seqs = []
print(pfam_ids[:10])
for i, pdb_id in enumerate(pdb_ids):
    pfam_id = pfam_ids[i]
    try:
        ER_fp_file = "%s/%s_%s_ER_fp%s" % (pdb_path, pdb_id, pfam_id, file_end)
        ER_tp_file = "%s/%s_%s_ER_tp%s" % (pdb_path, pdb_id, pfam_id, file_end)
        ER_fp = np.load(ER_fp_file)
        ER_tp = np.load(ER_tp_file)
        
        PMF_fp_file = "%s/%s_%s_PMF_fp%s" % (pdb_path, pdb_id, pfam_id, file_end)
        PMF_tp_file = "%s/%s_%s_PMF_tp%s" % (pdb_path, pdb_id, pfam_id, file_end)
        PMF_fp = np.load(PMF_fp_file)
        PMF_tp = np.load(PMF_tp_file)
        
        PLM_fp_file = "%s/%s_%s_PLM_fp%s" % (pdb_path, pdb_id, pfam_id, file_end)
        PLM_tp_file = "%s/%s_%s_PLM_tp%s" % (pdb_path, pdb_id, pfam_id, file_end)
        PLM_fp = np.load(PLM_fp_file)
        PLM_tp = np.load(PLM_tp_file)
        
        MF_fp_file = "%s/%s_%s_MF_fp%s" % (pdb_path, pdb_id, pfam_id, file_end)
        MF_tp_file = "%s/%s_%s_MF_tp%s" % (pdb_path, pdb_id, pfam_id, file_end)
        MF_fp = np.load(MF_fp_file)
        MF_tp = np.load(MF_tp_file)   
        
        ER_bootstrap_file = "%s/%s_%s_bootstrap_aucs.npy" % (pdb_path, pfam_id, pdb_id)
        ER_bootstrap = np.load(ER_bootstrap_file)
        
        MSA_file = "%s/%s_%s_preproc_msa.npy" % (data_path, pfam_id, pdb_id)
        MSA = np.load(MSA_file)
        
        pfam_dimensions_file = "%s%s_%s_pfam_dimensions.npy" % (processed_data_dir, pdb_id, pfam_id)
        pfam_dimensions = np.load(pfam_dimensions_file)

        ks_file = "%s/%s_%s_ks.pkl" % (pdb_path, pdb_id, pfam_id)
        with open(ks_file, "rb") as f:
            ks = pickle.load(f)
        f.close()
        
        compare_file = "%s/%s_%s_method_comparison.pkl" % (pdb_path, pdb_id, pfam_id)
        with open(compare_file, "rb") as f:
            comparison = pickle.load(f)
        f.close()

    except(FileNotFoundError):
        continue
    PMF_fprs.append(PMF_fp)
    PMF_tprs.append(PMF_tp)
    MF_fprs.append(MF_fp)
    MF_tprs.append(MF_tp)
    ER_fprs.append(ER_fp)
    ER_tprs.append(ER_tp)
    PLM_fprs.append(PLM_fp)
    PLM_tprs.append(PLM_tp)
    
    ks_compares.append(ks)
    mc.append(comparison)
    
    ER_bootstrap_aucs.append(ER_bootstrap)
    MSA_sizes.append(MSA.shape)
    if len(pfam_dimensions)==7:
        [n_col, n_seq, m_eff, ct_ER, ct_MF, ct_PMF, ct_PLM] = pfam_dimensions
    elif len(pfam_dimensions)==3:
        [n_col, n_seq, m_eff] = pfam_dimensions
    effective_seqs.append(m_eff)

print(len(pfam_ids), ' Pfams plotted')


#---------------- Plot Histogram of AUC for all methods ----------------#
# bin AUC values for each method
plt.figure(figsize=(5.,5.))
for i, method_auc in enumerate(method_aucs):
    method = method_label[i]
    plt.hist(method_auc, density=True, bins=25, alpha=.3, label = method, color=colors_hex[colors_key[i]])  # density=False would make counts
    plt.hist(method_auc, density=True, bins=25, histtype='step', color=method_colors_hex[colors_key[i]], linewidth=1.4)  # density=False would make counts

plt.legend()
plt.xlabel('AUC', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.savefig('%s/AUC_method_comparison.pdf' % biowulf_dir)
#-----------------------------------------------------------------------#


#---------------- Plot the best method for all proteins ----------------#

#-----------------------------------------------------------------------#
max_auc_indices = []
max_aucs = []
auc_differences = []
for i, er_auc in enumerate(method_aucs[0]):
    pmf_auc = method_aucs[1][i]
    plm_auc = method_aucs[2][i]
    auc_compare = [er_auc, pmf_auc, plm_auc]
    max_auc = max(auc_compare)
    max_aucs.append(max_auc)
    max_auc_index = auc_compare.index(max_auc)
    max_auc_indices.append(max_auc_index)
    auc_differences.append(abs(max_auc - np.mean([auc for auc in auc_compare if auc!=max_auc])))
    
    # confidence intervals
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(ER_bootstrap_aucs[i], p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(ER_bootstrap_aucs[i], p))
    #print('ER auc =%f PLM auc=%f PMF auc=%f\n%.1f confidence interval %.1f%% and %.1f%%' % 
    #      (er_auc, plm_auc, pmf_auc, alpha*100, lower*100, upper*100))
    if max_auc_index == 0 and np.mean([auc for auc in auc_compare if auc!=max_auc]) < lower:
        max_auc_indices.append(.25)


plt.figure(figsize=(26.0,12))
ax = plt.subplot2grid((1,2),(0,0))
ax.hist(max_auc_indices ,range=(0,2) )  # density=False would make counts
ax.set_xticks([0,.25,1,2])
ax.set_xticklabels(['ER', 'ER clear', 'PMF', 'PLM'])
plt.subplot2grid((1,2), (0,1))
plt.hist(auc_differences, bins = 100, range=(0,.3))  # density=False would make counts
plt.title('Difference between best and mean of losers\naverage: %f' % np.mean(auc_differences))
#-----------------------------------------------------------------------#




