import sys, os
import glob
import numpy as np
from pathlib import Path
from math import floor
import timeit
import os
import random
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from sklearn.metrics import auc, precision_recall_curve                                                  


def get_tp_val(fp_val, fpr, tpr):
    for i, fp in enumerate(fpr):
        if fp < fp_val:
            continue
        elif fp == fp_val:
            return tpr[i]
        elif tpr[i] == tpr[i-1]:
            return tpr[i]
        
        if i > 0:
            d = np.sqrt(abs(fp - fpr[i-1])**2 + abs(tpr[i]-tpr[i-1])**2)
            avg_p = abs(fp_val-fpr[i-1])/d
            tp_val = tpr[i-1] + avg_p * abs(tpr[i]-tpr[i-1])
        else:
            d = np.sqrt(abs(fp - 0)**2 + abs(tpr[i]-0)**2)
            avg_p = fp_val/d
            tp_val = avg_p * tpr[i]
        return tp_val

def get_full_length_tpr(fprs, tprs, full_fpr, i):
    tpr = [get_tp_val(fp_val, fprs[i], tprs[i]) for fp_val in full_fpr]
    # print('%d tpr full: ' % i,  len(tpr))
    return tpr
    
def get_average_roc(tprs, fprs, n_cpus):
    average_fpr = fprs[0]
    print(len(average_fpr))
    # fill average fpr with all
    for i, fpr in enumerate(fprs[1:]):
        average_fpr = np.unique(np.sort(np.concatenate((average_fpr, fpr))))
    print(len(average_fpr))
    
    average_tpr = np.zeros(len(average_fpr))
    tprs_full = Parallel(n_jobs = n_cpus-2)(delayed(get_full_length_tpr)(fprs, tprs, average_fpr, i) for i in range(len(fprs)))
    for i, fpr in enumerate(fprs):
        average_tpr = np.add(np.array(tprs_full[i]), average_tpr)
        #plt.plot(fprs[i], tprs[i])
    average_tpr = average_tpr / len(fprs)
    #plt.plot(average_fpr, average_tpr, lw=2.5)
    return average_fpr, average_tpr, tprs_full

if __name__ == "__main__":
    # Define data directories
    DCA_ER_dir = '/data/cresswellclayec/DCA_ER' # Set DCA_ER directory
    biowulf_dir = '%s/biowulf_full' % DCA_ER_dir
    processed_data_dir = "%s/protein_data/data_processing_output/" % biowulf_dir
    
    pdb_path = "/data/cresswellclayec/DCA_ER/biowulf_full/protein_data/metrics"
    data_path = "/data/cresswellclayec/DCA_ER/biowulf_full/protein_data/data_processing_output"
    
    n_cpus = int(sys.argv[1])
    method = sys.argv[2]
    
    tprs = []
    fprs = []
    
    file_end = '_uni_ld5.npy'
    file_end = '_uni.npy'
    file_end = '.npy'
    
    # Get list of files from completed TP files
    tp_files = list(Path(pdb_path).rglob("*%s_tp%s" % (method, file_end)))
    tp_files_str = [str(os.path.basename(path)) for path in tp_files]
    pfam_ids = [tp_str[5:12] for tp_str in tp_files_str] 
    pdb_ids = [tp_str[:4] for tp_str in tp_files_str] 
    
    effective_seqs = []
    print(pfam_ids[:10])
    for i, pdb_id in enumerate(pdb_ids):
        pfam_id = pfam_ids[i]
        try:
            fp_file = "%s/%s_%s_%s_fp%s" % (pdb_path, pdb_id, pfam_id, method, file_end)
            tp_file = "%s/%s_%s_%s_tp%s" % (pdb_path, pdb_id, pfam_id, method, file_end)
            fp = np.load(fp_file)
            tp = np.load(tp_file)
        except(FileNotFoundError):
            continue
        fprs.append(fp)
        tprs.append(tp)
        
    print(len(pfam_ids), ' Pfams plotted')
    
    
    
    avg_fpr, avg_tpr, tprs_full = get_average_roc(tprs, fprs, n_cpus)
    
    np.save('%s/%s_avg_fpr.npy' % (pdb_path, method), avg_fpr)
    np.save('%s/%s_avg_tpr.npy' % (pdb_path, method), avg_tpr)
