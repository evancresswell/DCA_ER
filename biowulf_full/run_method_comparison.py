import os.path, sys
from scipy import stats
from math import gcd
from collections import namedtuple
import pickle
 
import numpy as np
import pandas as pd
from scipy import linalg
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import distance
from joblib import Parallel, delayed
import timeit
from itertools import combinations

import matplotlib.pyplot as plt
import ecc_tools as tools

# # --- Import our Code ---# #
#import emachine as EM
from direct_info import direct_info

# import data processing and general DCA_ER tools
from pathlib import Path
np.random.seed(1)
print('Done with initial import')


import pandas as pd
import numpy as np
import scipy.stats

# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5*(i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float)
    ty = np.empty([k, n], dtype=np.float)
    tz = np.empty([k, m + n], dtype=np.float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    print('aucs: ', aucs)
    print('delong cov:', delongcov)
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def delong_roc_variance(ground_truth, predictions):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes log(p-value) for hypothesis that two ROC AUCs are different
    Args:
       ground_truth: np.array of 0 and 1
       predictions_one: predictions of the first model,
          np.array of floats of the probability of being class 1
       predictions_two: predictions of the second model,
          np.array of floats of the probability of being class 1
    """
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = np.vstack((predictions_one, predictions_two))[:, order]
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    return calc_pvalue(aucs, delongcov)
 
def ks_compare_roc_asymptotic(tprs, fprs, methods,n):
    """
    Notes
    -----
    This tests whether 2 samples are drawn from the same distribution. Note
    that, like in the case of the one-sample K-S test, the distribution is
    assumed to be continuous.
    This is the two-sided test, one-sided tests are not implemented.
    The test uses the two-sided asymptotic Kolmogorov-Smirnov distribution.
    If the K-S statistic is small or the p-value is high, then we cannot
    reject the hypothesis that the distributions of the two samples
    are the same.

    Here only the asymptotic ks computation is implemented to work with DCA_ER enviornment 
    """
    MAX_AUTO_N = 10000  # 'auto' will attempt to be exact if n1,n2 <= MAX_AUTO_N
    mode = 'auto'
    alternative = 'two-sided'
    ROC_KstestResult = namedtuple('KstestResult', ('statistic', 'pvalue'))
    
    results = {}
    # getting full length tprs as our cumulative distribution fucntions
    average_fpr, average_tpr, cdfs  = get_average_roc(tprs, fprs, n_cpus)
    method_combos =combinations(methods, 2) 
    for (method1, method2) in list(method_combos):
        cdf1 = cdfs[methods.index(method1)]
        cdf2 = cdfs[methods.index(method2)]
        n1 = int(n)
        n2 = int(n)
        cddiffs = np.asarray(cdf1) - np.asarray(cdf2)
        # Ensure sign of minS is not negative.
        minS = np.clip(-np.min(cddiffs), 0, 1)
        maxS = np.max(cddiffs)

        d = max(minS, maxS)
        g = gcd(n1, n2)
        n1g = n1 // g
        n2g = n2 // g
        prob = -np.inf
        original_mode = mode
        # The product n1*n2 is large.  Use Smirnov's asymptoptic formula.
        # Ensure float to avoid overflow in multiplication
        # sorted because the one-sided formula is not symmetric in n1, n2
        m, n = sorted([float(n1), float(n2)], reverse=True)
        en = m * n / (m + n)
        if alternative == 'two-sided':
            prob = stats.distributions.kstwo.sf(d, np.round(en))
        else:
            z = np.sqrt(en) * d
            # Use Hodges' suggested approximation Eqn 5.3
            # Requires m to be the larger of (n1, n2)
            expt = -2 * z**2 - 2 * z * (m + 2*n)/np.sqrt(m*n*(m+n))/3.0
            prob = np.exp(expt)

        prob = np.clip(prob, 0, 1)
        ks_test_result = ROC_KstestResult(d, prob)
        ks_val, p_val = ks_test_result[0], ks_test_result[1]
        results["%svs%s" % (method1, method2)] = [(ks_val, p_val)]
    return results


create_new = False
printing = True
removing_cols = True
remove_diagonals = False

pdb_path = "/pdb/pdb/"
data_path = Path('/data/cresswellclayec/DCA_ER/Pfam-A.full')
data_path = Path('/data/cresswellclayec/Pfam-A.full')

# Define data directories
DCA_ER_dir = '/data/cresswellclayec/DCA_ER' # Set DCA_ER directory
biowulf_dir = '%s/biowulf_full' % DCA_ER_dir

out_dir = '%s/protein_data/di/' % biowulf_dir
out_metric_dir = '%s/protein_data/metrics/' % biowulf_dir

processed_data_dir = "%s/protein_data/data_processing_output/" % biowulf_dir
pdb_dir = '%s/protein_data/pdb_data/' % biowulf_dir


# pdb_path = "/pdb/pdb/zd/pdb1zdr.ent.gz"
pdb_id = sys.argv[1]
pfam_id = sys.argv[2]
n_cpus = int(sys.argv[3])

file_end = ".npy"

 
ER_di = np.load("%s/%s_%s_ER_di.npy" % (out_dir, pdb_id, pfam_id))
MF_di = np.load("%s/%s_%s_MF_di.npy" % (out_dir, pdb_id, pfam_id))
PMF_di_data = np.load("%s/%s_%s_PMF_di.npy" % (out_dir, pdb_id, pfam_id),allow_pickle=True)
PLM_di_data = np.load("%s/%s_%s_PLM_di.npy" % (out_dir, pdb_id, pfam_id),allow_pickle=True)

s0 = np.load("%s/%s_%s_preproc_msa.npy" % (processed_data_dir, pfam_id, pdb_id))
s_index = np.load("%s/%s_%s_preproc_sindex.npy" % (processed_data_dir, pfam_id, pdb_id))
#pdb_s_index = np.load("%s/%s_%s_preproc_pdb_sindex.npy" % (processed_data_dir, pfam_id, pdb_id))
removed_cols = np.load("%s/%s_%s_removed_cols.npy" % (processed_data_dir, pfam_id, pdb_id))
ref_seq = np.load("%s/%s_%s_preproc_refseq.npy" % (processed_data_dir, pfam_id, pdb_id))


# PMF ROC
# translate PMF di tuple to contact matrix
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
# PLM ROC
# translate PMF di tuple to contact matrix
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
 
ER_fp_file = "%s/%s_%s_ER_fp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
ER_tp_file = "%s/%s_%s_ER_tp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
ER_fp = np.load(ER_fp_file)
ER_tp = np.load(ER_tp_file)

PMF_fp_file = "%s/%s_%s_PMF_fp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
PMF_tp_file = "%s/%s_%s_PMF_tp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
PMF_fp = np.load(PMF_fp_file)
PMF_tp = np.load(PMF_tp_file)

PLM_fp_file = "%s/%s_%s_PLM_fp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
PLM_tp_file = "%s/%s_%s_PLM_tp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
PLM_fp = np.load(PLM_fp_file)
PLM_tp = np.load(PLM_tp_file)

MF_fp_file = "%s/%s_%s_MF_fp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
MF_tp_file = "%s/%s_%s_MF_tp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
MF_fp = np.load(MF_fp_file)
MF_tp = np.load(MF_tp_file)  

tprs = [ER_tp, PMF_tp, PLM_tp, MF_tp]
fprs = [ER_fp, PMF_fp, PLM_fp, MF_fp]
methods = ['ER', 'PMF', 'PLM', 'MF']

print('di shapes:\nER: %d\nMF: %d\nPMF: %d\nPLM: %d' % (ER_di.shape[1],MF_di.shape[1],PMF_di.shape[1],PLM_di.shape[1] ))
from math import comb
print('pair combos (sample size): %d' % comb(ER_di.shape[1],2))
print('NOTE: Scikit-learn\'s ROC curve returncs tpr, fpr arrays <= len of observations..\ntp lengths:\nER: %d\nMF: %d\nPMF: %d\nPLM: %d' % (len(ER_tp), len(PMF_tp), len(PLM_tp), len(MF_tp)))


from generate_average_roc import get_average_roc
results = ks_compare_roc_asymptotic(tprs, fprs, methods, comb(ER_di.shape[1],2))
print(results)


# ------------------------------------------------------------------------------------------------------------------- #
# updated version of roc_curve_new 4/7/2022 
# flat-binary-contact array
from sklearn.metrics import roc_curve as roc_scikit
from sklearn.metrics import auc, precision_recall_curve                                                  
from math import comb

print('adding De Long AUC comparison..')
prody_df = pd.read_csv('%s/%s_pdb_df.csv' % (pdb_dir, pdb_id))

from data_processing import pdb2msa, data_processing_pdb2msa

pdb2msa_row = prody_df.iloc[0]
pfam_id = pdb2msa_row['Pfam']


pdb_s_index = s_index
ct_thres = 6
ld_thresh = 0

pdb_struct_file = '%spdb%s.ent' % (pdb_dir, pdb_id)
ct, ct_full = tools.contact_map_pdb2msa_new(pdb2msa_row, pdb_struct_file, removed_cols, pdb_s_index, pdb_out_dir=pdb_dir, printing=True)

ct1 = ct.copy()

ct_pos = ct1 < ct_thres
ct1[ct_pos] = 1
ct1[~ct_pos] = 0

# Set ER as the reference order for the contact-binary array
di_not = ER_di
mask = np.triu(np.ones(di_not.shape[0], dtype=bool), k=1)
# argsort sorts from low to high. [::-1] reverses 
order = di_not[mask].argsort()[::-1]
ct_flat = ct1[mask][order]

linear_distance = np.zeros((len(s_index),len(s_index)))
for i, ii in enumerate(s_index):
    for j, jj in enumerate(s_index):
        linear_distance[i,j] = abs(ii - jj)
ld = linear_distance >= ld_thresh
ld_flat = ld[mask][order]
#print(ld_flat)
old_len = len(ct_flat)
ct_flat = ct_flat[ld_flat]
ct_pos_flat = ct[mask][order][ld_flat]

print('observations: ', len(ct_flat))
fpr, tpr, thresholds = roc_scikit(ct_flat, di_not[mask][order][ld_flat])
roc_auc= auc(fpr, tpr)
print('ct thresh %f gives auc = %f' % (ct_thres, roc_auc))

MF_di_compare = MF_di[mask][order]
PMF_di_compare = PMF_di[mask][order]
PLM_di_compare = PLM_di[mask][order]


method_flat_di = [ER_di[mask][order], PMF_di_compare, PLM_di_compare, MF_di_compare]

method_combos =combinations(methods, 2) 
for (method1, method2) in list(method_combos):
    idx_1 = methods.index(method1)
    idx_2 = methods.index(method2)
    results["%svs%s" % (method1, method2)].append(delong_roc_test(ct_flat, method_flat_di[idx_1], method_flat_di[idx_2]))

print(results)
 
# ------------------------------------------------------------------------------------------------------------------- #

result_file = "%s/%s_%s_method_comparison.pkl" % (out_metric_dir, pdb_id, pfam_id)
with open(result_file, "wb") as f:
    pickle.dump(results, f, protocol=None)
    f.close()


