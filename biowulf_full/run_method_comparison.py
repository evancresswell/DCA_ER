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
# intepreting ks statistic results: https://www.graphpad.com/guides/prism/latest/statistics/interpreting_results_kolmogorov-smirnov_test.html
# Original ks function notes from scipy.stats:
"""
    Performs the two-sample Kolmogorov-Smirnov test for goodness of fit.
    This test compares the underlying continuous distributions F(x) and G(x)
    of two independent samples.  See Notes for a description
    of the available null and alternative hypotheses.
    Parameters
    ----------
    data1, data2 : array_like, 1-Dimensional
        Two arrays of sample observations assumed to be drawn from a continuous
        distribution, sample sizes can be different.
    alternative : {'two-sided', 'less', 'greater'}, optional
        Defines the null and alternative hypotheses. Default is 'two-sided'.
        Please see explanations in the Notes below.
    method : {'auto', 'exact', 'asymp'}, optional
        Defines the method used for calculating the p-value.
        The following options are available (default is 'auto'):
          * 'auto' : use 'exact' for small size arrays, 'asymp' for large
          * 'exact' : use exact distribution of test statistic
          * 'asymp' : use asymptotic distribution of test statistic
    Returns
    -------
    statistic : float
        KS statistic.
    pvalue : float
        One-tailed or two-tailed p-value.
    Notes
    -----
    There are three options for the null and corresponding alternative
    hypothesis that can be selected using the `alternative` parameter.
    - `two-sided`: The null hypothesis is that the two distributions are
      identical, F(x)=G(x) for all x; the alternative is that they are not
      identical.
    - `less`: The null hypothesis is that F(x) >= G(x) for all x; the
      alternative is that F(x) < G(x) for at least one x.
    - `greater`: The null hypothesis is that F(x) <= G(x) for all x; the
      alternative is that F(x) > G(x) for at least one x.
    Note that the alternative hypotheses describe the *CDFs* of the
    underlying distributions, not the observed values. For example,
    suppose x1 ~ F and x2 ~ G. If F(x) > G(x) for all x, the values in
    x1 tend to be less than those in x2.
    If the KS statistic is small or the p-value is high, then we cannot
    reject the null hypothesis in favor of the alternative.
    If the method is 'auto', the computation is exact if the sample sizes are
    less than 10000.  For larger sizes, the computation uses the
    Kolmogorov-Smirnov distributions to compute an approximate value.
    The 'two-sided' 'exact' computation computes the complementary probability
    and then subtracts from 1.  As such, the minimum probability it can return
    is about 1e-16.  While the algorithm itself is exact, numerical
    errors may accumulate for large sample sizes.   It is most suited to
    situations in which one of the sample sizes is only a few thousand.
    We generally follow Hodges' treatment of Drion/Gnedenko/Korolyuk [1]_.
    References
    ----------
    .. [1] Hodges, J.L. Jr.,  "The Significance Probability of the Smirnov
           Two-Sample Test," Arkiv fiur Matematik, 3, No. 43 (1958), 469-86.
    Examples
    --------
    Suppose we wish to test the null hypothesis that two samples were drawn
    from the same distribution.
    We choose a confidence level of 95%; that is, we will reject the null
    hypothesis in favor of the alternative if the p-value is less than 0.05.
    If the first sample were drawn from a uniform distribution and the second
    were drawn from the standard normal, we would expect the null hypothesis
    to be rejected.
    >>> from scipy import stats
    >>> rng = np.random.default_rng()
    >>> sample1 = stats.uniform.rvs(size=100, random_state=rng)
    >>> sample2 = stats.norm.rvs(size=110, random_state=rng)
    >>> stats.ks_2samp(sample1, sample2)
    KstestResult(statistic=0.5454545454545454, pvalue=7.37417839555191e-15)
    Indeed, the p-value is lower than our threshold of 0.05, so we reject the
    null hypothesis in favor of the default "two-sided" alternative: the data
    were *not* drawn from the same distribution.
    When both samples are drawn from the same distribution, we expect the data
    to be consistent with the null hypothesis most of the time.
    >>> sample1 = stats.norm.rvs(size=105, random_state=rng)
    >>> sample2 = stats.norm.rvs(size=95, random_state=rng)
    >>> stats.ks_2samp(sample1, sample2)
    KstestResult(statistic=0.10927318295739348, pvalue=0.5438289009927495)
    As expected, the p-value of 0.54 is not below our threshold of 0.05, so
    we cannot reject the null hypothesis.
    Suppose, however, that the first sample were drawn from
    a normal distribution shifted toward greater values. In this case,
    the cumulative density function (CDF) of the underlying distribution tends
    to be *less* than the CDF underlying the second sample. Therefore, we would
    expect the null hypothesis to be rejected with ``alternative='less'``:
    >>> sample1 = stats.norm.rvs(size=105, loc=0.5, random_state=rng)
    >>> stats.ks_2samp(sample1, sample2, alternative='less')
    KstestResult(statistic=0.4055137844611529, pvalue=3.5474563068855554e-08)
    and indeed, with p-value smaller than our threshold, we reject the null
    hypothesis in favor of the alternative.
"""

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
    code from scipy: https://github.com/scipy/scipy/blob/main/scipy/stats/_stats_py.py :: ks_2samp function.
                     -- adapted for ER and fast run.
                     -- adapted to auto run two-sided test asymptotic.
                     
    """
    MAX_AUTO_N = 10000  # 'auto' will attempt to be exact if n1,n2 <= MAX_AUTO_N
    mode = 'auto'
    alternative = 'two-sided'
    ROC_KstestResult = namedtuple('KstestResult', ('statistic', 'pvalue'))
    
    results = {}

    # getting full length tprs as our cumulative distribution fucntions
    # we assume at least 2 cpu (4-2). Dont need many for 3 curves
    average_fpr, average_tpr, cdfs  = get_average_roc(tprs, fprs, n_cpus=4) 

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
       
    return results, cdfs

def gen_ER_ROC(ct_mat, ER_di, pdb_id, pfam_id, file_end='.npy', gen_uniform_roc=False):
    # ER ROC
    auc_ER = np.zeros(n)
    for i in range(n):
        try:
            fpr, tpr, thresholds, auc, ct_pos_flat = roc_curve_new(ct_mat, ER_di, ct_thres[i] ,s_index, ld_thresh=ld_threshold)
            auc_ER[i] = auc
        except:
            auc_ER[i] = 0
    
    # Get ER method's best contact prediction
    i0_ER = np.argmax(auc_ER)
    print('ER auc max:',ct_thres[i0_ER],auc_ER[i0_ER])
    fpr0_ER, tpr0_ER, thresholds_ER, auc, tpr_uni, fpr_uni, auc_uni, uni_bins, ct_pos_flat = roc_curve_new(ct_mat, ER_di, ct_thres[i0_ER], s_index, ld_thresh=ld_threshold, get_uniform=True)
    if gen_uniform_roc:
        fp_uni_file = "%s%s_%s_ER_fp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
        tp_uni_file = "%s%s_%s_ER_tp_uni%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
        np.save(fp_uni_file, fpr_uni)
        np.save(tp_uni_file, tpr_uni)   

    fp_file = "%s%s_%s_ER_fp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    tp_file = "%s%s_%s_ER_tp%s" % (out_metric_dir, pdb_id, pfam_id, file_end)
    np.save(fp_file, fpr0_ER)
    np.save(tp_file, tpr0_ER)

    ct_pos_file = "%s%s_%s_ER_ct_flat%s" % (processed_data_dir, pdb_id, pfam_id, file_end)
    np.save(ct_pos_file, ct_pos_flat)


    #print('ER tp raw (len %d): ' % len(tpr0_ER), tpr0_ER[:10])
    #print('ER tp uni (len %d): ' % len(tpr_uni), tpr_uni[:10])   
    #print('raw auc: %f, binned auc %f' % (auc, auc_uni))

    print('ER tp raw (len %d): ' % len(tpr0_ER), tpr0_ER[:10])
    return fpr0_ER, tpr0_ER, ct_thres[i0_ER]


def gen_PYDCA_ROC(ct_mat, ER_di, s_index, PYDCA_di_data, pdb_id, pfam_id, method, file_end='.npy', gen_uniform_roc=False):
    # PMF ROC
    # translate PMF di tuple to contact matrix
    PYDCA_di = np.zeros(ER_di.shape)
    PYDCA_di_dict = {}
    for score_set in PYDCA_di_data:
        PYDCA_di_dict[(score_set[0][0], score_set[0][1])] = score_set[1]
    for i, index_i in enumerate(s_index):
        for j, index_j in enumerate(s_index):
            if i==j:
                PYDCA_di[i,j] = 1.
                continue
            try:
                PYDCA_di[i,j] = PYDCA_di_dict[(index_i, index_j)]
                PYDCA_di[j,i] = PYDCA_di_dict[(index_i, index_j)] # symetric
            except(KeyError):
                continue
    
    
    auc_PYDCA = np.zeros(n)
    for i in range(n):
        try:
            fpr, tpr, thresholds, auc, ct_pos_flat = roc_curve_new(ct_mat, PYDCA_di, ct_thres[i], s_index, ld_thresh=ld_threshold)
            auc_PYDCA[i] = auc
        except:
            auc_PYDCA[i] = 0
    
    # Get ER method's best contact prediction
    i0_PYDCA = np.argmax(auc_PYDCA)
    print('PYDCA auc max:',ct_thres[i0_PYDCA],auc_PYDCA[i0_PYDCA])
    fpr0_PYDCA, tpr0_PYDCA, thresholds_PYDCA, auc, tpr_uni, fpr_uni, auc_uni, uni_bins, ct_pos_flat = roc_curve_new(ct_mat, PYDCA_di, ct_thres[i0_PYDCA], s_index, ld_thresh=ld_threshold, get_uniform=True)
    if gen_uniform_roc:
        fp_uni_file = "%s%s_%s_%s_fp_uni%s" % (out_metric_dir, pdb_id, pfam_id, method, file_end)
        tp_uni_file = "%s%s_%s_%s_tp_uni%s" % (out_metric_dir, pdb_id, pfam_id, method, file_end)
        np.save(fp_uni_file, fpr_uni)
        np.save(tp_uni_file, tpr_uni)   
        print('%s tp uni (len %d): ' % (method, len(tpr_uni)), tpr_uni[:10])   

    fp_file = "%s%s_%s_%s_fp%s" % (out_metric_dir, pdb_id, pfam_id, method, file_end)
    tp_file = "%s%s_%s_%s_tp%s" % (out_metric_dir, pdb_id, pfam_id, method, file_end)
    np.save(fp_file, fpr0_PYDCA)
    np.save(tp_file, tpr0_PYDCA)
    print('%s tp raw (len %d): ' % (method, len(tpr0_PYDCA)), tpr0_PYDCA[:10])
    print('raw auc: %f, binned auc %f' % (auc, auc_uni))
    ct_pos_file = "%s%s_%s_%s_ct_flat%s" % (processed_data_dir, pdb_id, pfam_id, method, file_end)
    np.save(ct_pos_file, ct_pos_flat)
  
    return PYDCA_di, fpr0_PYDCA, tpr0_PYDCA, ct_thres[i0_PYDCA], auc, ct_pos_flat
   


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

file_end = ".npy"



# pdb_path = "/pdb/pdb/zd/pdb1zdr.ent.gz"

prody_df = pd.read_csv('%s/%s_pdb_df.csv' % (pdb_dir, pdb_id))

from data_processing import pdb2msa, data_processing_pdb2msa

pdb2msa_row = prody_df.iloc[0]


 
# ------------------------------------------------------------------------------------------------------------- # 
# -------------------------------- Load DI Data --------------------------------------------------------------- # 
ER_di = np.load("%s/%s_%s_ER_di.npy" % (out_dir, pdb_id, pfam_id))
PMF_di_data = np.load("%s/%s_%s_PMF_di.npy" % (out_dir, pdb_id, pfam_id),allow_pickle=True)
PLM_di_data = np.load("%s/%s_%s_PLM_di.npy" % (out_dir, pdb_id, pfam_id),allow_pickle=True)

s0 = np.load("%s/%s_%s_preproc_msa.npy" % (processed_data_dir, pfam_id, pdb_id))
s_index = np.load("%s/%s_%s_preproc_sindex.npy" % (processed_data_dir, pfam_id, pdb_id))
#pdb_s_index = np.load("%s/%s_%s_preproc_pdb_sindex.npy" % (processed_data_dir, pfam_id, pdb_id))
removed_cols = np.load("%s/%s_%s_removed_cols.npy" % (processed_data_dir, pfam_id, pdb_id))
ref_seq = np.load("%s/%s_%s_preproc_refseq.npy" % (processed_data_dir, pfam_id, pdb_id))
# ------------------------------------------------------------------------------------------------------------- # 
# ------------------------------------------------------------------------------------------------------------- # 



# --------------------------------------------------------------------------------------------- #
# ----------------------------------- Create Contact Map -------------------------------------- #
# uses pdb2msa_row df for alignement indices beginning and and NOT PFAM
pdb_s_index = s_index
ct, ct_full = tools.contact_map_pdb2msa_new(pdb2msa_row, "%s/pdb%s.ent" % (pdb_dir, pdb_id), removed_cols, pdb_s_index, pdb_out_dir=pdb_dir, printing=True) 
ct_file = "%s%s_%s_ct.npy" % (pdb_dir, pdb_id, pfam_id)
np.save(ct_file, ct)

#print("Direct Information from Expectation reflection:\n",di)
print('s_index (%d): ' %len(s_index), s_index)
print('ER DI shape (before removing cols): ' , ER_di.shape)
er_di = ER_di
#if not removing_cols:
#    er_di = np.delete(di, removed_cols_range,0)
#    er_di = np.delete(er_di, removed_cols_range,1)
#else:
#    er_di = di

print('contact dimentions: ', ct.shape)
print('Final ER DI shape (cols removed): ', er_di.shape)
if remove_diagonals: 
    ER_di = no_diag(er_di, 4, s_index)
else:
    ER_di = er_di

from ecc_tools import scores_matrix2dict
print(s_index)
#ER_di_dict = scores_matrix2dict(ER_di, s_index, removed_cols_range, removing_cols=removing_cols)
#print(ER_di_dict[:20])


from ecc_tools import roc_curve, roc_curve_new, precision_curve
# find optimal threshold of distance
ct_thres = np.linspace(4.,10.,18,endpoint=True)
n = ct_thres.shape[0]


ld_threshold = 0
if ld_threshold > 0:
    file_end = '_ld%d.npy' % ld_threshold
else:
    file_end = '.npy'
   

ct_mat = ct
# --------------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------------- #




# ------------------------------------------------------------------------------------------------------------------- #
# -------------------------------- Generate TP/FP Date -------------------------------------------------------------- #
ER_fp, ER_tp, ER_ct_thres = gen_ER_ROC(ct_mat, ER_di, pdb_id, pfam_id, file_end='.npy', gen_uniform_roc=False)

PMF_di, PMF_fp, PMF_tp, PMF_ct_thres, PMF_auc, PMF_ct_pos_flat = gen_PYDCA_ROC(ct_mat, ER_di, s_index, PMF_di_data, pdb_id, pfam_id, 'PMF', file_end='.npy', gen_uniform_roc=False)

PLM_di, PLM_fp, PLM_tp, PLM_ct_thres, PLM_auc, PLM_ct_pos_flat = gen_PYDCA_ROC(ct_mat, ER_di, s_index, PLM_di_data, pdb_id, pfam_id, 'PLM', file_end='.npy', gen_uniform_roc=False)

tprs = [ER_tp, PMF_tp, PLM_tp]
fprs = [ER_fp, PMF_fp, PLM_fp]
methods = ['ER', 'PMF', 'PLM']

print('di shapes:\nER: %d\nPMF: %d\nPLM: %d' % (ER_di.shape[1],PMF_di.shape[1],PLM_di.shape[1] ))
from math import comb

print('pair combos (sample size): %d' % comb(ER_di.shape[1],2))
print('NOTE: Scikit-learn\'s ROC curve returncs tpr, fpr arrays <= len of observations..\ntp lengths:\nER: %d\nPMF: %d\nPLM: %d' % (len(ER_tp), len(PMF_tp), len(PLM_tp)))


from generate_average_roc import get_average_roc
results,cdfs = ks_compare_roc_asymptotic(tprs, fprs, methods, comb(ER_di.shape[1],2))
print(results)

print('\n\nCDF lengths:')
for i, cdf in enumerate(cdfs):
    print(len(cdf))
    cdf_file = "%s%s_%s_%s_cdf.npy" % (out_metric_dir, pdb_id, pfam_id, methods[i])
print('\n\n')
# ------------------------------------------------------------------------------------------------------------------- #
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

pdb_s_index = s_index
ct_thres = 6
ld_thresh = 0

pdb_struct_file = '%spdb%s.ent' % (pdb_dir, pdb_id)

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


ER_di_compare = di_not[mask][order][ld_flat]
PMF_di_compare = PMF_di[mask][order]
PLM_di_compare = PLM_di[mask][order]


print('observations: ', len(ct_flat))
fpr, tpr, thresholds = roc_scikit(ct_flat, di_not[mask][order][ld_flat])
roc_auc= auc(fpr, tpr)
print('ct thresh %f gives auc = %f' % (ct_thres, roc_auc))


method_flat_di = [ER_di_compare, PMF_di_compare, PLM_di_compare]

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


# number of positions
n_var = s0.shape[1]
n_seq = s0.shape[0]
# compute effective number of sequences
dst = distance.squareform(distance.pdist(s0, 'hamming'))
theta = .2 													# minimum necessary distance (theta = 1. - seq_identity_thresh)
seq_ints = (dst < theta).sum(axis=1).astype(float)
ma_inv = 1/((dst < theta).sum(axis=1).astype(float))  
meff = ma_inv.sum()

print("Number of residue positions:",n_var)
print("Number of sequences:",n_seq)
print('N_effective ', meff)


# save relevant data for categorizing contact prediction metrics
pfam_dimensions = [n_var, n_seq, meff, ER_ct_thres,  PMF_ct_thres,  PLM_ct_thres]
pfam_dimensions_file = "%s%s_%s_pfam_dimensions%s" % (processed_data_dir, pdb_id, pfam_id, file_end)
np.save(pfam_dimensions_file, pfam_dimensions)


