import os.path, sys
from scipy import stats
from math import gcd
from scipy.stats._stats_py import _attempt_exact_2kssamp
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

# # --- Import our Code ---# #
#import emachine as EM
from direct_info import direct_info

# import data processing and general DCA_ER tools
from pathlib import Path
np.random.seed(1)
print('Done with initial import')

 
def ks_compare_roc(tprs, fprs, methods):
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
        n1 = len(cdf1)
        n2 = len(cdf2)
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
        if mode == 'auto':
            mode = 'exact' if max(n1, n2) <= MAX_AUTO_N else 'asymp'
        elif mode == 'exact':
            # If lcm(n1, n2) is too big, switch from exact to asymp
            if n1g >= np.iinfo(np.int32).max / n2g:
                mode = 'asymp'
                warnings.warn(
                    f"Exact ks_2samp calculation not possible with samples sizes "
                    f"{n1} and {n2}. Switching to 'asymp'.", RuntimeWarning)

        if mode == 'exact':
            success, d, prob = _attempt_exact_2kssamp(n1, n2, g, d, alternative)
            if not success:
                mode = 'asymp'
                if original_mode == 'exact':
                    warnings.warn(f"ks_2samp: Exact calculation unsuccessful. "
                                  f"Switching to method={mode}.", RuntimeWarning)

        if mode == 'asymp':
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
        results["%svs%s" % (method1, method2)] = (ks_val, p_val)
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

from generate_average_roc import get_average_roc

results = ks_compare_roc(tprs, fprs, methods)
print(results)

result_file = "%s/%s_%s_ks.pkl" % (out_metric_dir, pdb_id, pfam_id)
with open(result_file, "wb") as f:
    pickle.dump(results, f, protocol=None)
    f.close()


