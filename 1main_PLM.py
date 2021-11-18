import sys
import numpy as np
import pickle
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder
from pydca.plmdca import plmdca
import expectation_reflection as ER
from direct_info import direct_info
from joblib import Parallel, delayed
#========================================================================================
np.random.seed(1)
#pfam_id = 'PF00025'
pfam_id = sys.argv[1]

msa_outfile = 'pfam_ecc/MSA_%s.fa'%(pfam_id) 

# PLM instance
plmdca_inst = plmdca.PlmDCA(
    msa_outfile,
    'protein',
    seqid = 0.8,
    lambda_h = 1.0,
    lambda_J = 20.0,
    num_threads = 16,
    max_iterations = 500,
)

# Compute DCA scores 
sorted_DI_plm = plmdca_inst.compute_sorted_DI()

with open('DI/PLM/plm_DI_%s.pickle'%(pfam_id), 'wb') as f:
    pickle.dump(sorted_DI_plm, f)
f.close()

