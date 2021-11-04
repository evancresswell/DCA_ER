import sys
import numpy as np
from scipy import linalg
from scipy.spatial import distance
#import matplotlib.pyplot as plt

from inference_dca import direct_info_dca
#=========================================================================================

np.random.seed(1)
#pfam_id = 'PF00025'
pfam_id = sys.argv[1]

s0 = np.loadtxt('../pfam_10_100k/%s_s0.txt'%(pfam_id)).astype(int)

di = direct_info_dca(s0)
np.savetxt('%s/di.dat'%(pfam_id),di,fmt='% f')
