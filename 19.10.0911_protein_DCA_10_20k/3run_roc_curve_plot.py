import os
import numpy as np

pfam_list = np.loadtxt('pfam_list.txt',dtype='str')
#n = pfam_list.shape[0]

#os.system('python 1main_DCA.py PF00011 &')
#os.system('python 1main_DCA.py PF00014 &')
for pfam in pfam_list:
    os.system('python 3roc_curve_plot.py %s &'%pfam)

