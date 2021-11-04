import numpy as np

#pfam_list = np.loadtxt('pfam_list.txt',dtype='str')
#pfam_list = ['PF00011']
#print(pfam_list)

s = np.loadtxt('pfam_10_20k.txt',dtype='str')
pfam_list = s[:,0]

n = len(pfam_list)
ct = np.zeros((n,18))
auc = np.zeros((n,18))
for i,pfam in enumerate(pfam_list):
    ct_auc = np.loadtxt('%s/auc.dat'%pfam)
    ct[i,:] = ct_auc[0]
    auc[i,:] = ct_auc[1]

np.savetxt('auc_av.dat',(ct[0,:],auc.mean(axis=0),auc.std(axis=0)),fmt='%f')

