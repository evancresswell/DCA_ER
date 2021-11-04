import numpy as np

#pfam_list = np.loadtxt('pfam_list.txt',dtype='str')
#pfam_list = ['PF00011']
#print(pfam_list)

s = np.loadtxt('pfam_40_100k.txt',dtype='str')
pfam_list = s[:,0]

ct_thres_list = [1.5,2.0,3.0,4.0]
#ct_thres = 2.0

npfam = len(pfam_list)

tp = np.zeros((npfam,101))

for ct_thres in ct_thres_list:
    j= 0 
    for pfam in pfam_list:
        try:          
            fptp = np.loadtxt('%s/roc_%s.dat'%(pfam,ct_thres)).astype(float)        
            fp = fptp[:,0]
            tp[j,:] = fptp[:,1]
            j += 1 
        except:
            print('pfam does not exist:',pfam)
            pass
        
    tp_av = tp[0:j].mean(axis=0)
    tp_std = tp[0:j].std(axis=0)
    np.savetxt('roc_av_40_100k_%s.dat'%(ct_thres),zip(fp,tp_av,tp_std),fmt='% f')

print('number of pfam:',j)
