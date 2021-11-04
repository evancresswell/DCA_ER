# 2018.12.27: Receiver operating characteristic (ROC) curve
import sys
import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#=========================================================================================

#pfam_id = 'PF00025'
#ipdb = 1
pfam_id = sys.argv[1]
#ipdb = sys.argv[2]
#ipdb = int(ipdb)

#ext_name = '%s/%02d'%(pfam_id,ipdb)

#pfam_list = np.loadtxt('pfam_list.txt',dtype='str')
#pfam_list = ['PF00011']
#ct_thres_list = [1.5,2.0,3.0,4.0]
ct_thres_list = [2.0,3.0,4.0]
n = len(ct_thres_list)
#for pfam_id in pfam_list:

try:    
    plt.figure(figsize=(3.0*n,3.2))

    for i,ct_thres in enumerate(ct_thres_list):
        fptp = np.loadtxt('%s/roc_%s.dat'%(pfam_id,ct_thres)).astype(float)
        fp = fptp[:,0]
        tp = fptp[:,1]
        auc = tp.sum()/tp.shape[0]
        plt.subplot2grid((1,n),(0,i))
        plt.title('thres = %3.2f, auc = %5.4f'%(ct_thres,auc))
        plt.plot(fp,tp,'r-')
        plt.plot([0,1],[0,1],'k--')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')

    plt.tight_layout(h_pad=1, w_pad=1.5)
    plt.savefig('%s/roc_thres.pdf'%(pfam_id), format='pdf', dpi=100)

except:
    pass  
#plt.show()
