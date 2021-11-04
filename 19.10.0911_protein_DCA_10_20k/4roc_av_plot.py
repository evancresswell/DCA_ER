# 2018.12.27: Receiver operating characteristic (ROC) curve
import sys
import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#=========================================================================================

ct_thres_list = [1.5,2.0,3.0,4.0]
n = len(ct_thres_list)
   
plt.figure(figsize=(3.0*n,3.2))

for i,ct_thres in enumerate(ct_thres_list):
    fptp = np.loadtxt('roc_av_40_100k_%s.dat'%(ct_thres)).astype(float)
    fp = fptp[:,0]
    tp = fptp[:,1]
    std = fptp[:,2]
    auc = tp.sum()/tp.shape[0]
    plt.subplot2grid((1,n),(0,i))
    plt.title('thres=%2.1f,auc=%5.4f'%(ct_thres,auc))
    #plt.errorbar(fp,tp,std)
    plt.plot(fp,tp,'b-')
    plt.fill_between(fp,tp-std,tp+std)
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

plt.tight_layout(h_pad=1, w_pad=1.5)
plt.savefig('roc_av_40_100k.pdf', format='pdf', dpi=100)
