import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

#===================================================================================================
#pfam_id = 'PF00004'
#pfam_id = sys.argv[1]
#s = np.loadtxt('pfam_40_200k.txt',dtype='str')
s = np.loadtxt('pfam_10_100k.txt',dtype='str')
pfam_list = s[:,0]

for pfam_id in pfam_list:
    try:    
        ct = np.loadtxt('../pfam_10_100k/%s_ct.txt'%pfam_id).astype(float)
        di = np.loadtxt('%s/di.dat'%pfam_id).astype(float)  

        plt.figure(figsize=(5,3.2))

        plt.subplot2grid((1,2),(0,0))
        plt.title('Pfam: %s'%pfam_id)
        plt.imshow(-np.log(1+ct),origin='lower')
        plt.xticks([])
        plt.yticks([])
        #plt.xticks([0,50,100])
        #plt.yticks([0,50,100])

        plt.subplot2grid((1,2),(0,1))
        plt.title('direct info')
        plt.imshow(di,origin='lower')
        plt.xticks([])
        plt.yticks([])
        #plt.colorbar()

        plt.tight_layout(h_pad=1, w_pad=1.5)
        plt.savefig('%s/ct_di_heatmap.pdf'%pfam_id, format='pdf', dpi=100)
        #plt.savefig('ct_di_heatmap.pdf', format='pdf', dpi=100)
        plt.close()

    except:
        pass

