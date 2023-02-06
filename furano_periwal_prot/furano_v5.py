#this version clusters in sequence space along with the overall landscape
# pdf plot version
from Bio import SeqIO
import numpy as np
import matplotlib
matplotlib.use('Agg')
import sys
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from iterate import mycharray,interval_permute,mybin,weighted_sum,plot3dplot,binarize

NUM_PERMS = 100
INTERVAL = 7
TOLERANCE = 1e-12
DIST_CUTOFF = 0.2 #looking for the shortest distance to a neighbor to define the Gamma stat
M=1e-5
NUM_EVS=80
clust_dim=3
prefix = ''
files = ['all_seqs.fa','cls1ab234.4.fasta','cls1.1.fasta','cls1.2.fasta','cls123.3.fasta','cls123.5.fasta','cls1abc234.6.fasta','cls123.7.fasta']
num_files = len(files)-1
seq_counts = []
cum_seq_counts = []
cum_seq_ct = 0

###snippet below generates the all_hg.fst file
out_rec = []
labels_true=[]
for label_int,filename in enumerate(files[1:]):
    seq_iter = SeqIO.parse(open(prefix+filename,'rU'),'fasta')
#    aln = AlignIO.read(open(filename),'fasta')
    for rec in seq_iter:
        out_rec.append(rec)
        labels_true.append(label_int)

for filename in files[1:]:
    seq_iter = SeqIO.parse(open(prefix+filename,'rU'),'fasta')
    seq_ct = 0
    for rec in seq_iter:
        seq_ct +=1
        cum_seq_ct +=1
    seq_counts.append(seq_ct)
    cum_seq_counts.append(cum_seq_ct)
files = files[:1] # only the all sequence file (which doesn't include the extra_files per Anthony's instructions!!)
##mutant_file = prefix+'151_cc_all_seqs-cg-null_032017_vp.1.fa'
mutant_file = prefix+'151_cc_all_peps_fnl_ed.1.fa'
seq_iter = SeqIO.parse(open(mutant_file,'rU'),'fasta')
mut_seqs = []
mut_activities,mut_labels = [],[]
for rec in seq_iter:
    str1 = str(rec.name)
    mut_activities.append(float(str1[str1.find('_')+1:str1.find('--')]))
    mut_labels.append(str1[str1.find('--')+2:])
    mut_seqs.append(str(rec.seq))
#print('mutant number:',len(mut_labels))
for filename in files:
    seq_iter = SeqIO.parse(open(prefix+filename,'rU'),'fasta')
    seqs = []
    for rec in seq_iter:
        seqs.append(str(rec.seq))
    set_residues = set([])
    num_seq = len(seqs)
    num_pos = len(seqs[0])
    filestr = filename[:-4]+'_'+str(INTERVAL)+'_M'+str(M)+'_EVS'+str(NUM_EVS)+'dim_cl'+str(clust_dim)+'eps0.16'
    print(filestr)
    for i in range(num_seq):
        set_residues = set_residues.union(list(seqs[i]))
    resid = list(set_residues)
    num_aa = len(resid)
    resid = {resid[i]:i for i in range(len(set_residues))}
    binvectors = np.array([mybin(seq,resid) for seq in seqs])
    mut_binvectors = np.array([mybin(mutseq,resid) for mutseq in mut_seqs])
    bin_freqs = np.sum(binvectors,axis=0).reshape((num_pos,num_aa))/float(num_seq)

    perm_expectations = np.zeros((NUM_PERMS,num_pos*num_aa))
    for perm in range(NUM_PERMS):
        perm_expectations[perm] = np.mean(np.array([mybin(seq,resid) for seq in interval_permute(seqs,INTERVAL)]),axis=0)
    mean_perm_exp = np.mean(perm_expectations,axis=0)
    std_perm_exp = np.std(perm_expectations,axis=0)
    def gradient(vectors,num_aa=num_aa,num_pos=num_pos,mean=mean_perm_exp,std=std_perm_exp):
##        std_nonzero = std>TOLERANCE
##        mean_std = np.mean(std[std_nonzero])
##        grad = np.zeros(num_pos*num_aa)
##        grad[std_nonzero] = - (np.sum(vectors[:,std_nonzero],axis=0)-mean[std_nonzero])/(std[std_nonzero]/mean_std)**2
        grad = mean/(0.5/float(vectors.shape[0])+np.sum(vectors,axis=0)) #Dirichlet
        grad = grad.reshape((num_pos,num_aa))
        grad[:] -= np.mean(grad,axis=1)[:,np.newaxis]
        return grad.reshape(num_pos*num_aa)

    def low_dim_pinv(quad):
        w,u = np.linalg.eigh(0.5*(quad+quad.T))
        w_order = np.argsort(w)[::-1]
        w = w[w_order][:NUM_EVS]
        w = 1/w
##        plt.plot(np.arange(0,w.shape[0]),w)
##        plt.show()
##        plt.close()
#        print 'Eigenvalues:',w
        return np.dot(u[:,w_order][:,:NUM_EVS],np.dot(np.diag(w),u.T[w_order,:][:NUM_EVS,:])),u.T[w_order,:][:NUM_EVS,:]

            
    def landscape(vtr,cluster=False,filestr=filestr,cols_sizes=False):
        J=np.array([np.random.rand(num_aa) for i in range(num_pos)])
        J[:] -= np.mean(J,axis = 1)[:,np.newaxis]
        J = J.reshape(num_pos*num_aa)
        J_prev = np.zeros(J.shape)
        J_diff=1
        while J_diff > TOLERANCE:
            x = weighted_sum(J,vtr,0)
            J_prev=np.copy(J)
            J = M*gradient(x)
            J_diff = np.max(np.abs(J-J_prev))
        vec_fp = np.sum(x,axis=0)
        quad_0 = np.inner(vtr.T,x.T)-np.outer(vec_fp,vec_fp) #connected two-point function
        qinv,vt=low_dim_pinv(quad_0)
        qinv_ij = np.abs(qinv)
        qinv_ij = np.sum(qinv_ij.reshape((num_pos*num_aa,num_pos,num_aa)),axis=2)
        qinv_ij = np.sum(qinv_ij.T.reshape((num_pos,num_pos,num_aa)),axis=2)
        qinv_ij -= np.diag(np.diag(qinv_ij))
        qinv_ij /= np.max(qinv_ij)
#        heatmap(filestr+'_qinv',qinv)
        low_coords0 = np.inner(vtr-vec_fp[np.newaxis,:],vt)
        mut_low_coords = np.inner(mut_binvectors-vec_fp[np.newaxis,:],vt)
        low_coords_list = [low_coords0[1:cum_seq_counts[0]],low_coords0[0:1]]
        for k in range(1,num_files):
            low_coords_list.append(low_coords0[cum_seq_counts[k-1]+1:cum_seq_counts[k]])
            low_coords_list.append(low_coords0[cum_seq_counts[k-1]:cum_seq_counts[k-1]+1])
        low_coords_list.append(mut_low_coords)
        low_coords = np.vstack(low_coords_list)
#        colors = cm.Spectral(np.linspace(0, 1, num_files+1))
        colors = [(.71,.32,.8),(1.0,.65,0.),(.55,0.,0.),(0,.8,.8),(0.,.55,.55),\
                  (1.0,.1,.58),(0.,0.,.80),(0.4,0.4,0.4)]
        if not cols_sizes:
            cols,sizes,labs,offs  = [],[],[],[]
            for k,col in zip(range(num_files+1),colors):
                if k<num_files:
                    cols += [col]*seq_counts[k]
                    sizes += [30]*(seq_counts[k])
                    labs += [' ']*(seq_counts[k])
                    offs += [(0.,0.,0.)]*(seq_counts[k])
                else:
                    cols += [col]*len(mut_activities)
                    sizes += [140*act+60 for act in mut_activities]
##                    labs += ['1','','2','3','4','5','','','','6','','','','7','','','','','8','','','','','','','9','10','','','11',\
##                             '12','','','13','14','15','16','17','18','','']
                    labs += mut_labels
                    offs += [(0,0,0.4)]*len(mut_activities)
                    lma = len(mut_activities)

                    offs[-lma] = (-0.6,-0.2,-0.0) #1
                    offs[-lma+1] = (0,0,-0.655) #2
                    offs[-lma+3] = (0,0,0.2) #4
                    offs[-lma+4] = (0,0,0.21) #5
                    offs[-lma+5] = (0.2,0,-0.1) #6
                    offs[-lma+6] = (0,0,0.3) #7
                    offs[-lma+7] = (-0.2,-0,0.425) #8
                    offs[-lma+8] = (0.2,0,0) #9
                    offs[-lma+9] = (-0.2,0,0.2) #10
                    offs[-lma+10] = (-0.1,0,-0.47) #11
                    offs[-lma+11] = (0.2,0,-0.27) #12
                    offs[-lma+12] = (-0.,0,0.25) #13
                    offs[-lma+13] = (-0.45,-0.2,0.27) #14
                    offs[-lma+14] = (-0,-0.,0.25) #15
                    offs[-lma+15] = (0.,-0.0,-0.55) #16
                    offs[-lma+16] = (-0.3,-1,-0) #17
                    offs[-lma+17] = (0.5,0.0,0)#18
                    offs[-lma+18] = (-0.05,0.0,0.4)#19
                    
        else:
            cols,sizes = cols_sizes
        if cluster:
            for epsi in range(16,17,1):
                classes,n_clusters_,uniq_labels,labels = dbscan_clusters(np.copy(low_coords0[:,:clust_dim]),\
                                                                         filename=filestr,eps=0.01*epsi,elev=30,azim=75)
                big_class_labels = [x for x in uniq_labels if x!=-1]
                unlabeled = len([x for x in labels if x==-1])
                big_classes = [[x for x in labels if x==y] for y in big_class_labels]
                big_class_sizes = [len(x) for x in big_classes]
                temp = [len(clas) for clas in classes]
                temp.sort()
                cluster_sizes = temp[::-1]
                kept_classes = set([x for x,y in zip(big_class_labels,big_class_sizes) if y in cluster_sizes])
                l_out,l_true_out = [],[]
                for lab,lab_true in zip(labels,labels_true):
                    if lab in kept_classes:
                        l_out.append(lab)
                        l_true_out.append(lab_true)
                pmi =p_m_i(l_out,l_true_out)
                print(clust_dim,len(big_class_labels),epsi,pmi,pmi[0]*(1-unlabeled/float(len(labels))))                    
            for i_clust in range(n_clusters_):
                landscape(vtr[classes[i_clust]],cluster=False,filestr=filestr+'dim_cl'+str(clust_dim)+'_clust'+str(i_clust),\
                          cols_sizes=([cols[j] for j in classes[i_clust]]+[colors[num_files]]*len(mut_activities),\
                                      [sizes[j] for j in classes[i_clust]]+[150*act+30 for act in mut_activities]))
#        colors = pylab.cm.Spectral(np.linspace(0, 1, num_files+1))
        ONLY_MUTS = False
        NO_MUTS = False
        if ONLY_MUTS:
            fstr = PdfPages(filestr+'_3d_only_mutants_1.pdf')
            cols,sizes,labs,offs  = [],[],[],[]
            for k,col in zip(range(num_files+1),colors):
                if k<num_files:
                    cols += [(0,0,0,0)]*seq_counts[k]
                    sizes += [.0]*(seq_counts[k])
                    labs += [' ']*(seq_counts[k])
                    offs += [(0.,0.,0.)]*(seq_counts[k])
                else:
                    cols += [col]*len(mut_activities)
                    sizes += [140*act+60 for act in mut_activities]
                    labs += mut_labels
                    offs += [(0,0,0.4)]*len(mut_activities)
                    lma = len(mut_activities)
                    offs[-lma] = (-0.6,-0.2,-0.0) #1
                    offs[-lma+1] = (0,0,-0.655) #2
                    offs[-lma+3] = (0,0,0.2) #4
                    offs[-lma+4] = (0,0,0.21) #5
                    offs[-lma+5] = (0.2,0,-0.1) #6
                    offs[-lma+6] = (0,0,0.3) #7
                    offs[-lma+7] = (-0.2,-0,0.425) #8
                    offs[-lma+8] = (0.2,0,0) #9
                    offs[-lma+9] = (-0.2,0,0.2) #10
                    offs[-lma+10] = (-0.1,0,-0.47) #11
                    offs[-lma+11] = (0.2,0,-0.27) #12
                    offs[-lma+12] = (-0.,0,0.25) #13
                    offs[-lma+13] = (-0.45,-0.2,0.27) #14
                    offs[-lma+14] = (-0,-0.,0.25) #15
                    offs[-lma+15] = (0.,-0.0,-0.55) #16
                    offs[-lma+16] = (-0.3,-1,-0) #17
                    offs[-lma+17] = (0.5,0.0,0)#18
                    offs[-lma+18] = (-0.05,0.0,0.4)#19
        if NO_MUTS:
            fstr = PdfPages(filestr+'_3d_no_mutants_1.pdf')
            cols,sizes,labs,offs  = [],[],[],[]
            for k,col in zip(range(num_files+1),colors):
                if k<num_files:
                    cols += [col]*seq_counts[k]
                    sizes += [30]*(seq_counts[k])
                    labs += [' ']*(seq_counts[k])
                    offs += [(0.,0.,0.)]*(seq_counts[k])
                else:
                    cols += [col]*len(mut_activities)
                    sizes += [0.0 for act in mut_activities]
                    offs += [(0.,0.,0.)]*len(mut_activities)
                    labs += ['']*len(mut_activities)
        if not NO_MUTS and not ONLY_MUTS:
            fstr = PdfPages(filestr+'_3d_all_1.pdf')
        for elev in [32]:
##        for elev in range(40,41,2):
#            for azim in [45,135,225,315]:
            for azim in [290]:
##                plot3dplot(low_coords,sizes,cols,filestr,elev=elev,azim=azim)
                fig = plt.figure()
                ax = fig.add_subplot(111,projection = '3d')
                for point in range(low_coords.shape[0]): #all
                    ax.scatter(low_coords[point,0],low_coords[point,1],low_coords[point,2],s=sizes[point],c=cols[point])
                for point in range(low_coords.shape[0]): #all
                    ax.text(low_coords[point,0]+offs[point][0],low_coords[point,1]+offs[point][1],low_coords[point,2]+\
                            offs[point][2],'%s'%(labs[point]),size=7,zorder=1,color='k')
                ax.view_init(elev,azim)
                ax.set_xlabel('PC1')#+str(elev)+' '+str(azim))
                ax.set_ylabel('PC2')
                ax.set_zlabel('PC3')

                fstr.savefig()
        fstr.close()
    landscape(binvectors)
