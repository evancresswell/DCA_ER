import numpy as np
from scipy import linalg
from scipy.spatial import distance
import os
#import matplotlib.pyplot as plt
#=========================================================================================
#np.random.seed(1)
#pfam_id = 'PF00018'
#s0 = np.loadtxt('data_processed/%s_s0.txt'%(pfam_id)).astype(int)

# pydca implementation to compute sequences weight (for use in calculation of M_effective)
def compute_sequences_weight(alignment_data, seqid, outfile=None):
    """Computes weight of sequences. The weights are calculated by lumping
    together sequences whose identity is greater that a particular threshold.
    For example, if there are m similar sequences, each of them will be assigned
    a weight of 1/m. Note that the effective number of sequences is the sum of
    these weights.

    Parameters
    ----------
        alignmnet_data : np.array()
            Numpy 2d array of the alignment data, after the alignment is put in
            integer representation
        seqid : float
            Value at which beyond this sequences are considered similar. Typical
            values could be 0.7, 0.8, 0.9 and so on

    Returns
    -------
        seqs_weight : np.array()
            A 1d numpy array containing computed weights. This array has a size
            of the number of sequences in the alignment data.
    """
    if os.path.exists(outfile) and outfile is not None:
        seqs_weight = np.load(outfile)
    else:
        alignment_shape = alignment_data.shape
        num_seqs = alignment_shape[0]
        seqs_len = alignment_shape[1]
        seqs_weight = np.zeros((num_seqs,), dtype=np.float64)
        #count similar sequences
        for i in range(num_seqs):
            seq_i = alignment_data[i]
            for j in range(num_seqs):
                seq_j = alignment_data[j]
                iid = np.sum(seq_i==seq_j)
                if np.float64(iid)/np.float64(seqs_len) > seqid:
                    seqs_weight[i] += 1
        #compute the weight of each sequence in the alignment
        for i in range(num_seqs): seqs_weight[i] = 1.0/float(seqs_weight[i])

        if outfile is not None:
            np.save(outfile, seqs_weight)
        else:
            np.save('seq_weight.npy', seqs_weight)

    return seqs_weight




def frequency(s0,q,theta,pseudo_weight, seq_weight_outfile=None):
    # q --> num site states (21)
    # theta --> minimum percent differnce columns (1-seqid)(.2)
    # pseudo_weight --> lambda equivalent (.5)

    l,n = s0.shape # l --> number of sequences, n --> number of aa in each sequence
    print(s0.shape)

    # hamming distance
    dst = distance.squareform(distance.pdist(s0, 'hamming'))
    ma_inv = 1/(1+(dst < theta).sum(axis=1).astype(float))

    if seq_weight_outfile is not None:
        seqs_weight = compute_sequences_weight(alignment_data = s0, seqid = float(1.-theta), outfile=seq_weight_outfile)
        meff = np.sum(seqs_weight)
    else:
        meff = ma_inv.sum()

    # fi_true:
    fi_true = np.zeros((n,q))
    for t in range(l):
        for i in range(n):
            fi_true[i,s0[t,i]] += ma_inv[t]
    print('meff = ', meff)

    fi_true /= meff

    # fij_true:
    fij_true = np.zeros((n,n,q,q))
    for t in range(l):
        for i in range(n-1):
            for j in range(i+1,n):
                fij_true[i,j,s0[t,i],s0[t,j]] += ma_inv[t]
                fij_true[j,i,s0[t,j],s0[t,i]] = fij_true[i,j,s0[t,i],s0[t,j]]

    fij_true /= meff  

    scra = np.eye(q)
    for i in range(n):
        for alpha in range(q):
            for beta in range(q):
                fij_true[i,i,alpha,beta] = fi_true[i,alpha]*scra[alpha,beta]     

    # fi, fij
    fi = (1 - pseudo_weight)*fi_true + pseudo_weight/q
    fij = (1 - pseudo_weight)*fij_true + pseudo_weight/(q**2)

    scra = np.eye(q)
    for i in range(n):
        for alpha in range(q):
            for beta in range(q):
                fij[i,i,alpha,beta] = (1 - pseudo_weight)*fij_true[i,i,alpha,beta] \
                + pseudo_weight/q*scra[alpha,beta] 

    return fi,fij            
#=========================================================================================
# convert index from 4d to 2d
def mapkey(i,alpha,q):
    return i*(q-1) + alpha
#=========================================================================================
def correlation(fi,fij,q,n):
    # compute correlation matrix:
    c = np.zeros((n*(q-1),n*(q-1)))
    for i in range(n):
        for j in range(n):
            for alpha in range(q-1):
                for beta in range(q-1):
                    c[mapkey(i,alpha,q),mapkey(j,beta,q)] = fij[i,j,alpha,beta] - fi[i,alpha]*fi[j,beta]
                    
    return c                
#=========================================================================================
# set w = - c_inv
def interactions(c_inv,q,n):
    w = np.zeros((n,n,q,q))
    w2 = np.zeros((n*q,n*q))
    for i in range(n):
        for j in range(i+1,n):
            for alpha in range(q-1):
                for beta in range(q-1):
                    w[i,j,alpha,beta] = -c_inv[mapkey(i,alpha,q),mapkey(j,beta,q)]
                    w2[mapkey(i,alpha,q),mapkey(j,beta,q)] = -c_inv[mapkey(i,alpha,q),mapkey(j,beta,q)]
                    
    w2 = w2+w2.T
    return w,w2
#=========================================================================================
# direct information
def direct_info(w,fi,q,n):
    
    ew_all = np.exp(w)
    di = np.zeros((n,n))
    tiny = 10**(-100.)
    diff_thres = 10**(-4.)

    for i in range(n-1):
        for j in range(i+1,n):        
            ew = ew_all[i,j,:,:]

            #------------------------------------------------------
            # find h1 and h2:

            # initial value
            diff = diff_thres + 1.
            eh1 = np.full(q,1./q)
            eh2 = np.full(q,1./q)

            fi0 = fi[i,:]
            fj0 = fi[j,:]

            for iloop in range(100):
                eh_ew1 = eh2.dot(ew.T)
                eh_ew2 = eh1.dot(ew)

                eh1_new = fi0/eh_ew1
                eh1_new /= eh1_new.sum()

                eh2_new = fj0/eh_ew2
                eh2_new /= eh2_new.sum()

                diff = max(np.max(np.abs(eh1_new - eh1)),np.max(np.abs(eh2_new - eh2)))

                eh1,eh2 = eh1_new,eh2_new    
                if diff < diff_thres: break        

            # direct information
            eh1eh2 = eh1[:,np.newaxis]*eh2[np.newaxis,:]
            pdir = ew*(eh1eh2)
            pdir /= pdir.sum() 

            fifj = fi0[:,np.newaxis]*fj0[np.newaxis,:]

            dijab = pdir*np.log((pdir+tiny)/(fifj+tiny))
            di[i,j] = dijab.sum()

    # symmetrize di
    di = di + di.T
    return di
#=========================================================================================
def direct_info_dca(s0,q=21,theta=0.2,pseudo_weight=0.5, seq_wt_outfile=None):
    l,n = s0.shape
    mx = np.full(n,q)
    
    if seq_wt_outfile is not None:
        fi,fij = frequency(s0,q,theta,pseudo_weight, seq_weight_outfile=seq_wt_outfile)
    else:
        fi,fij = frequency(s0,q,theta,pseudo_weight)
    c = correlation(fi,fij,q,n)

    # c_inv = linalg.inv(c)
    c_inv = np.linalg.inv(c)


    w,w2d = interactions(c_inv,q,n)
    #np.save('w.npy',w)  # 4d
    #np.savetxt('w2d.dat',w2d,fmt='%f') # 2d
    di = direct_info(w,fi,q,n)
    
    return di, fi, fij, c, c_inv

#np.savetxt('%s_di.dat'%(pfam_id),di,fmt='% f')
#plt.imshow(di,cmap='rainbow',origin='lower')

