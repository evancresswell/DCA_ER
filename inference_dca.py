import numpy as np
from scipy import linalg
from scipy.spatial import distance
import os, sys
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




def frequency(s0,q,theta,pseudo_weight, seq_weight_outfile=None,first10=False):
    # q --> num site states (21)
    # theta --> minimum percent differnce columns (1-seqid)(.2)
    # pseudo_weight --> lambda equivalent (.5)

    n, l = s0.shape # n --> number of sequences, l --> number of aa in each sequence
    print(s0.shape)

    # hamming distance  -- sequences weight calulation
    dst = distance.squareform(distance.pdist(s0, 'hamming'))
    seq_ints = (dst < theta).sum(axis=1).astype(float)
    ma_inv = 1/((dst < theta).sum(axis=1).astype(float))
    print('ma_inv (sequences weight shape: ', ma_inv.shape)
    meff_tai = ma_inv.sum()

    # pydca             -- sequences weight calculation
    if seq_weight_outfile is not None:
        seqs_weight = compute_sequences_weight(alignment_data = s0, seqid = float(1.-theta), outfile=seq_weight_outfile)
        meff = np.sum(seqs_weight)
        fi_pydca = np.zeros((l, q))

        for i in range(l):
            for a in range(q):#we need gap states single site freqs too
                column_i = s0[:,i]
                freq_ia = np.sum((column_i==a)*seqs_weight)
                fi_pydca[i, a-1] = freq_ia/meff

                if first10:
                    print('site %d-%d freq and count:' % (i,a), freq_ia, np.sum((column_i==a)))

        print(fi_pydca.shape)

        num_site_pairs = (l -1)*l/2
        num_site_pairs = np.int64(num_site_pairs)

        # pydca DOES NOT CONSIDER GAPS (-) so the dimensions are q-1
        fij_pydca = np.zeros(
            shape=(num_site_pairs, q - 1, q - 1), # does not consider - in frequency calc
            # shape=(q, q , q ), # considers gap in frequency calc
            dtype = np.float64
        )
        for i in range(l - 1):
            column_i = s0[:, i]
            for j in range(i+1, l):
                pair_site = int((l * (l - 1)/2) - (l - i) * ((l - i) - 1)/2  + j  - i - 1)
                column_j = s0[:, j]

                # pydca DOES NOT CONSIDER GAPS (-) so the range is q-1
                for a in range(q-1):
                    count_ai = column_i==a
                    for b in range(q-1):
                        count_bj = column_j==b
                        count_ai_bj = count_ai * count_bj
                        freq_ia_jb = np.sum(count_ai_bj*seqs_weight)

                        if first10:
                            print('freq for %d-%d, %d-%d:' %(i,a,j,b), freq_ia_jb)

                        fij_pydca[pair_site, a, b] += freq_ia_jb/meff
        np.save('fij_pydca.npy', fij_pydca)



    # fi_true:
    fi_true = np.zeros((l,q))
    for t in range(n):
        for i in range(l):
            fi_true[i,s0[t,i]] += ma_inv[t]
    print('meff for our MF = ', meff_tai)


    fi_true /= meff_tai

    # fij_true:
    fij_true = np.zeros((l,l,q,q))
    for t in range(n):
        for i in range(l-1):
            for j in range(i+1,l):
                fij_true[i,j,s0[t,i],s0[t,j]] += ma_inv[t]
                fij_true[j,i,s0[t,j],s0[t,i]] = fij_true[i,j,s0[t,i],s0[t,j]]

    fij_true /= meff_tai

    scra = np.eye(q)
    for i in range(l):
        for alpha in range(q):
            for beta in range(q):
                fij_true[i,i,alpha,beta] = fi_true[i,alpha]*scra[alpha,beta]     

    # fi, fij
    fi = (1 - pseudo_weight)*fi_true + pseudo_weight/q
    fij = (1 - pseudo_weight)*fij_true + pseudo_weight/(q**2)

    scra = np.eye(q)
    for i in range(l):
        for alpha in range(q):
            for beta in range(q):
                fij[i,i,alpha,beta] = (1 - pseudo_weight)*fij_true[i,i,alpha,beta] \
                + pseudo_weight/q*scra[alpha,beta] 


    if seq_weight_outfile is not None:
        return fi,fij, fi_pydca, fij_pydca, ma_inv, seq_ints
    else:
        return fi, fij
#=========================================================================================
# convert index from 4d to 2d
def mapkey(i,alpha,q):
    return i*(q-1) + alpha
#=========================================================================================
def correlation(fi,fij,q,l, fi_pydca=None, fij_pydca=None):
    print(fij_pydca.shape)

    # compute correlation matrix:
    c = np.zeros((l*(q-1),l*(q-1)))
    if fi_pydca is not None and fij_pydca is not None:
        corr_mat = np.zeros((l*(q-1),l*(q-1)), dtype=np.float64)
    pair_counter = 0
    for i in range(l):
        for j in range(i, l):
            for alpha in range(q-1):
                for beta in range(q-1):


                    #new improving diagonal..
                    if i==j:
                        fia, fib = fi[i, alpha], fi[i, beta]
                        c[mapkey(i,alpha,q), mapkey(j,beta,q)] = fia*(1.0 - fia) if alpha == beta else -1.0*fia*fib
                        c[mapkey(j,beta,q), mapkey(i,alpha,q)] = fia*(1.0 - fia) if alpha == beta else -1.0*fia*fib
                    else:
                        c[mapkey(i,alpha,q), mapkey(j,beta,q)] = fij[i, j, alpha, beta] - fi[i, alpha] * fi[j, beta]
                        c[mapkey(j,beta,q), mapkey(i,alpha,q)] = fij[i, j, alpha, beta] - fi[i, alpha] * fi[j,beta]


                    #c[mapkey(i,alpha,q),mapkey(j,beta,q)] = fij[i,j,alpha,beta] - fi[i,alpha]*fi[j,beta]
                    #c[mapkey(j,beta,q), mapkey(i,alpha,q)] = fij[i,j,alpha,beta] - fi[i,alpha]*fi[j,beta]
                    if fi_pydca is not None:
                        if i==j:
                            fia, fib = fi_pydca[i, alpha], fi_pydca[i, beta]
                            corr_ij_ab = fia*(1.0 - fia) if alpha == beta else -1.0*fia*fib
                        else:
                            corr_ij_ab = fij_pydca[pair_counter, alpha, beta] - fi_pydca[i, alpha] * fi_pydca[j, beta]
                        corr_mat[mapkey(i,alpha,q),mapkey(j,beta,q)] = corr_ij_ab
                        corr_mat[mapkey(j,beta,q), mapkey(i,alpha,q)] = corr_ij_ab
            if i != j : pair_counter += 1
    print(l*(q-1)) 
    if fi_pydca is None: 
        return c                
    else:
        return c, corr_mat
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
                    w[i,j,alpha,beta] = -c_inv[mapkey(i,alpha,q),mapkey(j,beta,q)]
                    w2[mapkey(i,alpha,q),mapkey(j,beta,q)] = -c_inv[mapkey(i,alpha,q),mapkey(j,beta,q)]
                    
    w2 = w2+w2.T
    return w,w2
#=========================================================================================
# direct information
def direct_info(w,fi,q,l):
    ew_all = np.exp(w)
    di = np.zeros((l,l))
    tiny = 10**(-100.)
    diff_thres = 10**(-4.)

    for i in range(l-1):
        for j in range(i+1,l):        
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
def direct_info_dca(s0,q=21,theta=0.2,pseudo_weight=0.5, seq_wt_outfile=None, first10=False):
    n, l = s0.shape # n --> number of sequences, l --> number of aa in each sequence
    mx = np.full(n,q)
    
    if seq_wt_outfile is not None:
        fi,fij,fi_pydca, fij_pydca, ma_inv,seq_ints = frequency(s0,q,theta,pseudo_weight, seq_weight_outfile=seq_wt_outfile, first10=first10)
    else:
        fi,fij = frequency(s0,q,theta,pseudo_weight)

    print(fi_pydca.shape)
    # regularization of pydca's frequency
    reg_fi_pydca = fi_pydca

    print(reg_fi_pydca.shape)
    theta_by_q = np.float64(pseudo_weight)/np.float64(q)
    for i in range(l):
        for a in range(q):
            reg_fi_pydca[i, a] = theta_by_q + \
                (1.0 - pseudo_weight)*reg_fi_pydca[i, a]
    reg_fij_pydca = fij_pydca
    theta_by_qsqrd = pseudo_weight/float(q * q)
    pair_counter = 0
    for i in range(l - 1):
        for j in range(i + 1, l):
            for a in range(q-1):
                for b in range(q-1):
                    reg_fij_pydca[pair_counter, a, b] = theta_by_qsqrd + \
                        (1.0 - pseudo_weight)*reg_fij_pydca[pair_counter, a, b]
            pair_counter += 1





    c, c_pydca = correlation(fi,fij,q,l, fi_pydca, fij_pydca)

    # c_inv = linalg.inv(c)
    c_inv = np.linalg.inv(c)
    c_inv_pydca = np.linalg.inv(c_pydca)


    w,w2d = interactions(c_inv,q,l)
    w_pydca,w2d_pydca = interactions(c_inv_pydca,q,l)

    #np.save('w.npy',w)  # 4d
    #np.savetxt('w2d.dat',w2d,fmt='%f') # 2d

    di = direct_info(w,fi,q,l)
    di_pydca = direct_info(w_pydca,fi_pydca,q,l)
    
    return di, fi, fij, c, c_inv, w, w2d, reg_fi_pydca, reg_fij_pydca, c_pydca, c_inv_pydca, w_pydca, w2d_pydca, di_pydca, ma_inv,seq_ints

#np.savetxt('%s_di.dat'%(pfam_id),di,fmt='% f')
#plt.imshow(di,cmap='rainbow',origin='lower')

