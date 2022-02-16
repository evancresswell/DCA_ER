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
def compute_sequences_weight(alignment_data, seqid, outfile=None, create_new = False):
    # this function is directly from PYDCA.. with the sole addition of outfile and create new
	# -- this simply allows us to save and load the sequence weights. plays no role in computation
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
    if os.path.exists(outfile) and outfile is not None and not create_new:
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

def compute_single_site_freqs(alignment_data=None,
        num_site_states=None, seqs_weight=None):
    # this function is directly from PYDCA.. with only the following corrections:
	# -- Change 1: range of amino acid for loop is 0-20 not 1-21 (PYDCA) this is simply an indexing difference as we encode A as 0 not as 1 (PYDCA)
	# -- Change 2: indexing shift for updated difference between 
    print('PYDCA\'s compute_singel_site_freqs: 2 changes')
    """Computes single site frequency counts for a particular aligmnet data.

    Parameters
    ----------
        alignment_data : np.array()
            A 2d numpy array of alignment data represented in integer form.

        num_site_states : int
            An integer value fo the number of states a sequence site can have
            including a gap state. Typical value is 5 for RNAs and 21 for
            proteins.

        seqs_weight : np.array()
            A 1d numpy array of sequences weight

    Returns
    -------
        single_site_freqs : np.array()
            A 2d numpy array of of data type float64. The shape of this array is
            (seqs_len, num_site_states) where seqs_len is the length of sequences
            in the alignment data.
    """
    alignment_shape = alignment_data.shape
    #num_seqs = alignment_shape[0]
    seqs_len = alignment_shape[1]
    m_eff = np.sum(seqs_weight)
    print('m_eff for pydca calculation: ', m_eff)
    single_site_freqs = np.zeros(shape = (seqs_len, num_site_states),
        dtype = np.float64)
    for i in range(seqs_len):
        ## for a in range(1, num_site_states + 1):#we need gap states single site freqs too 
        for a in range(num_site_states):#we need gap states single site freqs too ##ECC change 1 -- indexing shift for our code.. A == 0 not A == 1 (PYDCA)
            column_i = alignment_data[:,i]
            freq_ia = np.sum((column_i==a)*seqs_weight) 
            # single_site_freqs[i, a-1] = freq_ia/m_eff 
            single_site_freqs[i, a] = freq_ia/m_eff ##ECC change 1 -- In accordance with change 1 we correct indexing for filling single_site-freqs matrix
    return single_site_freqs

def get_reg_single_site_freqs(single_site_freqs = None, seqs_len = None,
        num_site_states = None, pseudocount = None):
    """Regularizes single site frequencies.

    Parameters
    ----------
        single_site_freqs : np.array()
            A 2d numpy array of single site frequencies of shape
            (seqs_len, num_site_states). Note that gap state frequencies are
            included in this data.
        seqs_len : int
            The length of sequences in the alignment data
        num_site_states : int
            Total number of states that a site in a sequence can accommodate. It
            includes gap states.
        pseudocount : float
            This is the value of the relative pseudo count of type float.
            theta = lambda/(meff + lambda), where meff is the effective number of
            sequences and lambda is the real pseudo count.

    Returns
    -------
        reg_single_site_freqs : np.array()
            A 2d numpy array of shape (seqs_len, num_site_states) of single site
            frequencies after they are regularized.
    """
    reg_single_site_freqs = single_site_freqs
    theta_by_q = np.float64(pseudocount)/np.float64(num_site_states)
    for i in range(seqs_len):
        for a in range(num_site_states):
            reg_single_site_freqs[i, a] = theta_by_q + \
                (1.0 - pseudocount)*reg_single_site_freqs[i, a]
    return reg_single_site_freqs


def compute_pair_site_freqs(alignment_data=None, num_site_states=None, seqs_weight=None):
    # this function is directly from PYDCA.. with only the following corrections:
	# -- Change 1: indexing shift in outer amino acid for loop to accomodate our indexing for amino acids (A : 0 vs A : 1 in PYDCA) 
	# -- Change 2: indexing shift in inner amino acid for loop to accomodate our indexing for amino acids (A : 0 vs A : 1 in PYDCA) 

    """Computes pair site frequencies for an alignmnet data.

    Parameters
    ----------
        alignment_data : np.array()
            A 2d numpy array conatining alignment data. The residues in the
            alignment are in integer representation.
        num_site_states : int
            The number of possible states including gap state that sequence
            sites can accomodate. It must be an integer
        seqs_weight:
            A 1d numpy array of sequences weight

    Returns
    -------
        pair_site_freqs : np.array()
            A 3d numpy array of shape
            (num_pairs, num_site_states, num_site_states) where num_pairs is
            the number of unique pairs we can form from sequence sites. The
            pairs are assumed to in the order (0, 1), (0, 2) (0, 3), ...(0, L-1),
            ... (L-1, L). This ordering is critical and any change must be
            documented.
    """
    alignment_shape = alignment_data.shape
    num_seqs = alignment_shape[0]
    seqs_len = alignment_shape[1]
    num_site_pairs = (seqs_len -1)*seqs_len/2
    num_site_pairs = np.int64(num_site_pairs)
    m_eff = np.sum(seqs_weight)
    pair_site_freqs = np.zeros(
        shape=(num_site_pairs, num_site_states - 1, num_site_states - 1),
        dtype = np.float64
    )
    for i in range(seqs_len - 1):
        column_i = alignment_data[:, i]
        for j in range(i+1, seqs_len):
            pair_site = int((seqs_len * (seqs_len - 1)/2) - (seqs_len - i) * ((seqs_len - i) - 1)/2  + j  - i - 1)
            column_j = alignment_data[:, j]
            #for a in range(1, num_site_states):
            for a in range(num_site_states-1): # ECC Change 1: shifting indexing (see compute_single_site_freqs() above)
                count_ai = column_i==a
                # for b in range(1, num_site_states):
                for b in range(num_site_states-1): # ECC Change 2: shifting indexing (see compute_single_site_freqs() above)
                    count_bj = column_j==b
                    count_ai_bj = count_ai * count_bj
                    freq_ia_jb = np.sum(count_ai_bj*seqs_weight)
                    #pair_site_freqs[pair_site, a-1, b-1] += freq_ia_jb/m_eff
                    pair_site_freqs[pair_site, a, b] += freq_ia_jb/m_eff # ECC Change 3: filling matrices no longer requires -1 because of our indexing
    return pair_site_freqs 

def get_reg_pair_site_freqs(pair_site_freqs = None, seqs_len = None,
        num_site_states = None, pseudocount = None):
    """Regularizes pair site frequencies

    Parameters
    ----------
        pair_site_freqs : np.array()
            A 3d numpy array of shape (num_unique_site_pairs, num_site_states -1,
            num_site_states -1) containing raw pair site frequency counts where
            num_unique_site_pairs is the total number of unique site pairs
            excluding self pairing. Note that the order in with the pairing is
            done is important. It must be taken in (0, 1), (0,2), ...,
            (0, seqs_len-1), (1, 2)... order. Note that this data does not
            contain pairings with gap states.
        seqs_len : int
            The length of sequences in the alignment.
        num_site_states : int
            The total number of states that a site in the sequences can
            accommodate. This includes gap states.

    Returns
    -------
        reg_pair_site_freqs : np.array()
            A numpy array of shape the same as pair_site_freqs
    """
    reg_pair_site_freqs = pair_site_freqs
    theta_by_qsqrd = pseudocount/float(num_site_states * num_site_states)
    pair_counter = 0
    for i in range(seqs_len - 1):
        for j in range(i + 1, seqs_len):
            for a in range(num_site_states-1):
                for b in range(num_site_states-1):
                    reg_pair_site_freqs[pair_counter, a, b] = theta_by_qsqrd + \
                        (1.0 - pseudocount)*reg_pair_site_freqs[pair_counter, a, b]
            pair_counter += 1
    return reg_pair_site_freqs

def construct_corr_mat(reg_fi = None, reg_fij = None, seqs_len = None,
        num_site_states = None):
    """Constructs correlation matrix from regularized frequency counts.

    Parameters
    ----------
        reg_fi : np.array()
            A 2d numpy array of shape (seqs_len, num_site_states) of regularized
            single site frequncies. Note that only fi[:, 0:num_site_states-1] are
            used for construction of the correlation matrix, since values
            corresponding to fi[:, num_site_states]  are the frequncies of gap
            states.
        reg_fij : np.array()
            A 3d numpy array of shape (num_unique_pairs, num_site_states -1,
            num_site_states - 1), where num_unique_pairs is the total number of
            unique site pairs execluding self-pairings.
        seqs_len : int
            The length of sequences in the alignment
        num_site_states : int
            Total number of states a site in a sequence can accommodate.

    Returns
    -------
        corr_mat : np.array()
            A 2d numpy array of shape (N, N)
            where N = seqs_len * num_site_states -1
    """
    corr_mat_len = seqs_len * (num_site_states - 1)
    corr_mat = np.zeros((corr_mat_len, corr_mat_len), dtype=np.float64)
    pair_counter = 0
    for i in range(seqs_len):
        site_i = i * (num_site_states - 1)
        for j in range(i, seqs_len):
            site_j = j * (num_site_states - 1)
            for a in range(num_site_states - 1):
                row = site_i + a
                for b in range(num_site_states -1):
                    col = site_j + b
                    if i==j:
                        fia, fib = reg_fi[i, a], reg_fi[i, b]
                        corr_ij_ab = fia*(1.0 - fia) if a == b else -1.0*fia*fib
                    else:
                        corr_ij_ab = reg_fij[pair_counter, a, b] - reg_fi[i, a] * reg_fi[j, b]
                    corr_mat[row, col] = corr_ij_ab
                    corr_mat[col, row] = corr_ij_ab
            if i != j: pair_counter += 1

    return corr_mat

def compute_couplings(corr_mat = None):
    """Computes the couplings by inverting the correlation matrix

    Parameters
    ----------
        corr_mat : np.array()
            A numpy array of shape (N, N) where N = seqs_len *(num_site_states -1)
            where seqs_len  is the length of sequences in the alignment data and
            num_site_states is the total number of states a site in a sequence
            can accommodate, including gapped states.

    Returns
    -------
        couplings : np.array()
            A 2d numpy array of the same shape as the correlation matrix. Note
            that the couplings are the negative of the inverse of the
            correlation matrix.
    """
    couplings = np.linalg.inv(corr_mat)
    couplings = -1.0*couplings
    return couplings




def frequency_wPYDCA(s0,q,theta,pseudo_weight, seq_weight_outfile=None,first10=False):
    # Calulated the single site and pairsite frequency for Mean Field model
    # 	-- ALSO: includes PYDCA version of calculation for verification of method


    # q --> num site states (21)
    # theta --> minimum percent differnce columns (1-seqid)(.2)
    # pseudo_weight --> lambda equivalent (.5)

    n, l = s0.shape # n --> number of sequences, l --> number of aa in each sequence
    print(s0.shape)

    # hamming distance  -- sequences weight calulation
    dst = distance.squareform(distance.pdist(s0, 'hamming'))
    seq_ints = (dst < theta).sum(axis=1).astype(float)

    # ma_inv = 1/(1+(dst < theta).sum(axis=1).astype(float)) ## tai's version
    ma_inv = 1/((dst < theta).sum(axis=1).astype(float))  ## ECC CHANGE - not adding 1 for identity distance since distance.squareform does that already

    print('ma_inv (sequences weight shape: ', ma_inv.shape)
    meff_tai = ma_inv.sum()
    print('m_eff for our MF = ', meff_tai)


    # ------------------------------------------------------------------------------------------------------------------------------------------------------- #
    # If sequence weight file is provided (assume PYDCA computation is desired) compute PYDCA sequence weigths, single site frequence and pair site frequency..
    if seq_weight_outfile is not None:
        try:
            # pydca -- sequences weight calculation
            seqs_weight = compute_sequences_weight(alignment_data = s0, seqid = float(1.-theta), outfile=seq_weight_outfile)

            # pydca -- single site frequency calculation
            single_site_freqs = compute_single_site_freqs(alignment_data=s0, num_site_states=21, seqs_weight=seqs_weight)


        except(ValueError): # likely from compute_sequences_weight loading an incorrect/old sequence weight file. 

            # pydca -- sequences weight calculation -- force compuation
            seqs_weight = compute_sequences_weight(alignment_data = s0, seqid = float(1.-theta), outfile=seq_weight_outfile, create_new=True)
    
            # pydca -- single site frequency calculation
            single_site_freqs = compute_single_site_freqs(alignment_data=s0, num_site_states=21, seqs_weight=seqs_weight)



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
                        fij_pydca[pair_site, a, b] += freq_ia_jb/meff
        np.save('fij_pydca.npy', fij_pydca)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------- #


    # fi_true:
    fi_true = np.zeros((l,q))
    fi_count = np.zeros((l,q))
    for t in range(n):
        for i in range(l):
            freq_ia = ma_inv[t] 
            fi_true[i,s0[t,i]] += freq_ia  # set correct location (dictated by amino acid number) to sequence weight scaled by effective number of sequences
            fi_count[i,s0[t,i]] += 1  # get aa frequency counts for comparison with pydca

    fi_true /= meff_tai # # / meff_tai  # adding division here, step by step, instead of at the end ## ECC CHANGE


    #for i in range(l):
    #    for a in range(q-1):
    #        if fi_true[i,a] > 0.:
    #            print('site %d-%d freq and count: ' % (i, a), fi_true[i,a], fi_count[i, a])



    # done in line ecc - 12/28/21 added in loop ## ECC CHANGE
    # fi_true /= meff_tai

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

    #for i in range(l):
    #    for a in range(q-1):
    #        if fi[i,a] > 0.:
    #            print('site %d-%d regularized freq: ' % (i, a), fi[i,a])
    #print('pseudo weight = ', pseudo_weight)



    scra = np.eye(q)
    for i in range(l):
        for alpha in range(q):
            for beta in range(q):
                fij[i,i,alpha,beta] = (1 - pseudo_weight)*fij_true[i,i,alpha,beta] \
                + pseudo_weight/q*scra[alpha,beta] 
                #if first10:
                #    print('freq for %d-%d, %d-%d:' %(i,s0[t,j],j,s0[t,j]), fij_true[i,j,s0[t,i],s0[t,j]])




    if seq_weight_outfile is not None:
        return fi,fij, fi_pydca, fij_pydca, ma_inv, seq_ints
    else:
        return fi, fij



def frequency(s0,q,theta,pseudo_weight, seq_weight_outfile=None,first10=False):
    # q --> num site states (21)
    # theta --> minimum percent differnce columns (1-seqid)(.2)
    # pseudo_weight --> lambda equivalent (.5)

    n, l = s0.shape # n --> number of sequences, l --> number of aa in each sequence
    print(s0.shape)

    # hamming distance  -- sequences weight calulation
    dst = distance.squareform(distance.pdist(s0, 'hamming'))
    seq_ints = (dst < theta).sum(axis=1).astype(float)
    # ma_inv = 1/(1+(dst < theta).sum(axis=1).astype(float)) ## tai's version
    ma_inv = 1/((dst < theta).sum(axis=1).astype(float))  ## ECC CHANGE - not adding 1 for identity distance since distance.squareform does that already
    print('ma_inv (sequences weight shape: ', ma_inv.shape)
    meff_tai = ma_inv.sum()
    print('tais meff = %f' % meff_tai)

    # pydca             -- sequences weight calculation
    if seq_weight_outfile is not None:
        try:
            seqs_weight = compute_sequences_weight(alignment_data = s0, seqid = float(1.-theta), outfile=seq_weight_outfile)
            meff = np.sum(seqs_weight)
            fi_pydca = np.zeros((l, q))
    
            for i in range(l):
                for a in range(q):#we need gap states single site freqs too
                    column_i = s0[:,i]
                    freq_ia = np.sum((column_i==a)*seqs_weight)
                    fi_pydca[i, a-1] = freq_ia/meff
        except(ValueError): # likely from compute_sequences_weight loading an incorrect/old sequence weight file. 
            seqs_weight = compute_sequences_weight(alignment_data = s0, seqid = float(1.-theta), outfile=seq_weight_outfile, create_new=True)
            meff = np.sum(seqs_weight)
            fi_pydca = np.zeros((l, q))
    
            for i in range(l):
                for a in range(q):#we need gap states single site freqs too
                    column_i = s0[:,i]
                    freq_ia = np.sum((column_i==a)*seqs_weight)
                    fi_pydca[i, a-1] = freq_ia/meff

        # print('fi_pydca.shape: ', fi_pydca.shape)

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
                        fij_pydca[pair_site, a, b] += freq_ia_jb/meff
        np.save('fij_pydca.npy', fij_pydca)



    # fi_true:
    fi_true = np.zeros((l,q))
    fi_count = np.zeros((l,q))
    for t in range(n):
        for i in range(l):
            freq_ia = ma_inv[t] 
            fi_true[i,s0[t,i]] += freq_ia  # set correct location (dictated by amino acid number) to sequence weight scaled by effective number of sequences
            fi_count[i,s0[t,i]] += 1  # get aa frequency counts for comparison with pydca

    fi_true /= meff_tai # # / meff_tai  # adding division here, step by step, instead of at the end ## ECC CHANGE


    #for i in range(l):
    #    for a in range(q-1):
    #        if fi_true[i,a] > 0.:
    #            print('site %d-%d freq and count: ' % (i, a), fi_true[i,a], fi_count[i, a])


    print('meff for our MF = ', meff_tai)

    # done in line ecc - 12/28/21 added in loop ## ECC CHANGE
    # fi_true /= meff_tai

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

    #for i in range(l):
    #    for a in range(q-1):
    #        if fi[i,a] > 0.:
    #            print('site %d-%d regularized freq: ' % (i, a), fi[i,a])
    #print('pseudo weight = ', pseudo_weight)



    scra = np.eye(q)
    for i in range(l):
        for alpha in range(q):
            for beta in range(q):
                fij[i,i,alpha,beta] = (1 - pseudo_weight)*fij_true[i,i,alpha,beta] \
                + pseudo_weight/q*scra[alpha,beta] 
                #if first10:
                #    print('freq for %d-%d, %d-%d:' %(i,s0[t,j],j,s0[t,j]), fij_true[i,j,s0[t,i],s0[t,j]])




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
    #di_pydca = direct_info(w_pydca,fi_pydca,q,l)
    di_pydca = 'Useless'

    # ------------------------------------------------------------------------------------------------------------------------------------------------------- #
    # --------------------------------------------------- PYDCA Calculations -------------------------------------------------------------------------------- #
    # ------------------------------------------------------------------------------------------------------------------------------------------------------- #
    # Commands taken from PYDCA.meanfield_dca.get_site_pair_di_score() to obtain computed direct information (DI) scores 
    # input parameters taken from our code and put in pydca-variable-format
    pydca_pseudocount = pseudo_weight 
    pydca_seq_id = 1 - theta
    seqs_len = len(s0[0])

    # ------------ Single Site Frequencies ------------ #
    # If sequence weight file is provided (assume PYDCA computation is desired) compute PYDCA sequence weigths, single site frequence and pair site frequency..
    if seq_wt_outfile is not None:
        try:
            # pydca -- sequences weight calculation
            seqs_weight = compute_sequences_weight(alignment_data = s0, seqid = float(1.-theta), outfile=seq_wt_outfile)
            # pydca -- single site frequency calculation
            single_site_freqs = compute_single_site_freqs(alignment_data=s0, num_site_states=q, seqs_weight=seqs_weight)
        except(ValueError): # likely from compute_sequences_weight loading an incorrect/old sequence weight file. 
            # pydca -- sequences weight calculation -- force compuation
            seqs_weight = compute_sequences_weight(alignment_data = s0, seqid = float(1.-theta), outfile=seq_wt_outfile, create_new=True)
            # pydca -- single site frequency calculation
            single_site_freqs = compute_single_site_freqs(alignment_data=s0, num_site_states=q, seqs_weight=seqs_weight)
    reg_fi = get_reg_single_site_freqs(single_site_freqs=single_site_freqs, seqs_len=seqs_len, num_site_states=q, pseudocount=pydca_pseudocount)
    # ------------------------------------------------- #

    # ------------ Pair Site Frequenceies ------------- #
    pair_site_freqs = compute_pair_site_freqs(alignment_data=s0, num_site_states=q, seqs_weight=seqs_weight)
    reg_fij = get_reg_pair_site_freqs(pair_site_freqs=pair_site_freqs, seqs_len=seqs_len, num_site_states=q, pseudocount=pydca_pseudocount)
    # ------------------------------------------------- #

    # ------------ Correlation/Couplings -------------- #
    corr_mat = construct_corr_mat(reg_fi, reg_fij, seqs_len=seqs_len, num_site_states=q)
    couplings = compute_couplings(corr_mat=corr_mat)
    # ------------------------------------------------- #

    c_pydca = corr_mat
    w_pydca = couplings

    # ------------------------------------------------------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------------------------------------------------------------------------------------------------------------------- #
    # ------------------------------------------------------------------------------------------------------------------------------------------------------- #



    # fields not required for now.
    # fields_ij = compute_two_site_model_fields(couplings, reg_fi)

    
    return di, fi, fij, c, c_inv, w, w2d, reg_fi_pydca, reg_fij_pydca, c_pydca, c_inv_pydca, w_pydca, w2d_pydca, di_pydca, ma_inv,seq_ints

#np.savetxt('%s_di.dat'%(pfam_id),di,fmt='% f')
#plt.imshow(di,cmap='rainbow',origin='lower')

