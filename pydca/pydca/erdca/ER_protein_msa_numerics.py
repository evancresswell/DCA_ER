from scipy import linalg
import numpy as np
from scipy.spatial import distance
from joblib import Parallel, delayed
from sklearn.preprocessing import OneHotEncoder
from scipy.special import erf as sperf


"""This module implements computationally costly routines while performing
ER.

"""
def compute_sequence_weight(i0,alignment_data, seqs_len,num_seqs,seqs_weight,seqid):
    seq_i = alignment_data[i0]
    for j in range(num_seqs):
        seq_j = alignment_data[j]
        iid = np.sum(seq_i==seq_j)
        if np.float64(iid)/np.float64(seqs_len) > seqid:
            seqs_weight[i0] += 1

# joblib parallelized compute_sequences_weight from orginal Numba PYDCA version
#@jit(nopython=True, parallel=True)
def compute_sequences_weight(alignment_data=None, seqid=None,num_threads=1):
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
        sequence_identity : float
            Value at which beyond this sequences are considered similar. Typical
            values could be 0.7, 0.8, 0.9 and so on

    Returns
    -------
        seqs_weight : np.array()
            A 1d numpy array containing computed weights. This array has a size
            of the number of sequences in the alignment data.
    """
    alignment_shape = alignment_data.shape
    num_seqs = alignment_shape[0]
    seqs_len = alignment_shape[1]
    seqs_weight = np.zeros((num_seqs,), dtype=np.float64)
    #count similar sequences
    print('seqs_weight.shape: ',seqs_weight.shape)
    print('seqs_weight: ',seqs_weight[0])
     
    seqs_weight = Parallel(n_jobs = num_threads)(delayed(compute_sequence_weight)\
              (i0,alignment_data, seqs_len,num_seqs,seqs_weight,seqid)\
              for i0 in range(num_seqs))
    print('seqs_weight.shape: ',np.shape(seqs_weight))
    print('seqs_weight: ',seqs_weight[0])

    #compute the weight of each sequence in the alignment
    for i in range(num_seqs): seqs_weight[i] = 1.0/float(seqs_weight[i])
    return seqs_weight


#========================================================================================================#
#------------------------------------- Expectation Reflection -------------------------------------------#
#========================================================================================================#
# ------- Author: Evan Cresswell-Clay ---------- Date: 8/24/2020 ----------------------------------------#
#========================================================================================================#



def er_fit(x,y_onehot,niter_max,l2,couplings= None):       
    l,n = x.shape
    m = y_onehot.shape[1] # number of categories
    
    x_av = np.mean(x,axis=0)
    dx = x - x_av
    c = np.cov(dx,rowvar=False,bias=True)

    c += l2*np.identity(n)/(2*l)
    c_inv = linalg.pinvh(c)

    H0 = np.zeros(m)
    W = np.zeros((n,m))

    for i in range(m):
        y = y_onehot[:,i]  # y = {0,1}
        y1 = 2*y - 1       # y1 = {-1,1}
        # initial values
        h0 = 0.

        # If couplings (ie initial weight state) is passed, use it otherwise random.
        if couplings is not None: 
            w = couplings[:,i]
        else: 
            w = np.random.normal(0.0,1./np.sqrt(n),size=(n))
        
        cost = np.full(niter_max,100.)
        for iloop in range(niter_max):
            h = h0 + x.dot(w)
            y1_model = np.tanh(h/2.)    

            cost[iloop] = ((y1[:]-y1_model[:])**2).mean()

            if iloop > 0 and cost[iloop] >= cost[iloop-1] : break
                        
            # update local field
            t = h!=0    
            h[t] *= y1[t]/y1_model[t]
            h[~t] = 2*y1[~t]

            # find w from h    
            h_av = h.mean()
            dh = h - h_av 
            dhdx = dh[:,np.newaxis]*dx[:,:]

            dhdx_av = dhdx.mean(axis=0)
            w = c_inv.dot(dhdx_av)
            h0 = h_av - x_av.dot(w)

        H0[i] = h0
        W[:,i] = w
    
    return H0,W  

def er_fit_lad(x,y_onehot,niter_max,l2,couplings= None):       
    l,n = x.shape
    m = y_onehot.shape[1] # number of categories
    
    x_av = np.mean(x,axis=0)
    dx = x - x_av
    c = np.cov(dx,rowvar=False,bias=True)

    c += l2*np.identity(n)/(2*l)

    H0 = np.zeros(m)
    W = np.zeros((n,m))

    for i in range(m):
        if couplings is not None:
            H0[i],W[:,i] = fit_LAD(x,y_onehot[:,i],niter_max,l2,couplings=couplings)
        else:
            H0[i],W[:,i] = fit_LAD(x,y_onehot[:,i],niter_max,l2)
    
    return H0,W  
if 0:
    def fit_LAD(x,y,niter_max,l2,couplings = None):      
        # convert 0, 1 to -1, 1
        y1 = 2*y - 1.
       
        #print(niter_max)    
        n = x.shape[1]
        
        x_av = np.mean(x,axis=0)
        dx = x - x_av
        c = np.cov(dx,rowvar=False,bias=True)

        # 2019.07.16:  
        c += l2*np.identity(n) / (2*len(y))

        # initial values
        h0 = 0.

        # If couplings (ie initial weight state) is passed, use it otherwise random.
        if couplings is not None: 
            w = couplings[:,i]
        else: 
            w = np.random.normal(0.0,1./np.sqrt(n),size=(n))
        
        cost = np.full(niter_max,100.)
        for iloop in range(niter_max):
            h = h0 + x.dot(w)
            y1_model = np.tanh(h/2.)    
    
            # stopping criterion
            h_too_neg = h < -15
            h[h_too_neg] = -15.
            p = 1/(1+np.exp(-h))                
            cost[iloop] = ((p-y)**2).mean()
    
            if iloop>0 and cost[iloop] >= cost[iloop-1]: break
    
            # update local field
            t = h!=0    
            h[t] *= y1[t]/y1_model[t]
            h[~t] = 2*y1[~t]
            # 2019.12.26: 
            h0,w = infer_LAD(x,h[:,np.newaxis],regu = l2)

        return h0,w

#-------------------------------------------------------------------------------------------------------#
#------------------------------- LAD ER Implementation (ECC) -------------------------------------------#
#-------------------------------------------------------------------------------------------------------#
def fit_LAD(x,y_onehot,niter_max,l2,couplings = None):      
   
    #print(niter_max)    
    n = x.shape[1]
    m = y_onehot.shape[1] # number of categories
    
    x_av = np.mean(x,axis=0)
    dx = x - x_av
    c = np.cov(dx,rowvar=False,bias=True)

    #------------------ EigenValue Reg ------------------------@
    # Calculate Eigenvalue of covariance matrix
    # Eigenvalue replaces l2 for regularization strength
    cov_eigen = np.linalg.eigvalsh(c)
    eig_hist, eig_ranges = np.histogram(cov_eigen) 

    #cov_eiv = max(cov_eigen)						# largest eigenvalue
    cov_eiv = min(eig_ranges[eig_ranges > 1e-4]) 			# smallest non-zero eigenvalue
    #cov_eiv = sorted(list(set(cov_eigen.flatten().tolist())))[-5]	# 5th largest eigenvalue
    #print('Regularizing using EV of Cov Mat: ',cov_eiv)

    l2 = cov_eiv # replace l2 (fit_LAD passed parmeter)
    #----------------------------------------------------------@

    # 2019.07.16:  
    c += l2*np.identity(n) / (2*len(y_onehot))

    H0 = np.zeros(m)
    W = np.zeros((n,m))
    #print('y_onehot shape: ',y_onehot.shape)


    for i in range(m):
        y = y_onehot[:,i]  # y = {0,1}
        y1 = 2*y - 1       # y1 = {-1,1}

        # initial values
        h0 = 0.
    
        # If couplings (ie initial weight state) is passed, use it otherwise random.
        if couplings is not None: 
            w = couplings[:,i]
        else: 
            w = np.random.normal(0.0,1./np.sqrt(n),size=(n))
       
        cost = np.full(niter_max,100.)
        #for iloop in range(niter_max):
        # instead of n_iter times, we iterate through scalling values of regu
        regu_scalling_vals = [0.2,0.4,0.6,1.]
        cost = np.full(len(regu_scalling_vals),100.) # redefine cost to be regu scalling iterations
        for iloop,regu_coef in enumerate(regu_scalling_vals):
            h = h0 + x.dot(w)
            y1_model = np.tanh(h/2.)    
            #print('h shape ', h.shape)
    
            # stopping criterion
            h_too_neg = h < -15
            h[h_too_neg] = -15.
            p = 1/(1+np.exp(-h))                
            #print('p shape ', p.shape)
            #print('y shape ', y.shape)
            cost[iloop] = ((p-y)**2).mean()
    
            if iloop>0 and cost[iloop] >= cost[iloop-1]: break
    
            # update local field
            t = h!=0    
            h[t] *= y1[t]/y1_model[t]
            h[~t] = 2*y1[~t]
            # 2019.12.26: 
            h0,w = infer_LAD(x,h[:,np.newaxis],regu = regu_coef*l2)

        H0[i] = h0
        W[:,i] = w

    return H0,W

if 0:
    def infer_LAD(x, y, regu=0.1,tol=1e-8, max_iter=5000):
    ## 2019.12.26: Jungmin's code    
        weights_limit = sperf(1e-10)*1e10
        
        s_sample, s_pred = x.shape
        s_sample, s_target = y.shape
        
        mu = np.zeros(x.shape[1])

        w_sol = 0.0*(np.random.rand(s_pred,s_target) - 0.5)
        b_sol = np.random.rand(1,s_target) - 0.5

        for index in range(s_target):
            error, old_error = np.inf, 0
            weights = np.ones((s_sample, 1))
            cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, \
                     ddof=0, aweights=weights.reshape(s_sample))
            cov_xx, cov_xy = cov[:s_pred,:s_pred],cov[:s_pred,s_pred:(s_pred+1)]
            counter = 0
            while np.abs(error-old_error) > tol and counter < max_iter:
                counter += 1
                old_error = np.mean(np.abs(b_sol[0,index] + x.dot(w_sol[:,index]) - y[:,index]))
                sigma_w = np.std(w_sol[:,index])
                
                w_eq_0 = np.abs(w_sol[:,index]) < 1e-10
                mu[w_eq_0] = 2./np.sqrt(np.pi)
            
                mu[~w_eq_0] = sigma_w*sperf(w_sol[:,index][~w_eq_0]/sigma_w)/w_sol[:,index][~w_eq_0]
                                    
                w_sol[:,index] = np.linalg.solve(cov_xx + regu*np.diag(mu),cov_xy).reshape(s_pred)
            
                b_sol[0,index] = np.mean(y[:,index]-x.dot(w_sol[:,index]))
                weights = (b_sol[0,index] + x.dot(w_sol[:,index]) - y[:,index])
                sigma = np.std((weights))
                error = np.mean(np.abs(weights))
                weights_eq_0 = np.abs(weights) < 1e-10
                weights[weights_eq_0] = weights_limit
                weights[~weights_eq_0] = sigma*sperf(weights[~weights_eq_0]/sigma)/weights[~weights_eq_0]
                weights /= np.mean(weights) #now the mean weight is 1.0
                cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, \
                     ddof=0, aweights=weights.reshape(s_sample))
                cov_xx, cov_xy = cov[:s_pred,:s_pred],cov[:s_pred,s_pred:(s_pred+1)]
        return b_sol[0][0],w_sol[:,0] # for only one target case

# ECC implementation to speed up - 12/15
def infer_LAD(x, y, regu=0.1,tol=1e-8, max_iter=5000):
## 2019.12.26: Jungmin's code    
    #weights_limit = sperf(1e-10)*1e10
    weights_limit = (1e-10)*1e10
    
    s_sample, s_pred = x.shape
    s_sample, s_target = y.shape
    
    mu = np.zeros(x.shape[1])

    w_sol = 0.0*(np.random.rand(s_pred,s_target) - 0.5)
    b_sol = np.random.rand(1,s_target) - 0.5

    for index in range(s_target):
        error, old_error = np.inf, 0
        weights = np.ones((s_sample, 1))
        cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, \
                     ddof=0, aweights=weights.reshape(s_sample))
        cov_xx, cov_xy = cov[:s_pred,:s_pred],cov[:s_pred,s_pred:(s_pred+1)]
        counter = 0
        while np.abs(error-old_error) > tol and counter < max_iter:
            counter += 1


            old_error = np.mean(np.abs(b_sol[0,index] + x.dot(w_sol[:,index]) - y[:,index]))
            sigma_w = np.std(w_sol[:,index])
                
            w_eq_0 = np.abs(w_sol[:,index]) < 1e-10
            mu[w_eq_0] = 2./np.sqrt(np.pi)
        
            #mu[~w_eq_0] = sigma_w*sperf(w_sol[:,index][~w_eq_0]/sigma_w)/w_sol[:,index][~w_eq_0]
            mu[~w_eq_0] = sigma_w*(w_sol[:,index][~w_eq_0]/sigma_w)/w_sol[:,index][~w_eq_0]
                                                        
            w_sol[:,index] = np.linalg.solve(cov_xx + regu * np.diag(mu),cov_xy).reshape(s_pred)
        
            b_sol[0,index] = np.mean(y[:,index]-x.dot(w_sol[:,index]))
            weights = (b_sol[0,index] + x.dot(w_sol[:,index]) - y[:,index])
            sigma = np.std((weights))
            error = np.mean(np.abs(weights))
            weights_eq_0 = np.abs(weights) < 1e-10
            weights[weights_eq_0] = weights_limit

            #weights[~weights_eq_0] = sigma*sperf(weights[~weights_eq_0]/sigma)/weights[~weights_eq_0]
            weights[~weights_eq_0] = sigma*(weights[~weights_eq_0]/sigma)/weights[~weights_eq_0]
            
            weights /= np.mean(weights) #now the mean weight is 1.0
            cov = np.cov(np.hstack((x,y[:,index][:,None])), rowvar=False, \
                         ddof=0, aweights=weights.reshape(s_sample))
            cov_xx, cov_xy = cov[:s_pred,:s_pred],cov[:s_pred,s_pred:(s_pred+1)]
    return b_sol[0][0],w_sol[:,0] # for only one target case

if 0:
	def predict_w_LADER(s,i0,i1i2,niter_max,l2):
	    i1,i2 = i1i2[i0,0],i1i2[i0,1]

	    x = np.hstack([s[:,:i1],s[:,i2:]])
	    y = s[:,i1:i2]

	    # h01,w1 = er_fit(x,y,niter_max,l2) # old inference
	    h01,w1 = er_fit_lad(x,y,niter_max,l2)

	    return h01,w1
def predict_w_LADER(s,i0,i1i2,niter_max,l2,couplings = None):
    i1,i2 = i1i2[i0,0],i1i2[i0,1]

    x = np.hstack([s[:,:i1],s[:,i2:]])
    y = s[:,i1:i2]

    # h01,w1 = er_fit(x,y,niter_max,l2) # old inference
    if couplings is not None:
        y_couplings = np.delete(couplings,[range(i1,i2)],0)	# remove subject rows  from original coupling matrix 
        y_couplings = np.delete(y_couplings,[range(i1,i2)],1)   # remove subject columns from original coupling matrix 

        h01,w1 = fit_LAD(x,y,niter_max,l2,couplings = y_couplings)

    else:
        h01,w1 = fit_LAD(x,y,niter_max,l2)

    return h01,w1



def compute_lader_weights(n_var,s,i1i2,num_threads=1,couplings=None):
    # parallel
    # parallel
    print('Compute ER weights in parallel using %d threads for %d variables'%(num_threads,n_var))
    print('matrix s: shape: ',s.shape,'\n\n')
    res = Parallel(n_jobs = num_threads)(delayed(predict_w_LADER)\
            (s, i0, i1i2, niter_max=10, l2=100.0)\
            for i0 in range(n_var))
    print('Done Parallel processing')
    return res 

# direct information from w, ONLY apply for our method, NOT DCA since w is converted to 2D
def direct_info_value(w2d,fi,q,i1i2):
    # w2d[nm,nm], fi[l,n],q[n]
    n = q.shape[0]


    #ew_all = np.exp(w2d)

    # dealing with RuntimeWarning
    try:
        ew_all = np.exp(w2d)
    except(RuntimeWarning):
        max_w2d = max([max(w) for w in w2d])
        print('subtracting max w2d value: ',max_w2d)
        w2d = w2d - max_w2d
        ew_all = np.exp(w2d)
   
 
    di = np.zeros((n,n))
    tiny = 10**(-100.)
    diff_thres = 10**(-4.)

    for i in range(n-1):
        i1,i2 = i1i2[i,0],i1i2[i,1]
        for j in range(i+1,n):
            j1,j2 = i1i2[j,0],i1i2[j,1]
            #ew = ew_all[i,j,:q[i],:q[j]]
            ew = ew_all[i1:i2,j1:j2]
            #------------------------------------------------------
            # find h1 and h2:

            # initial value
            diff = diff_thres + 1.
            eh1 = np.full(q[i],1./q[i])
            eh2 = np.full(q[j],1./q[j])

            #fi0 = fi[i,0:q[i]]
            #fj0 = fi[j,0:q[j]]
            fi0 = fi[i1:i2]
            fj0 = fi[j1:j2]
                
            for iloop in range(100):
                eh_ew1 = eh2.dot(ew.T)
                eh_ew2 = eh1.dot(ew)

                eh1_new = fi0/(eh_ew1+tiny) # ecc added tiny
                eh1_new /= eh1_new.sum()

                eh2_new = fj0/(eh_ew2+tiny) # ecc added tiny
                eh2_new /= eh2_new.sum()

                diff = max(np.max(np.abs(eh1_new - eh1)),np.max(np.abs(eh2_new - eh2)))

                eh1,eh2 = eh1_new,eh2_new    
                if diff < diff_thres: break        

            # direct information
            eh1eh2 = eh1[:,np.newaxis]*eh2[np.newaxis,:]
            pdir = ew*(eh1eh2)
            pdir /= (pdir.sum()+tiny) # ecc added tiny

            fifj = fi0[:,np.newaxis]*fj0[np.newaxis,:]

            dijab = pdir*np.log((pdir+tiny)/(fifj+tiny))
            di[i,j] = dijab.sum()

    # symmetrize di
    di = di + di.T
    return di

def frequency(s0,q,i1i2,theta=0.2,pseudo_weight=0.5):
    n = s0.shape[1]
    # hamming distance
    dst = distance.squareform(distance.pdist(s0, 'hamming'))
    ma_inv = 1/(1+(dst < theta).sum(axis=1).astype(float))
    meff = ma_inv.sum() 

    onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
    s = onehot_encoder.fit_transform(s0)

    # fi_true:
    fi_true = (ma_inv[:,np.newaxis]*s[:,:]).sum(axis=0)
    fi_true /= meff

    # fi, fij
    fi = np.zeros(q.sum())
    for i in range(n):
        i1,i2 = i1i2[i,0],i1i2[i,1]
        fi[i1:i2] = (1 - pseudo_weight)*fi_true[i1:i2] + pseudo_weight/q[i]

    return fi

def direct_info(s0,w):
    w = (w+w.T)/2

    l,n = s0.shape
    mx = np.array([len(np.unique(s0[:,i])) for i in range(n)])
    #mx = np.array([m for i in range(n)])
    mx_cumsum = np.insert(mx.cumsum(),0,0)
    i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T

    q = mx   # just for convenience
    fi = frequency(s0,q,i1i2)
    di = direct_info_value(w,fi,q,i1i2)
     
    return di

def sort_di(di):
    """
    Returns array of sorted DI values
    """
    ind = np.unravel_index(np.argsort(di,axis=None),di.shape)    
    tuple_list = [((indices[0],indices[1]),di[indices[0],indices[1]]) for i,indices in enumerate(np.transpose(ind))]    
    tuple_list = tuple_list[::-1]
    return tuple_list

def sindex_di(sorted_di,s_index):
    s_index_di = []
    for di_tuple in sorted_di:
        #print(di_tuple ,"-->",((s_index[di_tuple[0][0]], s_index[di_tuple[0][1]]),di_tuple[1]))
        s_index_di.append(((s_index[di_tuple[0][0]], s_index[di_tuple[0][1]]),di_tuple[1]))     
    return s_index_di
        

