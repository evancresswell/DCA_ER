#mycharray,interval_permute,mybin,weighted_sum,plot3dplot,binarize
import sys
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
##matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math as m
import csv
heat_colors = 20
dpi=100
def binarize(lab,llist):
    ret,sum_ret = [],0
    for item in llist:
        if lab==item:
            ret.append(1)
            sum_ret += 1
        else:
            ret.append(0)
    return ret,sum_ret

def myQR(a,num_seq,num_pos,num_aa):
    eigen_limit=1e-12
    d,u = np.linalg.eigh(a)
    d = npfloat(d)
    u = npfloat(u)
    dorder = np.argsort(d)[::-1]
    d = d[dorder]
    vt = u.T[dorder,:]
##    u,d,vt=np.linalg.svd(a)
    irange = myRange(num_seq,num_pos,num_aa,d,vt)
    if irange <=0:
        irange=1
    dsub = d[:irange]
    if dsub[0]<eigen_limit:
        dsub = dsub + eigen_limit
    vtsub = vt[:irange,:]
    d2E = np.dot(vtsub.T,np.dot(np.diag(1/dsub),vtsub))
    return irange,npfloat(dsub),npfloat(vtsub),npfloat(d2E)
def keep_positions(seqs_and_wts,keeper):
    s_and_w = []
    for i in range(len(seqs_and_wts)):
        s_and_w.insert(i,[])
        s = ''
        for j in range(len(keeper)):
            s = s + seqs_and_wts[i][0][keeper[j][0]:keeper[j][1]+1]
        s_and_w[i].insert(0,s)
        s_and_w[i].insert(1,seqs_and_wts[i][1])
    return s_and_w


def plot3dplot(low_coords,sizes,cols,filestr,elev=45,azim=45):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection = '3d')
    ax.scatter(low_coords[:,0],low_coords[:,1],low_coords[:,2],s=sizes,c=cols)
    ax.view_init(elev,azim)
    plt.savefig(filestr+'_3d_'+str(elev)+str(azim)+'.tiff',format='tiff')
    plt.close()
    
def info_from_freqs(bin_freqs):
    return myInfo(bin_freqs)
def myInfo(q,precision=1e-15): #compute information in frequencies
    shape = q.shape
    if np.min(q)>0:
        qnew = np.copy(q)
    else:
        qnew = np.copy(q)
        qnew[q==0.0] = precision
    qnew = - qnew*np.log(qnew)
    return np.sum(qnew,axis=-1) #sum over the last axis

    
def low_coords(binvectors,vt,num_low,c1_choice,vt_choice):
    sign_choice = np.zeros(num_low)
    c1_low = c1_choice.reshape(binvectors.shape[1])
    for j in range(num_low):
        sign_choice[j] = np.sign(np.inner(vt_choice[j,:],vt[j,:]))
    bin_choice = np.copy(binvectors)
    for j in range(binvectors.shape[0]):
        bin_choice[j,:] = bin_choice[j,:] - c1_low
    vt_coords = np.inner(bin_choice,vt_choice[:num_low,:])
    for j in range(binvectors.shape[0]):
        vt_coords[j,:] = sign_choice*vt_coords[j,:]
    return vt_coords.reshape(num_low*binvectors.shape[0])

def mylog(a):
    if a>1e-15:
        return m.log(float(a))
    else:
        return -40.0
npmylog = np.frompyfunc(mylog,1,1)
def vlog(a):
    vl = np.vectorize(mylog)
    return vl(a)

def permuted_strings(strings):
    return charray_to_strings(columnwise_permute(mycharray(strings)))

def total_permuted_strings(strings):
    row_permute = charray_to_strings(columnwise_permute(mycharray(strings).T).T)
    return permuted_strings(charray_to_strings(columnwise_permute(mycharray(row_permute))))
                        
def mycharray(strings,aa=[]):
    num_seq = len(strings)
    num_pos =len(strings[0])
    num_aa = len(aa)
    charray = np.empty(shape=(num_seq,num_pos),dtype='S')
    for i in range(num_seq):
        for j in range(num_pos):
            charray[i,j]=strings[i][j]
##        if i==1:
##            print strings[i],charray[1,:]
#    print charray[0:20]
    return charray
def scramble(string):
    num_char = len(string)
    x = np.arange(num_char)
    np.random.shuffle(x)
    return ''.join([string[j] for j in x])
def interval_permute_one(string,interval):
    num_cols = len(string)
    num_seg = num_cols/interval
    offset = np.random.randint(0,interval-1)
    new_string = scramble(string[:offset+1])
    for seg in range(num_seg):
        new_string += scramble(string[offset+1+seg*interval:offset+1+(seg+1)*interval])
    new_string += scramble(string[offset+1+num_seg*interval:])
    return new_string
def interval_permute(strings,interval):
    ret = []
    for string in strings:
        ret.append(interval_permute_one(string,interval))
    return ret
    
def columnwise_permute(arr):
    num_rows,num_cols = arr.shape
    permarray = np.empty_like(arr)
    for j in range(num_cols):
        permarray[:,j] = arr[:,j].take(np.random.permutation(num_rows))
#    print permarray[0:20]
    return permarray
def charray_to_strings(arr):
    num_seq,num_pos = arr.shape
    strarray=[] #np.empty(shape=(num_seq,),dtype='str')
    for i in range(num_seq):
        strar = ''.join(np.array(arr[i,:]))
        strarray.insert(i,strar)
    return strarray
def mybin(seq1,residues): #
    num_char=len(residues)
    num_pos = len(seq1)
    v = np.zeros((num_pos,num_char))
#    print residues,seq1
    for i in range(0,num_pos):
        v[i,residues[seq1[i]]] = 1
    return v.reshape((num_pos*num_char))
def mybin_reduce(seq1,residues,bin_totals):
    return 0
def myIterateStep(J,binvectors,M,wts):
    # J is assumed in the shape (num_pos,num_aa)
    shape = J.shape
    jdot = np.inner(binvectors,(J).reshape((binvectors.shape[1]))) #J dot sequence
    wts_j = (wts)*npexp(M*jdot) #np.inner(wts,np.diag(np.exp(M*jdot)))
    #np.array([np.exp(M*jdot)[i]*wts[i] for i in range(binvectors.shape[0])])
    wts_j = npfloat(((wts_j))/np.sum((wts_j))) #normalize wts
#    print 'wts ', wts_j[0:4]
    return (np.dot(wts_j,binvectors).reshape(shape)),wts_j #weighted frequencies

def weighted_sum(J,vectors,h=0):
#    print J[:10],vectors[:5,:10],h[:5]
    j_wts = np.exp(-h+np.sum((vectors.T*J[:,np.newaxis]),axis=0))
    sum_wts = np.sum(j_wts)
    return vectors*j_wts[:,np.newaxis]/sum_wts

def compute_J(q0,binvectors,M,wts,precision=1e-15,position_dep=False,numstep_limit=10,best_motifs=0):
    q = myQ(q0,position_dep,best_motifs=best_motifs)
    min_wts = np.min(wts)
    J = npmathify(np.zeros(q.shape))
    mpwts = npmathify(wts)
    p,wts_j = myIterateStep(J,binvectors,M,(mpwts))
    wts_j1 = np.ones(wts.shape)
    numsteps=0
    while (np.max(np.abs(wts_j-wts_j1))>precision) and (numsteps<=numstep_limit):
#        print 'iteration ', numsteps,wts_j[:3],np.max(np.abs(wts_j-wts_j1))
#        Jnew = myDGamma(p,q,M,position_dep,best_motifs=best_motifs)
#below forces multinomial, above line switches to Dirchlet when position_dep
        Jnew = myDGamma(p,q,M,False,best_motifs=best_motifs)
        wts_j1 = wts_j
        numsteps=numsteps+1
        p,wts_j = myIterateStep(Jnew,binvectors,M,mpwts)
    return npfloat(p),npfloat(wts_j),npfloat(Jnew),(numsteps<numstep_limit)
        
