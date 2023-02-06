import os.path, sys

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder

import Bio.PDB, warnings
pdb_list = Bio.PDB.PDBList()
pdb_parser = Bio.PDB.PDBParser()
from scipy.spatial import distance_matrix
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

from joblib import Parallel, delayed
import timeit
# %matplotlib inline

import matplotlib.pyplot as plt

# # --- Import our Code ---# #
#import emachine as EM
from direct_info import direct_info

# import data processing and general DCA_ER tools
from data_processing import data_processing_msa2pdb

data_path = Path('/data/cresswellclayec/DCA_ER/furano_periwal_prot')

# Define data directories
DCA_ER_dir = '/data/cresswellclayec/DCA_ER/furano_periwal_prot' # Set DCA_ER directory

data_dir = "%s/protein_data/" % DCA_ER_dir

all_seqs= []
with open('protein_data/all_seqs.fa', 'rU') as f:
    seq_iter = SeqIO.parse(f,'fasta')
    for seq in seq_iter:
        all_seqs.append(seq)

lp_msas = []
lp_fa_prefix = 'protein_data/'
lp_names = ['LPa1', 'LPa2', 'LPa3', 'LPa4', 'LPa5', 'LPa6', 'LPa7' ]
total_len = 0
lp_files = ['cls1.1.fasta','cls1.2.fasta','cls123.3.fasta','cls1ab234.4.fasta','cls123.5.fasta','cls1abc234.6.fasta','cls123.7.fasta']
lp_ids = []
for i,filename in enumerate(lp_files):
    lp_msas.append([])
    lp_ids.append([])
    print('Loading MSA for ',lp_names[i])
    with open(lp_fa_prefix+filename, 'rU') as f:
        seq_iter = SeqIO.parse(f,'fasta')
        for seq in seq_iter:
#             print(seq)
            lp_msas[-1].append(seq.seq) 
            lp_ids[-1].append(seq.id)
    f.close()
#     print(lp_msas[-1][0][:5])
    print(len(lp_msas[-1]))
    total_len += len(lp_msas[-1])
print('number of all individual LPa MSA sequences: ',total_len)
print('number of all aligned sequences: ',len(all_seqs))

s0 = np.array([seq.seq for seq in all_seqs])
print(s0.shape)

onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
print('s0: ',s0.shape,'\n',s0)
s = onehot_encoder.fit_transform(s0)
print('s: ',s.shape,'\n',s)


# number of positions
n_var = s0.shape[1]
n_seq = s0.shape[0]

print("Number of residue positions:",n_var)
print("Number of sequences:",n_seq)

# number of aminoacids at each position
mx = np.array([len(np.unique(s0[:,i])) for i in range(n_var)])
#mx = np.array([m for i in range(n_var)])
print("Number of different amino acids at each position",mx)

mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T
# print(\"(Sanity Check) Column indices of first and (\",i1i2[0],\") and last (\",i1i2[-1],\") positions\")
# print(\"(Sanity Check) Column indices of second and (\",i1i2[1],\") and second to last (\",i1i2[-2],\") positions\")


# number of variables
mx_sum = mx.sum()
print("Total number of variables",mx_sum)

# number of bias term
n_linear = mx_sum - n_var

from joblib import Parallel, delayed                                                                     
import expectation_reflection as ER                                                                      
s_centered = s - s.mean(axis=0)                                                                                           
# Define wight matrix with variable for each possible amino acid at each sequence position               
w_ER = np.zeros((mx.sum(),mx.sum()))                                                                     
h0 = np.zeros(mx.sum())             

# Expectation Reflection                                                                                 
#=========================================================================================#
def predict_w(s,i0,i1i2,niter_max,l2):                                                                   
    #print('i0:',i0)                                                                                     
    i1,i2 = i1i2[i0,0],i1i2[i0,1]                                                                        
    x = np.hstack([s[:,:i1],s[:,i2:]])                                                                   
    y = s[:,i1:i2]                                                                                       
    h01,w1 = ER.fit(x,y,niter_max,l2)                                                                    
    return h01,w1                                                                                        


w_file = "%s/w.npy" % (data_dir)        
# if os.path.exists(w_file) and not create_new:                                                          
if 0:                                                                                                    
    w_ER = np.load(w_file)                                                                               
else:                                                                                                    
    #-------------------------------                                                                     
    # parallel                                                                                           
    start_time = timeit.default_timer()                                                                  
    res = Parallel(n_jobs = 20-2)(delayed(predict_w)                                                   
            (s_centered,i0,i1i2,niter_max=10,l2=100.0)                                                          
            for i0 in range(n_var))                                                                      
                                                                                                         
    run_time = timeit.default_timer() - start_time                                                       
    print('run time:',run_time)                                                                          
    #------------------------------- 
    for i0 in range(n_var):
        i1,i2 = i1i2[i0,0],i1i2[i0,1]                                                                    
        
        h01 = res[i0][0]                                                                                 
        w1 = res[i0][1]
        
        h0[i1:i2] = h01                                                                                  
        w_ER[:i1,i1:i2] = w1[:i1,:]                                                                      
        w_ER[i2:,i1:i2] = w1[i1:,:]                                                                      
        
    # make w symmetric                                                                                   
    w_ER = (w_ER + w_ER.T)/2.                                                                            
    
    np.save(w_file, w_ER)


from scipy.spatial import distance

def E(i1i2, s, w):
    E = 0
    s_len = len(i1i2)
    for i in range(s_len):
        i1,i2 = i1i2[i,0],i1i2[i,1]
        si_vec = s[i1:i2]
        for j in range(s_len):
            j1,j2 = i1i2[j,0],i1i2[j,1]
            sj_vec = s[j1:j2]
            E += np.dot(si_vec, np.dot(w[i1:i2,j1:j2], np.transpose(sj_vec)))
    return E

print(E(i1i2, s[1,:], w_ER))


def energy_diff(i1i2, s1, s2, w):
    e_diff = 0.
    s_len = len(i1i2)

    E1 = E(i1i2, s1, w)
    E2 = E(i1i2, s2, w)
    #print(E1)
    #print(E2)
    
    e_diff1 = 0
    for i in range(s_len):
        i1,i2 = i1i2[i,0],i1i2[i,1]
        si_vec = s1[i1:i2]
        for j in range(s_len):
            j1,j2 = i1i2[j,0],i1i2[j,1]
            sj_vec = s2[j1:j2]
            e_diff1 += np.dot(si_vec, np.dot(w[i1:i2,j1:j2], np.transpose(sj_vec)))
    
    e_diff2 = 0
    for i in range(s_len):
        i1,i2 = i1i2[i,0],i1i2[i,1]
        si_vec = s2[i1:i2]
        for j in range(s_len):
            j1,j2 = i1i2[j,0],i1i2[j,1]
            sj_vec = s1[j1:j2]
            e_diff2 += np.dot(si_vec, np.dot(w[i1:i2,j1:j2], np.transpose(sj_vec)))
    return E1 + E2 - e_diff1 - e_diff2   


print('identity energy difference: ', energy_diff(i1i2,s[1,:],s[1,:],w_ER))
print('seq 1 vs 2 energy difference: ', energy_diff(i1i2,s[1,:],s[2,:],w_ER))
print('seq 1 vs seq2 hamming distance: ', distance.squareform(distance.pdist(s[1:3,:], 'hamming')))
print(distance.squareform(distance.pdist([s[1,:],s[2,:]], 'hamming'))[0][1])

import itertools
s_E_dist = np.zeros((len(s),len(s)))


def energy_diff_row(s_E_dist, i1i2, s, w, i):
    s_E_row = np.zeros(len(s_E_dist))
    for j in range(i):
        s_E_row[j] = energy_diff(i1i2,s[i,:],s[j,:],w)   
    return s_E_row


res = Parallel(n_jobs = 58)(delayed(energy_diff_row)\
        (s_E_dist, i1i2, s, w_ER, i0)\
        for i0 in range(len(s)))

print(len(res), len(res[0]))
for i in range(len(s_E_dist)): 
    for j in range(i):
        s_E_dist[i,j] = res[i][j]
s_E_dist += np.array(np.transpose(s_E_dist))
#print(res_ham)
#print(s_E_dist)
np.save('furano_E_dist.npy', s_E_dist)
