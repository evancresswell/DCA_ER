# Data Processing Added Steps:
## 2018.12.24: replace 'Z', 'X', and gap by elements in the same columns and with probability
## 2018.12.26: separate remove gaps (first) and remove conserved positions (last)
import numpy as np
import pandas as pd
import pickle
# from scipy.stats import itemfreq #removed due to warning
import os, sys

"""
seq_file = 'fasta_final.txt'

#amino_acids = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S',\
#    'T','V','W','Y']

# B: Asparagine (N) or Aspartic (D)
# Z: glutamine (Q) or glutamic (E)
# X: unknown (can be any animo acid)

def read_seq_file(seq_file):
    seq = []
    with open(seq_file,'r') as seq_file:
        for i,line in enumerate(seq_file):
            seq.append(line.rstrip('\n'))
    
    l,n = len(seq),len(seq[0])
    
    seq = np.array([seq[t][i] for t in range(l) for i in range(n)]).reshape((l,n))
    #np.savetxt('seq.txt',seq,fmt='%s',delimiter='')
    
    return seq
"""


# =========================================================================================#
def remove_bad_seqs(s, tpdb, fgs=0.3):
    # remove bad sequences having a gap fraction of fgs  
    l, n = s.shape

    frequency = [(s[t, :] == '-').sum() / float(n) for t in range(l)]
    bad_seq = [t for t in range(l) if frequency[t] > fgs]
    new_s = np.delete(s, bad_seq, axis=0)
    # Find new sequence index of Reference sequence tpdb
    seq_index = np.arange(s.shape[0])
    seq_index = np.delete(seq_index, bad_seq)
    new_tpdb = np.where(seq_index == tpdb)
    print("After removing bad sequences, tpdb is now ", new_tpdb[0][0])

    return new_s, new_tpdb[0][0]


# ------------------------------#
def remove_bad_cols(s, fg=0.3, fc=0.9):
    # remove positions having a fraction fc of converved residues or a fraction fg of gaps 

    l, n = s.shape
    # gap positions:
    frequency = [(s[:, i] == '-').sum() / float(l) for i in range(n)]
    cols_gap = [i for i in range(n) if frequency[i] > fg]

    # conserved positions:
    frequency = [max(np.unique(s[:, i], return_counts=True)[1]) for i in range(n)]
    cols_conserved = [i for i in range(n) if frequency[i] / float(l) > fc]

    cols_remove = cols_gap + cols_conserved

    return np.delete(s, cols_remove, axis=1), cols_remove


# ------------------------------#
def find_bad_cols(s, fg=0.2):
    # remove positions having a fraction fg of gaps
    l, n = s.shape
    # gap positions:
    frequency = [(s[:, i] == '-').sum() / float(l) for i in range(n)]
    bad_cols = [i for i in range(n) if frequency[i] > fg]

    # return np.delete(s,gap_cols,axis=1),np.array(gap_cols)
    return np.array(bad_cols)


# ------------------------------#
def find_conserved_cols(s, fc=0.8):
    # remove positions having a fraction fc of converved residues
    l, n = s.shape

    # conserved positions:
    frequency = [max(np.unique(s[:, i], return_counts=True)[1]) for i in range(n)]
    conserved_cols = [i for i in range(n) if frequency[i] / float(l) > fc]

    # return np.delete(s,conserved_cols,axis=1),np.array(conserved_cols)
    return np.array(conserved_cols)


# ------------------------------#
def number_residues(s):
    # number of residues at each position
    l, n = s.shape
    mi = np.zeros(n)
    for i in range(n):
        s_unique = np.unique(s[:, i])
        mi[i] = len(s_unique)

    return mi


# ------------------------------
def covert_letter2number(s):
    letter2number = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, \
                     'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '-': 20,
                     'U': 21}
    # ,'B':20, 'Z':21, 'X':22}

    l, n = s.shape
    return np.array([letter2number[s[t, i]] for t in range(l) for i in range(n)]).reshape(l, n)


# ------------------------------
def convert_number2letter(s):
    number2letter = {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'K', 9: 'L', \
                     10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S', 16: 'T', 17: 'V', 18: 'W', 19: 'Y', 20: '-',
                     21: 'U'}
    print('converting s with shape : ', s.shape)
    try:
        l, n = s.shape
        return np.array([number2letter[s[t, i]] for t in range(l) for i in range(n)]).reshape(l, n)
    except(ValueError):
        return np.array([number2letter[r] for r in s])


# =========================================================================================
# 2018.12.24: replace value at a column with probility of elements in that column
def value_with_prob(a, p1):
    """ generate a value (in a) with probability
    input: a = np.array(['A','B','C','D']) and p = np.array([0.4,0.5,0.05,0.05]) 
    output: B or A (likely), C or D (unlikely)
    """
    p = p1.copy()
    # if no-specific prob --> set as uniform distribution
    if p.sum() == 0:
        p[:] = 1. / a.shape[0]  # uniform
    else:
        p[:] /= p.sum()  # normalize

    ia = int((p.cumsum() < np.random.rand()).sum())  # cordinate

    return a[ia]


# ------------------------------
def find_and_replace(s, z, a):
    """ find positions of s having z and replace by a with a probality of elements in s column
    input: s = np.array([['A','Q','A'],['A','E','C'],['Z','Q','A'],['A','Z','-']])
           z = 'Z' , a = np.array(['Q','E'])    
    output: s = np.array([['A','Q','A'],['A','E','C'],['E','Q','A'],['A','Q','-']]           
    """
    xy = np.argwhere(s == z)

    for it in range(xy.shape[0]):
        t, i = xy[it, 0], xy[it, 1]

        na = a.shape[0]
        p = np.zeros(na)
        for ii in range(na):
            p[ii] = (s[:, i] == a[ii]).sum()

        s[t, i] = value_with_prob(a, p)
    return s


# =========================================================================================
def replace_lower_by_higher_prob(s, p0=0.3):
    # input: s: 1D numpy array ; threshold p0
    # output: s in which element having p < p0 were placed by elements with p > p0, according to prob

    # f = itemfreq(s)  replaced by next line due to warning
    f = np.unique(s, return_counts=True)
    # element and number of occurence
    a, p = f[0], f[1].astype(float)

    # probabilities    
    p /= float(p.sum())

    # find elements having p > p0:
    iapmax = np.argwhere(p > p0).reshape((-1,))  # position

    apmax = a[iapmax].reshape((-1,))  # name of aminoacid
    pmax = p[iapmax].reshape((-1,))  # probability

    # find elements having p < p0
    apmin = a[np.argwhere(p < p0)].reshape((-1,))

    if apmin.shape[0] > 0:
        for a in apmin:
            ia = np.argwhere(s == a).reshape((-1,))
            for iia in ia:
                s[iia] = value_with_prob(apmax, pmax)

    return s


# --------------------------------------

# --------------------------------------
def write_FASTA(msa, pfam_id, s_ipdb, number_form=True, processed=True, path='./'):
    # Processed MSA to file in FASTA format
    msa_outfile = 'MSA_' + pfam_id + '.fa'

    # Reference sequence to file in FASTA format
    ref_outfile = 'ref_' + pfam_id + '.fa'
    ref_seq = s_ipdb

    # print("Reference Sequence (shape=",msa[ref_seq].shape,"):\n",msa[ref_seq])

    if number_form:
        msa_letters = convert_number2letter(msa)
        ref_array = msa_letters[ref_seq]
        if not processed:
            gap_ref = ref_array == '-'  # remove gaps from reference array
            ref_letters = msa_letters[ref_seq][~gap_ref]
        else:
            ref_letters = msa_letters[ref_seq]
    else:
        msa_letters = msa
        ref_array = msa[ref_seq]
        gap_ref = ref_array == '-'  # remove gaps from reference array
        ref_letters = ref_array[~gap_ref]

    printing = False
    if printing:
        print("Reference Sequence number: ", ref_seq)
        print("Reference Sequence (shape=", ref_letters.shape, "):\n", ref_letters)

        print("Writing processed MSA (shape=", msa_letters.shape, ") to FASTA files:\n", msa_letters)

    # First save reference sequence to FASTA file
    ref_str = ''
    ref_list = ref_letters.tolist()
    ref_str = ref_str.join(ref_list)
    if processed:
        with open(path + ref_outfile, 'w') as fh:
            fh.write('>{}\n{}\n'.format(pfam_id + ' | REFERENCE', ref_str))
    else:
        ref_outfile = 'orig_ref_' + pfam_id + '.fa'
        with open(path + ref_outfile, 'w') as fh:
            fh.write('>{}\n{}\n'.format(pfam_id + ' | REFERENCE', ref_str))

    # Next save MSA to FAST file

    with open(path + msa_outfile, 'w') as fh:
        for seq_num, seq in enumerate(msa_letters):
            msa_list = seq.tolist()
            msa_str = ''
            msa_str = msa_str.join(msa_list)
            if seq_num == ref_seq:
                fasta_header = pfam_id + ' | REFERENCE'
            else:
                fasta_header = pfam_id
            if printing:
                print(msa_str)
            fh.write('>{}\n{}\n'.format(fasta_header, msa_str))



# --------------------------------------

# --------------------------------------
def min_res(s):
    n = s.shape[1]
    minfreq = np.zeros(n)
    for i in range(n):
        # f = itemfreq(s[:,i])
        f = ""  # removing previous line due to warning
        minfreq[i] = np.min(f[:, 1])

    return minfreq


# =========================================================================================
def create_unprocessed_FASTA(pdb, data_path, pfam_id, ipdb=0):
    printing = True

    # read parse_pfam data:
    s = np.load('%s/%s/msa.npy' % (data_path, pfam_id)).T
    if printing:
        print("shape of s (import from msa.npy):\n", s.shape)

    # convert bytes to str
    s = np.array([s[t, i].decode('UTF-8') for t in range(s.shape[0]) \
                  for i in range(s.shape[1])]).reshape(s.shape[0], s.shape[1])
    if printing:
        print("shape of s (after UTF-8 decode):\n", s.shape)

    # pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))
    # if printing:
    #	print("pdb:\n",pdb)

    # convert bytes to str (python 2 to python 3)
    # pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0]) \
    #	for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])
    # if printing:
    #	print("pdb (after UTF-8 decode, removing 'b'):\n",pdb)

    # tpdb is the sequence #
    tpdb = int(pdb[ipdb, 1])

    # write unprocessed MSA to FASTA
    msa_outfile, ref_outfile = write_FASTA(s, pfam_id, ipdb, number_form=False)

    return msa_outfile, ref_outfile


def delete_sorted_DI_duplicates(sorted_DI):
    temp1 = []
    temp2 = []
    for (a, b), score in sorted_DI:
        if (a, b) not in temp1 and (b, a) not in temp1:  # to check for the duplicate tuples
            temp1.append(((a, b)))
            temp2.append(((a, b), score))
    temp2.sort(key=lambda x: x[1], reverse=True)
    return temp2


# =========================================================================================
# noinspection PyBroadException
def load_msa(data_path, pfam_id):
    s = np.load('%s/%s/msa.npy' % (data_path, pfam_id)).T
    # print("shape of s (import from msa.npy):\n",s.shape)

    # convert bytes to str
    try:
        s = np.array([s[t, i].decode('UTF-8') for t in range(s.shape[0]) \
                      for i in range(s.shape[1])]).reshape(s.shape[0], s.shape[1])
    # print("shape of s (after UTF-8 decode):\n",s.shape)
    except:
        print("\n\nUTF not decoded, pfam_id: %s \n\n" % pfam_id, s.shape)
        print("Exception: ", sys.exc_info()[0])
        # Create list file for missing pdb structures
        if not os.path.exists('missing_MSA.txt'):
            file_missing_msa = open("missing_MSA.txt", 'w')
            file_missing_msa.write("%s\n" % pfam_id)
            file_missing_msa.close()
        else:
            file_missing_msa = open("missing_MSA.txt", 'a')
            file_missing_msa.write("%s\n" % pfam_id)
            file_missing_msa.close()
        return
    return s


def data_processing(data_path, pfam_id, ipdb=0, gap_seqs=0.2, gap_cols=0.2, prob_low=0.004, conserved_cols=0.8, printing=True, out_dir='./', letter_format=False, remove_cols=True):
    # def data_processing(data_path,pfam_id,ipdb=0,gap_seqs=0.2,gap_cols=0.2,prob_low=0.004):
    # read parse_pfam data:
    # print('read original aligned pfam data')
    # s = np.load('../%s/msa.npy'%pfam_id).T
    s = load_msa(data_path, pfam_id)
    orig_seq_len = s.shape[1]

    # print('select only column presenting as uppercase at PDB sequence')
    # pdb = np.load('../%s/pdb_refs.npy'%pfam_id)
    pdb = np.load('%s/%s/pdb_refs.npy' % (data_path, pfam_id))
    # ipdb = 0

    # convert bytes to str (python 2 to python 3)
    pdb = np.array([pdb[t, i].decode('UTF-8') for t in range(pdb.shape[0]) \
                    for i in range(pdb.shape[1])]).reshape(pdb.shape[0], pdb.shape[1])
    if printing:
        print("pdb ref example (pdb[0])  (after UTF-8 decode, removing 'b'):\n", pdb[0])

    tpdb = int(pdb[ipdb, 1])
    if printing:
        print('tpdb (s_ipdb) is : ', tpdb)
    # tpdb is the sequence #
    # print(tpdb)

    if printing:
        print("#\n\n-------------------------Remove Gaps--------------------------#")
        print('Shape of s is : ', s.shape)
        print("s = \n", s)


    gap_pdb = s[tpdb] == '-'  # returns True/False for gaps/no gaps
    # print("removing gaps...")
    s = s[:, ~gap_pdb]  # removes gaps
    # print('shape of s without reference sequence gaps: ', s.shape)
    s_index = np.arange(s.shape[1])

    if printing:
        print("s[tpdb] shape is ", s[tpdb].shape)
        print("s = \n", s)
        print("though s still has gaps, s[%d] does not:\n" % (tpdb), s[tpdb])
        print("s shape is ", s.shape)
        print("Saving indices of reference sequence s[%d](length=%d):\n" % (tpdb, s_index.shape[0]), s_index)
        print("#--------------------------------------------------------------#\n\n")

    lower_cols = np.array([i for i in range(s.shape[1]) if s[tpdb, i].islower()])
    if printing:
        print("removing non aligned (lower case) columns in subject sequence:\n ", lower_cols, '\n')
    # lower case removal reference: https://onlinelibrary.wiley.com/doi/full/10.1002/1097-0134%2820001101%2941%3A2%3C224%3A%3AAID-PROT70%3E3.0.CO%3B2-Z

    # upper = np.array([x.isupper() for x in s[tpdb]])
    # print('select only column presenting as uppercase at the first row')
    # upper = np.array([x.isupper() for x in s[0]])
    # s = s[:,upper]

    if printing:
        print(s.shape)
        print("In Data Processing Reference Sequence (shape=", s[tpdb].shape, "): \n", s[tpdb])
    # print('remove sequences containing too many gaps')
    s, tpdb = remove_bad_seqs(s, tpdb, gap_seqs)  # removes all sequences (rows) with >gap_seqs gap %
    if printing:
        print('\nAfter removing bad sequences...\ntpdb (s_ipdb) is : ', tpdb)
        print(s.shape)

    bad_cols = find_bad_cols(s, gap_cols)
    if printing:
        print('found bad columns :=', bad_cols)

    # 2018.12.24:
    # replace 'Z' by 'Q' or 'E' with prob
    # print('replace Z by Q or E')
    s = find_and_replace(s, 'Z', np.array(['Q', 'E']))

    # replace 'B' by Asparagine (N) or Aspartic (D)
    # print('replace B by N or D')
    s = find_and_replace(s, 'B', np.array(['N', 'D']))

    # replace 'X' as amino acids with prob
    # print('replace X by other aminoacids')
    amino_acids = np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', \
                            'T', 'V', 'W', 'Y', 'U'])
    s = find_and_replace(s, 'X', amino_acids)

    # remove conserved cols
    conserved_cols = find_conserved_cols(s, conserved_cols)
    if printing:
        print("found conserved columns (80% repetition):\n", conserved_cols)

    # print(s.shape)
    # print('number of conserved columns removed:',conserved_cols.shape[0])

    removed_cols = np.array(list(set(bad_cols) | set(conserved_cols)))

    removed_cols = np.array(list(set(removed_cols) | set(lower_cols)))
    if printing:
        print("We remove conserved and bad columns with, at the following indices:\n", removed_cols)


    # info still pased through removed_cols but this way we can interact with full msa
    if remove_cols:
        s = np.delete(s, removed_cols, axis=1)
        s_index = np.delete(s_index, removed_cols)

    if printing:
        print("Removed Columns...")
        print("s now has shape: ", s.shape)
        print("s_index (length=%d) = \n" % s_index.shape[0], s_index)

    # print('replace gap(-) by other aminoacids')
    # s = find_and_replace(s,'-',amino_acids)

    if printing:
        print("In Data Processing Reference Sequence (shape=", s[tpdb].shape, "): \n", s[tpdb])

    # Convert S to number format (number representation of amino acids)
    if not letter_format:
        # convert letter to number:
        s = covert_letter2number(s)

    # print(s.shape)
    # replace lower probs by higher probs 
    # print('replace lower probs by higher probs')
    for i in range(s.shape[1]):
        s[:, i] = replace_lower_by_higher_prob(s[:, i], prob_low)

    # print("s[tpdb] (shape=",s[tpdb].shape,"):\n",s[tpdb])
    # min_res = min_res(s)
    # print(min_res)

    # remove_cols = np.hstack([gap_cols,conserved_cols])
    # remove_cols = np.hstack([remove_cols,lower_cols]) ## 2019.01.22

    # np.savetxt('s0.txt',s,fmt='%i')
    # np.savetxt('cols_remove.txt',remove_cols,fmt='%i')

    # f = open('n_pos.txt','w')
    # f.write('%i'%(s.shape[1]))
    # f.close()

    # mi = number_residues(s)
    # print(mi.mean())
    np.save("%s/%s_removed_cols.npy" % (out_dir, pfam_id), removed_cols)

    return s, removed_cols, s_index, tpdb, orig_seq_len


# =========================================================================================

def data_processing_covid(data_path, pfam_id, ipdb=0, gap_seqs=0.2, gap_cols=0.2, prob_low=0.004, conserved_cols=0.8):
    # def data_processing(data_path,pfam_id,ipdb=0,gap_seqs=0.2,gap_cols=0.2,prob_low=0.004):

    printing = True
    printing = False

    # read parse_pfam data:
    # print('read original aligned pfam data')
    # s = np.load('../%s/msa.npy'%pfam_id).T
    s = np.load('%s/%s/msa.npy' % (data_path, pfam_id))
    # print(type(s))
    # print(type(s[0]))
    # print(s[0])
    if printing:
        print("shape of s (import from msa.npy):\n", s.shape)

    # convert bytes to str
    """
    try:
        s = np.array([s[t,i].decode('UTF-8') for t in range(s.shape[0]) \
             for i in range(s.shape[1])]).reshape(s.shape[0],s.shape[1])
        if printing:
    	    print("shape of s (after UTF-8 decode):\n",s.shape)
    except:
        print("\n\nUTF not decoded, pfam_id: %s \n\n"%pfam_id,s.shape)
        print("Exception: ",sys.exc_info()[0])
        # Create list file for missing pdb structures
        if not os.path.exists('missing_MSA.txt'):
            file_missing_msa = open("missing_MSA.txt",'w')
            file_missing_msa.write("%s\n"% pfam_id)
            file_missing_msa.close()
        else:
            file_missing_msa = open("missing_MSA.txt",'a')
            file_missing_msa.write("%s\n"% pfam_id)
            file_missing_msa.close()
        return
    """
    if printing:
        print('\n\nstarting shape: ', s.shape)

    if printing:
        print("#\n\n-------------------------Remove Gaps--------------------------#")
        print("s = \n", s)

    # no pdb_ref structure for covid proteins, ref strucutre is always s[0]
    tpdb = ipdb
    gap_pdb = s[tpdb] == '-'  # returns True/False for gaps/no gaps

    # print("removing gaps...")
    s = s[:, ~gap_pdb]  # removes gaps
    if printing:
        print(s.shape)
    s_index = np.arange(s.shape[1])
    print(s_index)

    if printing:
        print("s[tpdb] shape is ", s[tpdb].shape)
        print("s = \n", s)
        print("though s still has gaps, s[%d] does not:\n" % (tpdb), s[tpdb])
        print("s shape is ", s.shape)
        print("Saving indices of reference sequence s[%d](length=%d):\n" % (tpdb, s_index.shape[0]), s_index)
        print("#--------------------------------------------------------------#\n\n")

    # print(s.shape)
    # print(s)

    lower_cols = np.array([i for i in range(s.shape[1]) if s[tpdb, i].islower()])
    # print("lower_cols: ",lower_cols)

    # upper = np.array([x.isupper() for x in s[tpdb]])

    # print('select only column presenting as uppercase at the first row')
    # upper = np.array([x.isupper() for x in s[0]])
    # s = s[:,upper]
    if printing:
        print(s.shape)

    if printing:
        print("In Data Processing Reference Sequence (shape=", s[tpdb].shape, "): \n", s[tpdb])
    # print('remove sequences containing too many gaps')
    s, tpdb = remove_bad_seqs(s, tpdb, gap_seqs)  # removes all sequences (rows) with >gap_seqs gap %
    if printing:
        print(s.shape)

    bad_cols = find_bad_cols(s, gap_cols)
    if printing:
        print('found bad columns :=', bad_cols)

    # 2018.12.24:
    # replace 'Z' by 'Q' or 'E' with prob
    # print('replace Z by Q or E')
    s = find_and_replace(s, 'Z', np.array(['Q', 'E']))

    # replace 'B' by Asparagine (N) or Aspartic (D)
    # print('replace B by N or D')
    s = find_and_replace(s, 'B', np.array(['N', 'D']))

    # replace 'X' as amino acids with prob
    # print('replace X by other aminoacids')
    amino_acids = np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', \
                            'T', 'V', 'W', 'Y', 'U'])
    s = find_and_replace(s, 'X', amino_acids)

    # remove conserved cols
    conserved_cols = find_conserved_cols(s, conserved_cols)
    if printing:
        print("found conserved columns (80% repetition):\n", conserved_cols)

    # print(s.shape)
    # print('number of conserved columns removed:',conserved_cols.shape[0])

    removed_cols = np.array(list(set(bad_cols) | set(conserved_cols)))

    removed_cols = np.array(list(set(removed_cols) | set(lower_cols)))
    if printing:
        print("We remove conserved and bad columns with, at the following indices:\n", removed_cols)

    # 2019.09.17: excluse conserved cols
    # removed_cols = np.array(list(set(bad_cols) | set(lower_cols)))

    s = np.delete(s, removed_cols, axis=1)
    s_index = np.delete(s_index, removed_cols)

    print(s_index)
    if printing:
        print("Removed Columns...")
        print("s now has shape: ", s.shape)
        print("s_index (length=%d) = \n" % s_index.shape[0], s_index)

    # print('replace gap(-) by other aminoacids')
    # s = find_and_replace(s,'-',amino_acids)

    if printing:
        print("In Data Processing Reference Sequence (shape=", s[tpdb].shape, "): \n", s[tpdb])

    # convert letter to number:
    s = covert_letter2number(s)
    # print(s.shape)
    # replace lower probs by higher probs 
    # print('replace lower probs by higher probs')
    for i in range(s.shape[1]):
        s[:, i] = replace_lower_by_higher_prob(s[:, i], prob_low)

    # print("s[tpdb] (shape=",s[tpdb].shape,"):\n",s[tpdb])
    # min_res = min_res(s)
    # print(min_res)

    # remove_cols = np.hstack([gap_cols,conserved_cols])
    # remove_cols = np.hstack([remove_cols,lower_cols]) ## 2019.01.22

    # np.savetxt('s0.txt',s,fmt='%i')
    # np.savetxt('cols_remove.txt',remove_cols,fmt='%i')

    # f = open('n_pos.txt','w')
    # f.write('%i'%(s.shape[1]))
    # f.close()

    # mi = number_residues(s)
    # print(mi.mean())
    np.save("%s/removed_cols.npy" % pfam_id, removed_cols)
    return s, removed_cols, s_index, tpdb


# =========================================================================================

def generate_pfam_data(data_path, pfam_id, ipdb):
    pdb = np.load('%s/%s/pdb_refs.npy' % (data_path, pfam_id))

    # Pre-Process Structure Data
    # delete 'b' in front of letters (python 2 --> python 3)
    print(pdb.shape)
    print(pdb)
    if len(pdb) == 0:
        print("Missing PDB structure")
        file_missing_pdb = open("missing_PDB.txt", 'a')
        file_missing_pdb.write("%s\n" % pfam_id)
        file_missing_pdb.close()
    else:
        pdb = np.array([pdb[t, i].decode('UTF-8') for t in range(pdb.shape[0]) \
                        for i in range(pdb.shape[1])]).reshape(pdb.shape[0], pdb.shape[1])

        # Create pandas dataframe for protein structure
        df = pd.DataFrame(pdb, columns=['PF', 'seq', 'id', 'uniprot_start', 'uniprot_start', \
                                        'pdb_id', 'chain', 'pdb_start', 'pdb_end'])
        print(df.head())

        # print('seq:',int(pdb[ipdb,1]))

        # try:
        # data processing
        s0, cols_removed, s_index, s_ipdb = data_processing(data_path, pfam_id, ipdb, \
                                                            gap_seqs=0.2, gap_cols=0.2, prob_low=0.004,
                                                            conserved_cols=0.9)

        # Save processed data
        msa_outfile, ref_outfile = write_FASTA(s0, pfam_id, s_ipdb, path='pfam_ecc/')
        pf_dict = {}
        pf_dict['s0'] = s0
        pf_dict['s_index'] = s_index
        pf_dict['s_ipdb'] = s_ipdb
        pf_dict['cols_removed'] = cols_removed

        with open('pfam_ecc/%s_DP.pickle' % (pfam_id), 'wb') as f:
            pickle.dump(pf_dict, f)
        f.close()
        """
        except:
            print("Could not generate data for %s: "%(pfam_id),sys.exc_info())

            if not os.path.exists('data_not_generated.txt'):
                data_fail = open("data_not_generated.txt",'w')
                data_fail.write("%s\n"% pfam_id)
                data_fail.close()
            else:
                data_fail = open("data_not_generated.txt",'a')
                data_fail.write("%s\n"% pfam_id)
                data_fail.close()
        return
        """
    return pf_dict, pdb
# -------------------------------
