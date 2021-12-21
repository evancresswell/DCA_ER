"""
author: Mehari B. Zerihun
"""

# -------------------------------------------------------------------------------------------------------------------------------------------------- #
print('importing11')
# import pydca modules
import pydca
print(pydca.plmdca)
print('importing1pydca1')
from pydca.plmdca import plmdca
print('importing1')
from pydca.meanfield_dca import meanfield_dca
print('importing2')
from pydca import sequence_backmapper
print('importing3')
from pydca.msa_trimmer import msa_trimmer
print('importing4')
from pydca.contact_visualizer import contact_visualizer
print('importing5')
from pydca import dca_utilities
print('importing6')
import os
print('importing7')
import pandas as pd
print('importing8')
import numpy as np
print('importing9')

print(os.getcwd())

pfam_id = 'PF00186'
method = "DI"
preprocess = True # do we want to preprocess FASTA files the same way as in DCA-ER method?


if 0:
    DCA_ER_dir = '/home/eclay/DCA_ER/'
    msa_npy_file = '/home/eclay/Pfam-A.full/%s/msa.npy' % pfam_id # Hurricane Location
    msa_fa_file  = '/home/eclay/Pfam-A.full/%s/msa.fa' % pfam_id # Hurricane Location
    pdb_ref_file = '/home/eclay/Pfam-A.full/%s/pdb_refs.npy' % pfam_id # Hurricane Location
if 1:
    DCA_ER_dir = '/home/evan/PycharmProjects/DCA_ER/'
    msa_npy_file = '/home/evan/PycharmProjects/DCA_ER/Pfam-A.full/%s/msa.npy' % pfam_id
    msa_fa_file  = '/home/evan/PycharmProjects/DCA_ER/Pfam-A.full/%s/msa.fa' % pfam_id
    pdb_ref_file = '/home/evan/PycharmProjects/DCA_ER/Pfam-A.full/%s/pdb_refs.npy' % pfam_id
    out_dir = '%sprotein_data/di/' % DCA_ER_dir
if 0:
    DCA_ER_dir = '/home/ecresswell/DCA_ER/'
    msa_npy_file = '/home/ecresswell/DCA_ER/Pfam-A.full/%s/msa.npy' % pfam_id
    msa_fa_file  = '/home/ecresswell/DCA_ER/Pfam-A.full/%s/msa.fa' % pfam_id
    pdb_ref_file = '/home/ecresswell/DCA_ER/Pfam-A.full/%s/pdb_refs.npy' % pfam_id
    out_dir = '%sprotein_data/di/' % DCA_ER_dir

if 1:
    DCA_ER_dir = '/data/cresswellclayec/DCA_ER/'
    msa_npy_file = '/data/cresswellclayec/DCA_ER/Pfam-A.full/%s/msa.npy' % pfam_id
    msa_fa_file  = '/data/cresswellclayec/DCA_ER/Pfam-A.full/%s/msa.fa' % pfam_id
    pdb_ref_file = '/data/cresswellclayec/DCA_ER/Pfam-A.full/%s/pdb_refs.npy' % pfam_id
    out_dir = '%sprotein_data/di/' % DCA_ER_dir


print("here1")
# os.chdir('/home/evan/PycharmProjects/DCA_ER')
# os.chdir('/home/eclay/DCA_ER') # Hurrican Location
os.chdir(DCA_ER_dir)
ipdb = 0
from ecc_tools import npy2fa
if not preprocess:
    msa_outfile, ref_outfile = npy2fa(pfam_id, msa_npy_file, pdb_ref_file=pdb_ref_file, ipdb=ipdb, preprocess=preprocess)
else:
    msa_outfile, ref_outfile, s, cols_removed, s_index, tpdb, orig_seq_len  = npy2fa(pfam_id, msa_npy_file, pdb_ref_file=pdb_ref_file, ipdb=ipdb, preprocess=preprocess,
    gap_seqs=.2, gap_cols=.2, prob_low=.004, conserved_cols=.9)
    print('Saved pre-processed msa to %s' % msa_outfile)
    # print(s)
    # print(s.shape)
    gap_pdb = s[tpdb] == '-'  # returns True/False for gaps/no gaps in reference sequence            
    s_gap = s[:, ~gap_pdb]  # removes gaps in reference sequence                                     
    print(gap_pdb)
    ref_s = s_gap[tpdb]
    print(s[tpdb])
    print(s[tpdb].shape)
    print(ref_s)
    print(ref_s.shape)
    print(s.shape)
#     for col_i in range(s.shape[1]):
#         if col_i in cols_removed:
#             print(s[tpdb,col_i])
#             print(s[:,col_i])

print("here2")
# -------------------------------------------------------------------------------------------------------------------------------------------------- #


# -------------------------------------------------------------------------------------------------------------------------------------------------- #
# create MSATrimmer instance 
trimmer = msa_trimmer.MSATrimmer(
    str(msa_outfile), biomolecule='protein', 
    refseq_file=str(ref_outfile),
)

trimmed_data = trimmer.get_msa_trimmed_by_refseq(remove_all_gaps=True)

#write trimmed msa to file in FASTA format
trimmed_data_outfile = '%sprotein_data/data_processing_output/MSA_%s_Trimmed.fa' % (DCA_ER_dir, pfam_id)
with open(trimmed_data_outfile, 'w') as fh:
    for seqid, seq in trimmed_data:
        fh.write('>{}\n{}\n'.format(seqid, seq))

print("here3")
# -------------------------------------------------------------------------------------------------------------------------------------------------- #


# -------------------------------------------------------------------------------------------------------------------------------------------------- #
# Compute DCA scores using Pseudolikelihood maximization algorithm

plmdca_inst = plmdca.PlmDCA(
    trimmed_data_outfile,
    'protein',
    seqid = 0.8,
    lambda_h = 1.0,
    lambda_J = 20.0,
    num_threads =6,
    max_iterations = 500,
)

if preprocess:
    plm_out_file = '%s%s_%s_pydca_plm_preproc.npy' % (out_dir, pfam_id, method)
else:
    plm_out_file = '%s%s_%s_pydca_plm.npy' % (out_dir, pfam_id, method)

if os.path.exists(plm_out_file):
    plmdca_scores = np.load(plm_out_file)
else:
    # compute PLM DCA scores summarized by various methods:
    # FN_APC: Frobenius norm and average product corrected
    # DI_APC: raw DI with average product correction
    # DI: raw DI score
    # FN: Frobenius norm of raw DI
    if method == "DI_APC":
        plmdca_scores = plmdca_inst.compute_sorted_DI_APC()
    if method == "DI":
        plmdca_scores = plmdca_inst.compute_sorted_DI()
    if method == "FN_APC":
          plmdca_scores = plmdca_inst.compute_sorted_FN_APC()
    if method == "FN":
        plmdca_scores = plmdca_inst.compute_sorted_FN()
# -------------------------------------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------------------------------------- #
# for site_pair, score in plmdca_scores[:5]:
#     print(site_pair, score)
# Save the scores for future use/comparison
np.save(plm_out_file, plmdca_scores)
# -------------------------------------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------------------------------------- #
#create mean-field DCA instance 
mfdca_inst = meanfield_dca.MeanFieldDCA(
    trimmed_data_outfile,
    'protein',
    pseudocount = 0.5,
    seqid = 0.8,

)

reg_fi = mfdca_inst.get_reg_single_site_freqs()                                               
reg_fij = mfdca_inst.get_reg_pair_site_freqs()                                               
corr_mat = mfdca_inst.construct_corr_mat(reg_fi, reg_fij)                                                                             
couplings = mfdca_inst.compute_couplings(corr_mat)        
fields_ij = mfdca_inst.compute_two_site_model_fields(couplings, reg_fi)
np.save('%s%s_pydca_fields_preproc.npy' % (out_dir, pfam_id), fields_ij)
np.save('%s%s_pydca_couplings_preproc.npy' % (out_dir, pfam_id), couplings)
np.save('%s%s_pydca_corr_preproc.npy' % (out_dir, pfam_id), corr_mat)
np.save('%s%s_pydca_fij_preproc.npy' % (out_dir, pfam_id), reg_fij)
np.save('%s%s_pydca_fi_preproc.npy' % (out_dir, pfam_id), reg_fi)


if preprocess:
    mf_out_file = '%s%s_%s_pydca_mf_preproc.npy' % (out_dir, pfam_id, method)
else:
    mf_out_file = '%s%s_%s_pydca_mf.npy' % (out_dir, pfam_id, method)

if os.path.exists(mf_out_file):
    mfdca_scores = np.load(mf_out_file)
else:
    # compute MF DCA scores summarized by various methods:
    # FN_APC: Frobenius norm and average product corrected
    # DI_APC: raw DI with average product correction
    # DI: raw DI score
    # FN: Frobenius norm of raw DI
    if method == "DI_APC":
        mfdca_scores = mfdca_inst.compute_sorted_DI_APC()
    if method == "DI":
        mfdca_scores = mfdca_inst.compute_sorted_DI()
    if method == "FN_APC":
        mfdca_scores = mfdca_inst.compute_sorted_FN_APC()
    if method == "FN":
        mfdca_scores = mfdca_inst.compute_sorted_FN()
# -------------------------------------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------------------------------------- #
print(couplings.shape)
print(mfdca_scores[:-10])
index_i = []
index_j = []
for [(i, j), score] in mfdca_scores:
    index_i.append(i)
    index_j.append(j)
print(max(index_i))
print(max(index_j))
# -------------------------------------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------------------------------------- #
for site_pair, score in mfdca_scores[:5]:
    print(site_pair, score)
    
# Save the scores for future use/comparison
np.save(mf_out_file, mfdca_scores)
# -------------------------------------------------------------------------------------------------------------------------------------------------- #

