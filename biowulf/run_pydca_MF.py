import pydca
from pydca.meanfield_dca import meanfield_dca
from pydca import sequence_backmapper
from pydca.msa_trimmer import msa_trimmer
from pydca.contact_visualizer import contact_visualizer
from pydca import dca_utilities
import os, sys
import pandas as pd
import numpy as np

print(os.getcwd())

method = "DI"
preprocess = False # do we want to preprocess FASTA files the same way as in DCA-ER method?

pfam_id = sys.argv[1]
n_cpu = sys.argv[2]
print('Finding pydca MF contacts for %s', pfam_id)


DCA_ER_dir = '/data/cresswellclayec/DCA_ER/'
msa_npy_file = '/data/cresswellclayec/DCA_ER/Pfam-A.full/%s/msa.npy' % pfam_id
msa_fa_file  = '/data/cresswellclayec/DCA_ER/Pfam-A.full/%s/msa.fa' % pfam_id
pdb_ref_file = '/data/cresswellclayec/DCA_ER/Pfam-A.full/%s/pdb_refs.npy' % pfam_id
out_dir = '%sprotein_data/di/' % DCA_ER_dir
processed_data_dir = "%s/protein_data/data_processing_output" % DCA_ER_dir



# os.chdir('/home/evan/PycharmProjects/DCA_ER')
# os.chdir('/home/eclay/DCA_ER') # Hurrican Location
os.chdir(DCA_ER_dir)
ipdb = 0
from ecc_tools import npy2fa
if not preprocess:
    msa_outfile, ref_outfile, s = npy2fa(pfam_id, msa_npy_file, pdb_ref_file=pdb_ref_file, ipdb=ipdb, preprocess=preprocess)
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


print('Preprocessed: %r, s dimensions: ' % preprocess, s.shape)


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #


# create MSATrimmer instance 
trimmer = msa_trimmer.MSATrimmer(
    str(msa_outfile), biomolecule='protein', 
    refseq_file=str(ref_outfile),
)

trimmed_data = trimmer.get_msa_trimmed_by_refseq(remove_all_gaps=True)

#write trimmed msa to file in FASTA format
trimmed_data_outfile = '%s/MSA_%s_Trimmed.fa' % (processed_data_dir, pfam_id)
with open(trimmed_data_outfile, 'w') as fh:
    for seqid, seq in trimmed_data:
        fh.write('>{}\n{}\n'.format(seqid, seq))
        

# Taken from begining of MSA trimmer (trim_by_refseq)
from pydca.sequence_backmapper.sequence_backmapper import SequenceBackmapper
seqbackmapper = SequenceBackmapper(msa_file = str(msa_outfile), refseq_file = str(ref_outfile), biomolecule = 'protein')

matching_seqs = seqbackmapper.find_matching_seqs_from_alignment()
pydca_refseq = matching_seqs[0]

for i, seq in enumerate(s):
    seq_str = ""
    for a in seq:
        seq_str += a
    if seq_str == pydca_refseq:
        sbm_tpdb = i
        npy_refseq = seq
        break

print('SequenceBackmapper found reference, first ocurring sequence has index ', sbm_tpdb)
print('s[%d] = ' % sbm_tpdb, s[sbm_tpdb])


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

#create mean-field DCA instance 
mfdca_inst = meanfield_dca.MeanFieldDCA(
    trimmed_data_outfile,
    'protein',
    pseudocount = 0.5,
    seqid = 0.8,

)

if preprocess:
    mf_out_file = '%s%s_%s_pydca_mf_preproc_di.npy' % (out_dir, pfam_id, method)
else:
    mf_out_file = '%s%s_%s_pydca_mf_di.npy' % (out_dir, pfam_id, method)

if 0: #os.path.exists(mf_out_file):
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


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

for site_pair, score in mfdca_scores[:5]:
    print(site_pair, score)

   
# Save the scores for future use/comparison
np.save(mf_out_file, mfdca_scores)

# save processed data
np.save('%s/%s_pydca_MF_s0.npy' 		% (processed_data_dir, pfam_id), s0)
np.save('%s/%s_pydca_MF_ref_seq_mapping.npy' 	% (processed_data_dir, pfam_id), mfdca_inst.__refseq_mapping_dict)
np.save('%s/%s_pydca_MF_refseq.npy' 		% (processed_data_dir, pfam_id), npy_refseq)
np.save('%s/%s_pydca_MF_tpdb.npy' 		% (processed_data_dir, pfam_id), sbm_tpdb)

print('Complete...')
