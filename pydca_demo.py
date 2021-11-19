""" 
pydca  or selected modules
import pydca 
from pydca.plmdca import plmdca uthor: Mehari B. Zerihun
"""

# import pydca modules
from pydca.plmdca import plmdca
# from pydca.meanfield_dca import meanfield_dca
# from pydca.sequence_backmapper import sequence_backmapper
from pydca.msa_trimmer import msa_trimmer

from pydca.contact_visualizer import contact_visualizer
# from pydca.dca_utilities import dca_utilities
import os
import numpy as np
import pandas as pd
from ecc_tools import npy2fa

print(os.getcwd())

# Define Protein Family (pfam_id) and pdb-structure index (ipdb)
pfam_id = 'PF00186'
ipdb = 0

msa_npy_file = '/home/eclay/Pfam-A.full/%s/msa.npy' % pfam_id # Hurricane Location
msa_fa_file  = '/home/eclay/Pfam-A.full/%s/msa.fa' % pfam_id # Hurricane Location
pdb_ref_file = '/home/eclay/Pfam-A.full/%s/pdb_refs.npy' % pfam_id # Hurricane Location

msa_npy_file = '/home/evan/PycharmProjects/DCA_ER/Pfam-A.full/%s/msa.npy' % pfam_id # Home Location
msa_fa_file  = '/home/evan/PycharmProjects/DCA_ER/Pfam-A.full/%s/msa.fa' % pfam_id # Home Location
pdb_ref_file = '/home/evan/PycharmProjects/DCA_ER/Pfam-A.full/%s/pdb_refs.npy' % pfam_id # Home Location

os.chdir('/home/evan/PycharmProjects/DCA_ER')
# os.chdir('/home/eclay/DCA_ER') # Hurrican Location
# Set DCA_ER directory
DCA_dir = os.getcwd()

msa_outfile, ref_outfile = npy2fa(pfam_id, msa_npy_file, pdb_ref_file=pdb_ref_file, ipdb=ipdb)

# Define pdb-references for Protein Family
pdb = np.load(pdb_ref_file)
processed_data_dir = "%s/protein_data/data_processing_output" % DCA_dir

# delete 'b' in front of letters (python 2 --> python 3)
pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0]) \
         for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])

# Print number of pdb structures in Protein ID folder
npdb = pdb.shape[0]
print('number of pdb structures:',npdb)

# Create pandas dataframe for protein structure
pdb_df = pd.DataFrame(pdb,columns = ['PF','seq','id','uniprot_start','uniprot_start',\
                                 'pdb_id','chain','pdb_start','pdb_end'])
pdb_df.head()
pdb_id = pdb_df.iloc[ipdb]['pdb_id']
pdb_chain = pdb_df.iloc[ipdb]['chain']

# create MSATrimmer instance 
trimmer = msa_trimmer.MSATrimmer(
    str(msa_outfile), biomolecule='protein', 
    refseq_file=str(ref_outfile),
)

trimmed_data = trimmer.get_msa_trimmed_by_refseq(remove_all_gaps=True)

#write trimmed msa to file in FASTA format
trimmed_data_outfile = 'MSA_%s_Trimmed.fa' % pfam_id
with open(trimmed_data_outfile, 'w') as fh:
    for seqid, seq in trimmed_data:
        fh.write('>{}\n{}\n'.format(seqid, seq))


# Compute DCA scores using Pseudolikelihood maximization algorithm

plmdca_inst = plmdca.PlmDCA(
    trimmed_data_outfile,
    'protein',
    seqid = 0.8,
    lambda_h = 1.0,
    lambda_J = 20.0,
    num_threads = 10,
    max_iterations = 500,
)

# compute DCA scores summarized by Frobenius norm and average product corrected
plmdca_FN_APC = plmdca_inst.compute_sorted_FN_APC()
plm_out_file = '/home/evan/PycharmProjects/DCA_ER/%s_di_plm.npy' % pfam_id
np.save(plm_out_file, plmdca_FN_APC)

for site_pair, score in plmdca_FN_APC[:5]:
    print(site_pair, score)

plmdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
    refseq_file = str(ref_outfile),
    sorted_dca_scores = plmdca_FN_APC,
    linear_dist = 4,
    contact_dist = 8.0,
)

contact_map_data = plmdca_visualizer.plot_contact_map()

tp_rate_data = plmdca_visualizer.plot_true_positive_rates()
