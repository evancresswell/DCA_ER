#--------------------------------------------------------------------------------------#
#------------------------ Load Modules ------------------------------------------------#
#--------------------------------------------------------------------------------------#
import sys,os
import data_processing as dp
import ecc_tools as tools
import timeit
# import pydca-ER module
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import linalg
from sklearn.preprocessing import OneHotEncoder

from pydca.erdca import erdca
from pydca.sequence_backmapper import sequence_backmapper
from pydca.msa_trimmer import msa_trimmer
from pydca.msa_trimmer.msa_trimmer import MSATrimmerException
from pydca.dca_utilities import dca_utilities
import numpy as np
import pickle
from gen_ROC_jobID_df import add_ROC

# Import Bio data processing features 
import Bio.PDB, warnings
from Bio.PDB import *
pdb_list = Bio.PDB.PDBList()
pdb_parser = Bio.PDB.PDBParser()
from scipy.spatial import distance_matrix
from Bio import BiopythonWarning

warnings.filterwarnings("error")
warnings.simplefilter('ignore', BiopythonWarning)
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', ResourceWarning)
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
#--------------------------------------------------------------------------------------#
#--------------------------------------------------------------------------------------#
"""
RUN COMMAND:
sinteractive --mem=50g --cpus-per-task=20
module load singularity
singularity exec -B /path/to/hoangd2_data/,/path/to/DCA_ER/manu/ /path/to/DCA_ER/erdca_regularized.simg python run_singlePFAM_ER.py %PFAM_ID% $SLURM_CPUS_PER_TASK $SLURM_JOB_ID

ex:
singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/manu/ /data/cresswellclayec/DCA_ER/erdca_regularized.simg python run_singlePFAM_ER.py PF00186 $SLURM_CPUS_PER_TASK $SLURM_JOB_ID
"""
#--------------------------------------------------------------------------------------#

# Define paths to protein data (Pfam-A.full/) and preprocessed data (pfam_ecc/)
#data_path = '/home/eclay/Pfam-A.full'
#preprocess_path = '/home/eclay/DCA_ER/biowulf/pfam_ecc/'
data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'
preprocess_path = '/data/cresswellclayec/DCA_ER/manu/pfam_ecc/'


#------ Command Line Arguments -----#
#pfam_id = 'PF00025'
pfam_id = sys.argv[1]

cpus_per_job = int(sys.argv[2])

job_id = sys.argv[3]
print("Calculating DI for %s using %d (of %d) threads (JOBID: %s)"%(pfam_id,cpus_per_job-4,cpus_per_job,job_id))
#-----------------------------------#



#------------------------------------#
#--------- Load PDB file ------------#

# Read in Reference Protein Structure
pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))                                                                                                                   
# convert bytes to str (python 2 to python 3)                                                                       
pdb = np.array(
               [pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0])  \
    	       for i in range(pdb.shape[1])]  \
              ).reshape(pdb.shape[0],pdb.shape[1])

ipdb = 0
tpdb = int(pdb[ipdb,1])

# Load Multiple Sequence Alignment (MSA)
s = dp.load_msa(data_path,pfam_id)

print('pdb line 0: \n',pdb[ipdb,:])
pdb_id = pdb[ipdb,5]                                                                              
pdb_chain = pdb[ipdb,6]                                                                           
pdb_start,pdb_end = int(pdb[ipdb,7]),int(pdb[ipdb,8])                                             
pdb_range = [pdb_start-1, pdb_end]
#print('pdb id, chain, start, end, length:',pdb_id,pdb_chain,pdb_start,pdb_end,pdb_end-pdb_start+1)                        

#print('download pdb file')                                                                       
pdb_file = pdb_list.retrieve_pdb_file(str(pdb_id),file_format='pdb')                              
#pdb_file = pdb_list.retrieve_pdb_file(pdb_id)                                                    

#------------------------------------#
#------------------------------------#


pfam_dict = {}
#---------------------------------------------------------------------------------------------------------------------#            
#--------------------------------------- Create PDB-PP Reference Sequence --------------------------------------------#            
#---------------------------------------------------------------------------------------------------------------------#            
chain = pdb_parser.get_structure(str(pdb_id),pdb_file)[0][pdb_chain] 
ppb = PPBuilder().build_peptides(chain)                                                       
#    print(pp.get_sequence())
print('peptide build of chain produced %d elements\n\n'%(len(ppb)))                               

matching_seq_dict = {}
poly_seq = list()
for i,pp in enumerate(ppb):
    for char in str(pp.get_sequence()):
        poly_seq.append(char)                                     
print('PDB Polypeptide Sequence: \n',poly_seq)

poly_seq_range = poly_seq[pdb_range[0]:pdb_range[1]]
print('PDB Polypeptide Sequence (In Proteins PDB range len=%d): \n'%len(poly_seq_range),poly_seq_range)
if len(poly_seq_range) < 10:
	print('PP sequence overlap with PDB range is too small.\nWe will find a match\nBAD PDB-RANGE')
	poly_seq_range = poly_seq
else:
	pp_msa_file_range, pp_ref_file_range = tools.write_FASTA(poly_seq_range, s, pfam_id, number_form=False,processed=False,path='./pfam_ecc/',nickname='range')

pp_msa_file, pp_ref_file = tools.write_FASTA(poly_seq, s, pfam_id, number_form=False,processed=False,path='./pfam_ecc/')
#---------------------------------------------------------------------------------------------------------------------#            


#---------------------------------------------------------------------------------------------------------------------#            
#---------------------------------- PreProcess FASTA Alignment -------------------------------------------------------#            
#---------------------------------------------------------------------------------------------------------------------#            

trimmed_data_outfile = preprocess_path+'MSA_%s_Trimmed.fa'%pfam_id

print('Trimming MSA')
try:
	print('\n\nPre-Processing MSA with Range PP Seq\n\n')
	trimmer = msa_trimmer.MSATrimmer(
	    pp_msa_file_range, biomolecule='PROTEIN', 
	    refseq_file=pp_ref_file_range
	)
	pfam_dict['ref_file'] = pp_ref_file_range
except:
	print('\nDidnt work, using full PP seq\nPre-Processing MSA wth PP Seq\n\n')
	sys.exit()
	"""
	# create MSATrimmer instance 
	trimmer = msa_trimmer.MSATrimmer(
	    pp_msa_file, biomolecule='protein', 
	    refseq_file=pp_ref_file
	)
	pfam_dict['ref_file'] = pp_ref_file
	"""
try:
	trimmed_data = trimmer.get_msa_trimmed_by_refseq(remove_all_gaps=True)
	print('\n\nTrimmed Data: \n',trimmed_data[:10])

except(MSATrimmerException):
	ERR = 'PPseq-MSA'
	print('Error with MSA trimms\n%s\n'%ERR)
	sys.exit()


#--- Write trimmed msa to file in FASTA format ---#
with open(trimmed_data_outfile, 'w') as fh:
	for seqid, seq in trimmed_data:
		fh.write('>{}\n{}\n'.format(seqid, seq))
fh.close()

s_index = list(np.arange(len(''.join(seq))))
#---------------------------------------------------------------------------------------------------------------------#            
#---------------------------------------------------------------------------------------------------------------------#            
#---------------------------------------------------------------------------------------------------------------------#            




#---------------------------------------------------------------------------------------------------------------------#            
#----------------------------------------- Run Simulation ERDCA ------------------------------------------------------#            
#---------------------------------------------------------------------------------------------------------------------#            

try:
	print('Initializing ER instance\n\n')
	# Compute DI scores using Expectation Reflection algorithm
	erdca_inst = erdca.ERDCA(
	    #preprocessed_data_outfile,
	    trimmed_data_outfile,
	    'PROTEIN',
	    refseq_file=pp_ref_file_range,
	    s_index = s_index,
	    pseudocount = 0.5,
	    num_threads = cpus_per_job-4,
	    seqid = 0.8)

except:
	ref_seq = s[tpdb,:]
	print('\nERROR using trimmed DATA\n\nUsing PDB defined reference sequence from MSA:\n',ref_seq)

	sys.exit()

	"""
	msa_file, ref_file = tools.write_FASTA(ref_seq, s, pfam_id, number_form=False,processed=False,path=preprocess_path)
	pfam_dict['ref_file'] = ref_file

	print('Re-trimming MSA with pdb index defined ref_seq\n\n\n')
	# create MSATrimmer instance 
	trimmer = msa_trimmer.MSATrimmer(
	    msa_file, biomolecule='protein', 
	    refseq_file=ref_file
	)

	preprocessed_data,s_index, cols_removed,s_ipdb,s = trimmer.get_preprocessed_msa(printing=True, saving = False)
	#write trimmed msa to file in FASTA format
	with open(preprocessed_data_outfile, 'w') as fh:
		for seqid, seq in preprocessed_data:
			fh.write('>{}\n{}\n'.format(seqid, seq))

	erdca_inst = erdca.ERDCA(
		    preprocessed_data_outfile,
		    'PROTEIN',
		    s_index = s_index,
		    pseudocount = 0.5,
		    num_threads = cpus_per_job-4,
		    seqid = 0.8)
	"""


print('Running ER simulation\n\n')

start_time = timeit.default_timer()

erdca_DI = erdca_inst.compute_sorted_DI(LAD=False,init_w = None) # init_w initialized in code

run_time = timeit.default_timer() - start_time
print('ER run time:',run_time)


# Print resulting DI score
print('Top 5 DIs')
for site_pair, score in erdca_DI[:5]:
    print(site_pair, score)

with open('DI/ER/er_coup_DI_%s.pickle'%(pfam_id), 'wb') as f:
    pickle.dump(erdca_DI, f)
f.close()

# Save processed data dictionary and FASTA file
print('s shape (msa): ',s.shape)
pfam_dict['msa'] = s  
pfam_dict['processed_msa'] = trimmed_data 
pfam_dict['s_ipdb'] = tpdb
pfam_dict['cols_removed'] = []

input_data_file = preprocess_path+"%s_DP_ER_coup.pickle"%(pfam_id)
with open(input_data_file,"wb") as f:
	pickle.dump(pfam_dict, f)
f.close()

#---------------------------------------------------------------------------------------------------------------------#            



plotting = False
if plotting:
	# Print Details of protein PDB structure Info for contact visualizeation
	print('Using chain ',pdb_chain)
	print('PDB ID: ', pdb_id)

	from pydca.contact_visualizer import contact_visualizer

	visualizer = contact_visualizer.DCAVisualizer('protein', pdb_chain, pdb_id,
	refseq_file = pp_ref_file,
	sorted_dca_scores = erdca_DI,
	linear_dist = 4,
	contact_dist = 8.)

	contact_map_data = visualizer.plot_contact_map()
	#plt.show()
	#plt.close()
	tp_rate_data = visualizer.plot_true_positive_rates()
	#plt.show()
	#plt.close()
	#print('Contact Map: \n',contact_map_data[:10])
	#print('TP Rates: \n',tp_rate_data[:10])

	with open(preprocess_path+'ER_%s_contact_map_data.pickle'%(pfam_id), 'wb') as f:
	    pickle.dump(contact_map_data, f)
	f.close()

	with open(preprocess_path+'ER_%s_tp_rate_data.pickle'%(pfam_id), 'wb') as f:
	    pickle.dump(tp_rate_data, f)
	f.close()



