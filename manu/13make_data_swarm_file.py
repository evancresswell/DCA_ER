import os
import data_processing as dp
import numpy as np
from subtract_lists import subtract_lists
from joblib import Parallel, delayed

#pfam_list = np.loadtxt('pfam_list.txt',dtype='str')
#s1 = np.loadtxt('pfam_10_20k.txt',dtype='str')
#s2 = np.loadtxt('pfam_20_40k.txt',dtype='str')
#s3 = np.loadtxt('pfam_40_100k.txt',dtype='str')

#s = np.vstack([s1,s2])
#s = np.vstack([s,s3])

#s = np.loadtxt('pfam_10_20k.txt',dtype='str')
s_er = np.loadtxt('test_list.txt',dtype='str')
s_plm = np.loadtxt('test_list.txt',dtype='str')
s_mf = np.loadtxt('test_list.txt',dtype='str')

s_er = np.loadtxt('pfam_pdb_list.txt',dtype='str')
s_plm = np.loadtxt('pfam_pdb_list.txt',dtype='str')
s_mf = np.loadtxt('pfam_pdb_list.txt',dtype='str')

#n = s.shape[0]
#pfam_list = s[:,0]
print( s_er)

#--------------------------------------------------------------#
#--------------------------------------------------------------#
# create swarmfiles for each method
data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'
def get_msa_size(data_path,pfam):
    s = np.load('%s/%s/msa.npy'%(data_path,pfam)).T
    return s.shape[0]

seq_num = Parallel(n_jobs = 16)(delayed(get_msa_size)(data_path,s_er[i0])for i0 in range(len(s_er)))

top_indices = sorted(range(len(seq_num)), key=lambda i: seq_num[i], reverse=True)
top_10p_size = [seq_num[i] for i in top_indices[:int(round(.10*len(seq_num)))]]
top_10p_pfam = [s_er[i] for i in top_indices[:int(round(.10*len(seq_num)))]]
print('Pfams for largemem (top %d): '%int(round(.10*len(seq_num))),top_10p_pfam)


f = open('erdca.swarm','w')
f_large = open('erdca_large.swarm','w')
for pfam in s_er:
    #f.write('python 1main_DCA.py %s\n'%(pfam))
    if pfam in top_10p_pfam:
        f_large.write('singularity exec -B /data/cresswellclayec/DCA_ER/,/data/cresswellclayec/hoangd2_data/ /data/cresswellclayec/DCA_ER/erdca.simg python run_data_prep_MSA-PDB.py %s 16 && muscle -profile -in1 pfam_ecc/MSA_%s_.fa -in2 pfam_ecc/PP_ref_%s_.fa -out pfam_ecc/PP_muscle_msa_%s.fa  && singularity exec -B /data/cresswellclayec/DCA_ER/,/data/cresswellclayec/hoangd2_data/ /data/cresswellclayec/DCA_ER/erdca.simg python run_ER_MSA-PDB.py %s $SLURM_CPUS_PER_TASK\n'%(pfam,pfam,pfam,pfam,pfam))    
    else:
        f.write('singularity exec -B /data/cresswellclayec/DCA_ER/,/data/cresswellclayec/hoangd2_data/ /data/cresswellclayec/DCA_ER/erdca.simg python run_data_prep_MSA-PDB.py %s 16 && muscle -profile -in1 pfam_ecc/MSA_%s_.fa -in2 pfam_ecc/PP_ref_%s_.fa -out pfam_ecc/PP_muscle_msa_%s.fa  && singularity exec -B /data/cresswellclayec/DCA_ER/,/data/cresswellclayec/hoangd2_data/ /data/cresswellclayec/DCA_ER/erdca.simg python run_ER_MSA-PDB.py %s $SLURM_CPUS_PER_TASK\n'%(pfam,pfam,pfam,pfam,pfam))    
    #f.write('python 1main_ERM.py %s\n'%(pfam))
f.close()
f_large.close()

# runs for new version in reserve
if 0:
	f = open('old_dca.swarm','w')
	f_large = open('old_dca_large.swarm','w')
	for pfam in s_er:
	    #f.write('python 1main_DCA.py %s\n'%(pfam))
	    if pfam in top_10p_pfam:
		f_large.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/pydca-py37.simg python run_singlePFAM_DCA.py %s $SLURM_CPUS_PER_TASK $SLURM_ARRAY_JOB_ID\n'%(pfam))    
	    else:
		f.write('singularity exec -B /data/cresswellclayec/hoangd2_data/,/data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/erdca.simg python run_singlePFAM_DCA.py %s $SLURM_CPUS_PER_TASK $SLURM_ARRAY_JOB_ID\n'%(pfam))    
	    #f.write('python 1main_ERM.py %s\n'%(pfam))
	f.close()
	f_large.close()

