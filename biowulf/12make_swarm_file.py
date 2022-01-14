import os
import numpy as np

if 0:
	#pfam_list = np.loadtxt('pfam_list.txt',dtype='str')
	#s1 = np.loadtxt('pfam_10_20k.txt',dtype='str')
	#s2 = np.loadtxt('pfam_20_40k.txt',dtype='str')
	#s3 = np.loadtxt('pfam_40_100k.txt',dtype='str')

	#s = np.vstack([s1,s2])
	#s = np.vstack([s,s3])

	s = np.loadtxt('pfam_10_20k.txt',dtype='str')
	
	n = s.shape[0]
	pfam_list = s[:,0]

	#--------------------------------------------------------------
	# create pfam folder
	for i in range(n):
	    os.system('rm -r %s'%(pfam_list[i]))
	    os.system('mkdir %s'%(pfam_list[i]))

	#--------------------------------------------------------------
	# create swarmfile
	f = open('swarmfile.txt','w')
	for pfam in pfam_list:
	    #f.write('python 1main_DCA.py %s\n'%(pfam))
	    f.write('python 1main_ER.py %s\n'%(pfam))    
	    #f.write('python 1main_ERM.py %s\n'%(pfam))

	f.close()


import os
import numpy as np

pfam_list = np.loadtxt('pfam_list.txt',dtype='str')
pfam_list = np.loadtxt('full_pfam_list.txt',dtype='str')
#s1 = np.loadtxt('pfam_10_20k.txt',dtype='str')
#s2 = np.loadtxt('pfam_20_40k.txt',dtype='str')
#s3 = np.loadtxt('pfam_40_100k.txt',dtype='str')

#s = np.vstack([s1,s2])
#s = np.vstack([s,s3])

#s = np.loadtxt('pfam_10_20k.txt',dtype='str')

#s_er = np.loadtxt('er_swarm.txt',dtype='str')
#s_plm = np.loadtxt('plm_swarm.txt',dtype='str')
#s_mf = np.loadtxt('mf_swarm.txt',dtype='str')

#n = s.shape[0]
#pfam_list = s[:,0]


#--------------------------------------------------------------#
#--------------------------------------------------------------#
# create swarmfiles for each method

f = open('er.swarm','w')
#for pfam in s_er:
for pfam in pfam_list:
    f.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
    f.write('conda activate DCA_ER; ')
    f.write('python run_ER.py %s $SLURM_CPUS_PER_TASK\n'%(pfam))
    #f.write('module load singularity; ')
    #f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_ER.py %s\n'%(pfam))
f.close()

f = open('mf.swarm','w')
#for pfam in s_plm:
for pfam in pfam_list:
    f.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
    f.write('conda activate DCA_ER; ')
    f.write('python run_MF.py %s $SLURM_CPUS_PER_TASK\n'%(pfam))
    #f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_PLM.py %s\n'%(pfam))
f.close()

f = open('pydca_mf.swarm','w')
#for pfam in s_plm:
for pfam in pfam_list:
    f.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
    f.write('conda activate PYDCA; ')
    f.write('python run_pydca_MF.py %s $SLURM_CPUS_PER_TASK\n'%(pfam))
    #f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_PLM.py %s\n'%(pfam))
f.close()

f = open('pydca_plm.swarm','w')
#for pfam in s_mf:
for pfam in pfam_list:
    #f.write('python 1main_DCA.py %s\n'%(pfam))    
    f.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
    f.write('conda activate PYDCA; ')
    f.write('python run_pydca_PLM.py %s $SLURM_CPUS_PER_TASK\n'%(pfam))    
    #f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_DCA.py %s\n'%(pfam))
f.close()
#--------------------------------------------------------------#


