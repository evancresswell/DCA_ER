import os
import numpy as np
from pathlib import Path

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

blocking = False
#pdb_list = np.loadtxt('full_pdb_list.txt',dtype='str')

if blocking:
    pdb_path = "./"
    result = list(Path(pdb_path).rglob("pdb_list_*.txt"))
    pdb_lists = [str(path) for path in result]
    
    for i, pdb_list_path in enumerate(pdb_lists):
        pdb_list = np.loadtxt(pdb_list_path, dtype='str')
        
        #--------------------------------------------------------------#
        #--------------------------------------------------------------#
        # create swarmfiles for each method
        
        f = open('pdb2msa_ER_%d.swarm' % i,'w')
        #for pdb in s_er:
        for pdb in pdb_list:
            f.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
            f.write('conda activate DCA_ER; ')
            f.write('python run_pdb2msa_ER.py %s $SLURM_CPUS_PER_TASK\n'%(pdb))
            #f.write('module load singularity; ')
            #f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_ER.py %s\n'%(pdb))
        f.close()
    
        f = open('pdb2msa_MF_%d.swarm' % i,'w')
        #for pdb in s_er:
        for pdb in pdb_list:
            f.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
            f.write('conda activate DCA_ER; ')
            f.write('python run_pdb2msa_MF.py %s $SLURM_CPUS_PER_TASK\n'%(pdb))
            #f.write('module load singularity; ')
            #f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_ER.py %s\n'%(pdb))
        f.close()
    
        #--------------------------------------------------------------#
else:    
    pdb_list_path = "pdb_ER_list.txt"
    pdb_list = np.loadtxt(pdb_list_path, dtype='str')
    
    #--------------------------------------------------------------#
    #--------------------------------------------------------------#
    # create swarmfiles for each method
    
    f = open('pdb2msa_MF.swarm','w')
    #for pdb in s_er:
    for pdb in pdb_list:
        f.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
        f.write('conda activate DCA_ER; ')
        f.write('python run_pdb2msa_MF.py %s $SLURM_CPUS_PER_TASK\n'%(pdb))
        #f.write('module load singularity; ')
        #f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_ER.py %s\n'%(pdb))
    f.close()

    f = open('pdb2msa_PLM.swarm','w')
    #for pdb in s_er:
    for pdb in pdb_list:
        f.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
        f.write('conda activate PYDCA; ')
        f.write('python run_pdb2msa_PLM.py %s $SLURM_CPUS_PER_TASK\n'%(pdb))
        #f.write('module load singularity; ')
        #f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_ER.py %s\n'%(pdb))
    f.close()

    f = open('pdb2msa_PMF.swarm','w')
    #for pdb in s_er:
    for pdb in pdb_list:
        f.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
        f.write('conda activate PYDCA; ')
        f.write('python run_pdb2msa_PMF.py %s $SLURM_CPUS_PER_TASK\n'%(pdb))
        #f.write('module load singularity; ')
        #f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_ER.py %s\n'%(pdb))
    f.close()

    #--------------------------------------------------------------#

