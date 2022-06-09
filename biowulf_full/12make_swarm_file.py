import os
import numpy as np
from pathlib import Path

# # Get memory used for every simulation. use this to stratify further simulations
# dashboard_cli jobs -u cresswellclayec --jobid 38114094_* --fields jobid,mem_max

import os
import numpy as np

#pdb_list = np.loadtxt('full_pdb_list.txt',dtype='str')

# DATA PREP
# Generate swarm for intial simulations
# Generates data prep simulation
if 1:
    pdb_path = "./"
    result = list(Path(pdb_path).rglob("pdb_list_*.txt"))
    pdb_lists = ["pdb_list_%d.txt" % i for i in range(10) ]
    
    for i, pdb_list_path in enumerate(pdb_lists):
        pdb_list = np.loadtxt(pdb_list_path, dtype='str')
        
        #--------------------------------------------------------------#
        #--------------------------------------------------------------#
        # create swarmfiles for each method
        
        f = open('pdb2msa_data_prep_%d.swarm' % i,'w')
        #for pdb in s_er:
        for pdb in pdb_list:
            f.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
            f.write('conda activate DCA_ER; ')
            f.write('python data_prep.py %s\n'%(pdb))
            #f.write('module load singularity; ')
            #f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_ER.py %s\n'%(pdb))
        f.close()
 
        

        #--------------------------------------------------------------#


# ER SIMULATION
# Generates ER simulations.
# organizes by memory used
# NEED data_prep swarm job_id!!!
if 0:
        f = open('pdb2msa_ER_%d.swarm' % i,'w')
        #for pdb in s_er:
        for pdb in pdb_list:
            f.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
            f.write('conda activate DCA_ER; ')
            f.write('python run_pdb2msa_ER.py %s $SLURM_CPUS_PER_TASK\n'%(pdb))
            #f.write('module load singularity; ')
            #f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_ER.py %s\n'%(pdb))
        f.close()



# Generate swarm for resulting simulations 
# Generates MF, PYMF, PYPLM simulations.
if 0:    
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

