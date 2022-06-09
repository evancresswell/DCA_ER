import os, sys
import numpy as np
from pathlib import Path
import os
import numpy as np
import pickle as pkl
import pandas as pd

# arguments 
swarm_list_file = sys.argv[1] # data_prep swarm file
if len(sys.argv) > 2:
    swarm_id  = int(sys.argv[2])  # the swarm simulation ID of the executred data-prep swarm


swarm_path_list = np.loadtxt(swarm_list_file,dtype='str')
# pdb_list = [path[15:19] for path in pdb_path_list] # for pdb list txt files
pdb_list = [path[7][15:19] for path in swarm_path_list] # for pdb list swarm files (ER)
print(pdb_list[:5])
pdb_path_list = ["/pdb/pdb/%s/pdb%s.ent.gz" % (pdb[1:3], pdb) for pdb in pdb_list]
print(pdb_path_list[:5])


# Define data directories
DCA_ER_dir = '/data/cresswellclayec/DCA_ER' # Set DCA_ER directory
biowulf_dir = '%s/biowulf_full' % DCA_ER_dir
pdb_dir = '%s/protein_data/pdb_data/' % biowulf_dir

# list of files that are created during completion of data prep.
pdb_out_paths = ['%s%s_pdb_df.csv' % (pdb_dir, pdb_id) for pdb_id in pdb_list]



finding_completed = True # We want PDB-Pfam with data-prep completed
if finding_completed:
    # to find PDB's with data-prep NOT COMPLETED
    new_pdb_list = []
    for i, pdb_list_path in enumerate(pdb_out_paths):
        if os.path.exists(pdb_list_path):
            continue
        else:
            new_pdb_list.append(pdb_list[i])
else: 
    # to find PDB's with data-prep COMPLETED 
    new_pdb_list = []
    for i, pdb_list_path in enumerate(pdb_out_paths):
        if os.path.exists(pdb_list_path):
            new_pdb_list.append(pdb_list[i])
        else:
            continue

idx = swarm_list_file.index('.swarm')
pdb_list_index = int(swarm_list_file[idx-1])

#--------------------------------------------------------------#
if finding_completed:

    # If looking for completed data-prep files the next step is simulations.
    # we need to break up by size of protein (this is encoded in the memory required during data_prep simulation..

    # load in ER-swarm memory results
    swarm_mem_file = '%d_mem.tsv' % swarm_id
    mem_max_label = 'job_id'
    jobid_label = 'mem_max'

    print('running:\ndashboard_cli jobs --args... jobidarray,mem_max ')
    os.system('dashboard_cli jobs -u cresswellclayec --tab --jobid %d_* --fields jobidarray,mem_max > %s' % (swarm_id, swarm_mem_file))
    swarm_mem_df = pd.read_csv(swarm_mem_file, delimiter='\t', header=None, names = [jobid_label, mem_max_label])
    print(swarm_mem_df.head())
    print(len(swarm_mem_df))
    swarm_mem_df = swarm_mem_df.drop([0], axis=0)
    print(len(swarm_mem_df))

    print(swarm_mem_df.head())
    label_list =  swarm_mem_df.columns.values.tolist()
    print(label_list)
    swarm_mem_df[mem_max_label] = swarm_mem_df[mem_max_label].map(lambda mem_string: int(mem_string.split(' ')[-1][:-2]))
    swarm_mem_df = swarm_mem_df.sort_values(by=mem_max_label, ascending=False)
    print(swarm_mem_df.head())

    swarm_750p = swarm_mem_df.loc[swarm_mem_df[mem_max_label] >= 750][jobid_label].tolist()
    print(swarm_750p)
    swarm_mem_df = swarm_mem_df.drop(swarm_mem_df.loc[swarm_mem_df[mem_max_label] >= 750].index)
    print(swarm_mem_df.head())
    print(len(swarm_mem_df))
    swarm_500p = swarm_mem_df.loc[swarm_mem_df[mem_max_label] >= 500][jobid_label].tolist()
    swarm_mem_df = swarm_mem_df.drop(swarm_mem_df.loc[swarm_mem_df[mem_max_label] >= 500].index)
    print(swarm_500p)
    print(swarm_mem_df.head())
    print(len(swarm_mem_df))
    swarm_300p = swarm_mem_df.loc[swarm_mem_df[mem_max_label] >= 300][jobid_label].tolist()
    swarm_mem_df = swarm_mem_df.drop(swarm_mem_df.loc[swarm_mem_df[mem_max_label] >= 300].index)
    print(swarm_300p)
    print(swarm_mem_df.head())
    print(len(swarm_mem_df))
    swarm_150p = swarm_mem_df.loc[swarm_mem_df[mem_max_label] >= 150][jobid_label].tolist()
    swarm_mem_df = swarm_mem_df.drop(swarm_mem_df.loc[swarm_mem_df[mem_max_label] >= 150].index)
    print(swarm_150p)
    print(swarm_mem_df.head())
    print(len(swarm_mem_df))
    swarm_small = swarm_mem_df[swarm_mem_df.columns.values.tolist()[0]].values.tolist()
    print(swarm_small)

    # create swarm files 
    # assumes data-prep out bundles in groups of 20!!!!
    #--------------------------------------------------------------#
    f_ER_750p_pdbids = []
    for job_id in swarm_750p:
        with open('swarm_output/swarm_%s.o' % job_id) as swarm_out:
            head = [next(swarm_out) for x in range(22)]
        swarm_out.close() 
        f_ER_750p_pdbids.extend([swarm_out_row[121:125] for swarm_out_row in head[1:-1]])
        #print(len(f_ER_750p_pdbids))
        #print(f_ER_750p_pdbids)

    with open('pdb2msa_ER_%d_750p.swarm' % pdb_list_index,'w') as f_ER_750p:
        for pdb_id in f_ER_750p_pdbids:
            f_ER_750p.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
            f_ER_750p.write('conda activate DCA_ER; ')
            f_ER_750p.write('python run_ER.py %s $SLURM_CPUS_PER_TASK\n'% pdb_id)
    f_ER_750p.close()
    #--------------------------------------------------------------#
    f_ER_500p_pdbids = []
    for job_id in swarm_500p:
        with open('swarm_output/swarm_%s.o' % job_id) as swarm_out:
            head = [next(swarm_out) for x in range(22)]
        swarm_out.close() 
        f_ER_500p_pdbids.extend([swarm_out_row[121:125] for swarm_out_row in head[1:-1]])
        #print(len(f_ER_500p_pdbids))
        #print(f_ER_500p_pdbids)

    with open('pdb2msa_ER_%d_500p.swarm' % pdb_list_index,'w') as f_ER_500p:
        for pdb_id in f_ER_500p_pdbids:
            f_ER_500p.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
            f_ER_500p.write('conda activate DCA_ER; ')
            f_ER_500p.write('python run_ER.py %s $SLURM_CPUS_PER_TASK\n'% pdb_id)
    f_ER_500p.close()
    #--------------------------------------------------------------#
    f_ER_300p_pdbids = []
    for job_id in swarm_300p:
        with open('swarm_output/swarm_%s.o' % job_id) as swarm_out:
            head = [next(swarm_out) for x in range(22)]
        swarm_out.close() 
        f_ER_300p_pdbids.extend([swarm_out_row[121:125] for swarm_out_row in head[1:-1]])
        #print(len(f_ER_300p_pdbids))
        #print(f_ER_300p_pdbids)

    with open('pdb2msa_ER_%d_300p.swarm' % pdb_list_index,'w') as f_ER_300p:
        for pdb_id in f_ER_300p_pdbids:
            f_ER_300p.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
            f_ER_300p.write('conda activate DCA_ER; ')
            f_ER_300p.write('python run_ER.py %s $SLURM_CPUS_PER_TASK\n'% pdb_id)
    f_ER_300p.close()
    #--------------------------------------------------------------#
    f_ER_150p_pdbids = []
    for job_id in swarm_150p:
        with open('swarm_output/swarm_%s.o' % job_id) as swarm_out:
            head = [next(swarm_out) for x in range(22)]
        swarm_out.close() 
        f_ER_150p_pdbids.extend([swarm_out_row[121:125] for swarm_out_row in head[1:-1]])
        #print(len(f_ER_150p_pdbids))
        #print(f_ER_150p_pdbids)

    with open('pdb2msa_ER_%d_150p.swarm' % pdb_list_index,'w') as f_ER_150p:
        for pdb_id in f_ER_150p_pdbids:
            f_ER_150p.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
            f_ER_150p.write('conda activate DCA_ER; ')
            f_ER_150p.write('python run_ER.py %s $SLURM_CPUS_PER_TASK\n'% pdb_id)
    f_ER_150p.close()
    #--------------------------------------------------------------#
    f_ER_small_pdbids = []
    for job_id in swarm_small:
        with open('swarm_output/swarm_%s.o' % job_id) as swarm_out:
            head = [next(swarm_out) for x in range(22)]
        swarm_out.close() 
        f_ER_small_pdbids.extend([swarm_out_row[121:125] for swarm_out_row in head[1:-1]])
        #print(len(f_ER_small_pdbids))
        #print(f_ER_small_pdbids)
    with open('pdb2msa_ER_%d_small.swarm' % pdb_list_index,'w') as f_ER_small:
        for pdb_id in f_ER_small_pdbids:
            f_ER_small.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
            f_ER_small.write('conda activate DCA_ER; ')
            f_ER_small.write('python run_ER.py %s $SLURM_CPUS_PER_TASK\n'% pdb_id)
    f_ER_small.close()
 
    sys.exit()
    #--------------------------------------------------------------#
    #--------------------------------------------------------------#
    #--------------------------------------------------------------#
    #--------------------------------------------------------------#
    # create swarm file for non-completed data-prep simulations
    f_plm = open('pdb2msa_data_prep_%da.swarm' % pdb_list_index,'w')
    new_file = swarm_list_file[:idx] + 'a' + swarm_list_file[idx:]    
    f_text = open(new_file,'w')
    
    #for pdb in s_er:
    for i, pdb in enumerate(new_pdb_list):
        f_swarm.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
        f_swarm.write('conda activate DCA_ER; ')
        f_swarm.write('python data_prep.py %s\n' % pdb_path_list[i])
        #f.write('module load singularity; ')
        #f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_ER.py %s\n'%(pdb))
    f_swarm.close()
    f_text.close()
     
    #--------------------------------------------------------------#
else: 
    
    #--------------------------------------------------------------#
    # create swarm file for non-completed data-prep simulations
    f_swarm = open('pdb2msa_data_prep_%da.swarm' % pdb_list_index,'w')
    new_file = swarm_list_file[:idx] + 'a' + swarm_list_file[idx:]    
    f_text = open(new_file,'w')
    
    #for pdb in s_er:
    for i, pdb in enumerate(new_pdb_list):
        f_swarm.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
        f_swarm.write('conda activate DCA_ER; ')
        f_swarm.write('python data_prep.py %s\n' % pdb_path_list[i])
        #f.write('module load singularity; ')
        #f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_ER.py %s\n'%(pdb))
    f_swarm.close()
    f_text.close()
     
    #--------------------------------------------------------------#
    
    
