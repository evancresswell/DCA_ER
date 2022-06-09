import os, sys
import numpy as np
from pathlib import Path
import os
import numpy as np
import pickle as pkl
import pandas as pd
from data_processing import pdb2msa

# Define data directories
DCA_ER_dir = '/data/cresswellclayec/DCA_ER' # Set DCA_ER directory
biowulf_dir = '%s/biowulf_full' % DCA_ER_dir
pdb_dir = '%s/protein_data/pdb_data/' % biowulf_dir
di_dir = '%s/protein_data/di/' % biowulf_dir
 
# Generates PYDCA swarm files from ER swarm fils
# 	-- assumes data_prep was run and ER swarm was run
if 0:
    # arguments 
    swarm_list_file = sys.argv[1] # data_prep swarm file
    if len(sys.argv) > 2:
        swarm_id  = int(sys.argv[2])  # the swarm simulation ID of the executred data-prep swarm
    
    
    er_swarm_path_list = np.loadtxt(swarm_list_file,dtype='str')
    # pdb_list = [path[15:19] for path in pdb_path_list] # for pdb list txt files
    pdb_list = [path[7][15:19] for path in er_swarm_path_list] # for pdb list swarm files (ER)
    print(pdb_list[:5])
    pdb_path_list = ["/pdb/pdb/%s/pdb%s.ent.gz" % (pdb[1:3], pdb) for pdb in pdb_list]
    er_zipped_pdb_list = [path[7] for path in er_swarm_path_list]
    
    
   
    pfam_ids = []
    for pdb_path in er_zipped_pdb_list:
        pdb_out_path = "%s%s" % (pdb_dir, pdb_path)
        try:
            pdb2msa_results = pdb2msa(pdb_out_path, pdb_dir, create_new=False)
            prody_df = pdb2msa_results[0]
            pdb2msa_row  = prody_df.iloc[0]
            pfam_ids.append(pdb2msa_row['Pfam'])
        except(FileNotFoundError):
            pfam_ids.append(None)
    
    print(pfam_ids[:5])
    
    # list of files that are created during completion of ER method
    di_out_paths = []
    for i, pdb_id in enumerate(pdb_list):
        if pfam_ids[i] is not None:
            di_out_paths.append('%s%s_%s_ER_di.npy' % (di_dir, pdb_id, pfam_ids[i]))
    
    # to find PDB's with data-prep NOT COMPLETED
    new_pdb_list = []
    for i, pdb_list_path in enumerate(di_out_paths):
        if pdb_list_path is None: 
            continue
        if os.path.exists(pdb_list_path):
            continue
        else:
            new_pdb_list.append(pdb_list[i])
    
    idx = swarm_list_file.index('.swarm')
    pdb_list_index = int(swarm_list_file[idx-1])
    
    #--------------------------------------------------------------#
    
    # If looking for completed data-prep files the next step is simulations.
    # we need to break up by size of protein (this is encoded in the memory required during data_prep simulation..
    
    
    # create swarm files 
    # assumes data-prep out bundles in groups of 20!!!!
    #--------------------------------------------------------------#
    with open('pdb2msa_PLM_%d.swarm' % pdb_list_index,'w') as f_plm:
        for pdb_id in new_pdb_list:
            f_plm.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
            f_plm.write('conda activate PYDCA; ')
            f_plm.write('python run_pdb2msa_PLM.py /pdb/pdb/%s/pdb%s.ent.gz $SLURM_CPUS_PER_TASK\n'% (pdb_id[1:3], pdb_id))
    f_plm.close()
    #--------------------------------------------------------------#
    with open('pdb2msa_PMF_%d.swarm' % pdb_list_index,'w') as f_pmf:
        for pdb_id in new_pdb_list:
            f_pmf.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
            f_pmf.write('conda activate PYDCA; ')
            f_pmf.write('python run_pdb2msa_PMF.py /pdb/pdb/%s/pdb%s.ent.gz $SLURM_CPUS_PER_TASK\n'% (pdb_id[1:3], pdb_id))
    f_pmf.close()
    #--------------------------------------------------------------#
    with open('pdb2msa_MF_%d.swarm' % pdb_list_index,'w') as f_mf:
        for pdb_id in new_pdb_list:
            f_mf.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
            f_mf.write('conda activate PYDCA; ')
            f_mf.write('python run_pdb2msa_MF.py /pdb/pdb/%s/pdb%s.ent.gz $SLURM_CPUS_PER_TASK\n'% (pdb_id[1:3], pdb_id))
    f_mf.close()
    #--------------------------------------------------------------#
    



# Generate PYDCA swarm by finding all ER di files for which there is not PYDCA sim.
if 1:
    out_dir = '%s/protein_data/di/' % biowulf_dir
    out_metric_dir = '%s/protein_data/metrics/' % biowulf_dir

    # get list of files with DI for all three methods
    ER_di_files = list(Path(out_dir).rglob("*ER*"))
    ER_di_files_str = [str(os.path.basename(path)) for path in ER_di_files]
    ER_pdb_ids = [di_str[:4] for di_str in ER_di_files_str] 
    ER_pfam_ids = [di_str[5:12] for di_str in ER_di_files_str] 
    PMF_di_files = list(Path(out_dir).rglob("*PMF*"))
    PMF_di_files_str = [str(os.path.basename(path)) for path in PMF_di_files]
    PMF_pdb_ids = [di_str[:4] for di_str in PMF_di_files_str] 
    PMF_pfam_ids = [di_str[5:12] for di_str in PMF_di_files_str] 
    PLM_di_files = list(Path(out_dir).rglob("*PLM*"))
    PLM_di_files_str = [str(os.path.basename(path)) for path in PLM_di_files]
    PLM_pdb_ids = [di_str[:4] for di_str in PLM_di_files_str] 
    PLM_pfam_ids = [di_str[5:12] for di_str in PLM_di_files_str] 
    
    # get intersection of sets
    print('PYDCA PLM files: ', len(PLM_pdb_ids))
    print('PYDCA PMF files: ', len(PMF_pdb_ids))
    print('ER files: ', len(ER_pdb_ids))
    PYDCA_str_set = set.intersection(set(PLM_pdb_ids), set(PMF_pdb_ids))
    pydca_pdb_set = [pdb for pdb in PYDCA_str_set]
    print('Intersection of PLM and PMF files: ', len(pydca_pdb_set))

    residual_pdb_ids = [pdb_id for pdb_id in ER_pdb_ids if pdb_id not in pydca_pdb_set]
    print('%d ER di have no PYDCA counterpart. making swarm simulation...' % len(residual_pdb_ids))
    
    with open('pdb2msa_PLM_residual.swarm','w') as f_plm:
        for pdb_id in residual_pdb_ids:
            f_plm.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
            f_plm.write('conda activate PYDCA; ')
            f_plm.write('python run_pdb2msa_PLM.py /pdb/pdb/%s/pdb%s.ent.gz $SLURM_CPUS_PER_TASK\n'% (pdb_id[1:3], pdb_id))
    f_plm.close()
    #--------------------------------------------------------------#
    with open('pdb2msa_PMF_residual.swarm','w') as f_pmf:
        for pdb_id in residual_pdb_ids:
            f_pmf.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
            f_pmf.write('conda activate PYDCA; ')
            f_pmf.write('python run_pdb2msa_PMF.py /pdb/pdb/%s/pdb%s.ent.gz $SLURM_CPUS_PER_TASK\n'% (pdb_id[1:3], pdb_id))
    f_pmf.close()
    
    
    
    
      
        
