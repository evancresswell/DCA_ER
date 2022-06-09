import os.path, sys

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import distance_matrix

import timeit
#import emachine as EM

# import data processing and general DCA_ER tools
from pathlib import Path
np.random.seed(1)



data_path = Path('/data/cresswellclayec/DCA_ER/Pfam-A.full')
data_path = Path('/data/cresswellclayec/Pfam-A.full')

# Define data directories
DCA_ER_dir = '/data/cresswellclayec/DCA_ER' # Set DCA_ER directory
biowulf_dir = '%s/biowulf_full' % DCA_ER_dir


out_dir = '%s/protein_data/di/' % biowulf_dir
out_metric_dir = '%s/protein_data/metrics/' % biowulf_dir

processed_data_dir = "%s/protein_data/data_processing_output" % biowulf_dir
pdb_dir = '%s/protein_data/pdb_data/' % biowulf_dir


pdb_path = "/pdb/pdb/"
dir_path = "/data/cresswellclayec/DCA_ER/biowulf_full/protein_data/di/"
ks_path = "/data/cresswellclayec/DCA_ER/biowulf_full/protein_data/metrics/"

# Get list of files from completed auc-bootstrap files
if 0:
   boot_auc_files = list(Path(out_metric_dir).rglob("*bootstrap_aucs.npy"))
   print(len(boot_auc_files))
   boot_auc_files_str = [str(os.path.basename(path)) for path in boot_auc_files]
   pfam_ids = [tp_str[:7] for tp_str in boot_auc_files_str] 
   pdb_ids = [tp_str[8:12] for tp_str in boot_auc_files_str] 

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
comparison_pdb_str_set = set.intersection(set(ER_pdb_ids), set(PLM_pdb_ids), set(PMF_pdb_ids))
comparison_pdb_set = [pdb for pdb in comparison_pdb_str_set]

if 0:
    # if we dont want to run comparison on methods that have already been compared
    compared_files = list(Path(out_metric_dir).rglob("*ER*"))
    compared_files_str = [str(os.path.basename(path)) for path in ER_di_files]
    compared_pdb_ids = [di_str[:4] for di_str in ER_di_files_str] 
    
comparison_pfam_set = [ER_pfam_ids[ER_pdb_ids.index(pdb)] for pdb in comparison_pdb_set]


# ks exact and asymptotic now done in run_method_comparison
if 0:
    f = open('ks_exact.swarm','w')
    for i, pdb_id  in enumerate(comparison_pdb_set):    
        pfam_id = comparison_pfam_set[i]
        f.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
        f.write('conda activate plotting; ')
        f.write('python run_ks_comp_exact.py %s %s $SLURM_CPUS_PER_TASK\n'%(pdb_id, pfam_id))
        #f.write('module load singularity; ')
        #f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_ER.py %s\n'%(pdb))
    f.close()
 

f = open('method_comparison_ks.swarm','w')
for i, pdb_id  in enumerate(comparison_pdb_set):    
    pfam_id = comparison_pfam_set[i]
    f.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
    f.write('conda activate DCA_ER; ')
    f.write('python run_method_comparison.py %s %s $SLURM_CPUS_PER_TASK\n'%(pdb_id, pfam_id))
    #f.write('module load singularity; ')
    #f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_ER.py %s\n'%(pdb))
f.close()

