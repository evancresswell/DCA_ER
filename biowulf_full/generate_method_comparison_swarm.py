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
boot_auc_files = list(Path(out_metric_dir).rglob("*bootstrap_aucs.npy"))
print(len(boot_auc_files))
boot_auc_files_str = [str(os.path.basename(path)) for path in boot_auc_files]
pfam_ids = [tp_str[:7] for tp_str in boot_auc_files_str] 
pdb_ids = [tp_str[8:12] for tp_str in boot_auc_files_str] 

f = open('ks_exact.swarm','w')
for i, pdb_id  in enumerate(pdb_ids):    
    pfam_id = pfam_ids[i]
    f.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
    f.write('conda activate plotting; ')
    f.write('python run_ks_comp_exact.py %s %s $SLURM_CPUS_PER_TASK\n'%(pdb_id, pfam_id))
    #f.write('module load singularity; ')
    #f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_ER.py %s\n'%(pdb))
f.close()
 

f = open('compare_asymptotic.swarm','w')
for i, pdb_id  in enumerate(pdb_ids):    
    pfam_id = pfam_ids[i]
    f.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
    f.write('conda activate DCA_ER; ')
    f.write('python run_method_comparison.py %s %s $SLURM_CPUS_PER_TASK\n'%(pdb_id, pfam_id))
    #f.write('module load singularity; ')
    #f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_ER.py %s\n'%(pdb))
f.close()

