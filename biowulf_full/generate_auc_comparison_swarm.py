import os.path, sys

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
from scipy.spatial import distance_matrix

import timeit
#import emachine as EM
from direct_info import direct_info

# import data processing and general DCA_ER tools
from pathlib import Path
np.random.seed(1)


from data_processing import pdb2msa, data_processing_pdb2msa

import ecc_tools as tools

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

ER_result = list(Path(dir_path).rglob("*ER_di.npy"))
ER_di_pdb_list = [str(path)[-22:-18] for path in ER_result]
ER_di_pfam_list = [str(path)[-17:-10] for path in ER_result]
ER_pdb_path_str = ['%s%s/pdb%s.ent.gz' % (pdb_path, pdb[1:3], pdb) for pdb in ER_di_pdb_list]
print('ER di: ', len(ER_di_pdb_list))
print(ER_di_pdb_list[:5])
print(ER_di_pfam_list[:5])

MF_result = list(Path(dir_path).rglob("*_MF_di.npy"))
MF_di_pdb_list = [str(path)[-22:-18] for path in MF_result]
MF_di_pfam_list = [str(path)[-17:-10] for path in MF_result]
MF_pdb_path_str = ['%s%s/pdb%s.ent.gz' % (pdb_path, pdb[1:3], pdb) for pdb in MF_di_pdb_list]
print('MF di: ', len(MF_di_pdb_list))
print(MF_di_pdb_list[:5])
print(MF_di_pfam_list[:5])



PLM_result = list(Path(dir_path).rglob("*PLM_di.npy"))
PLM_di_pdb_list = [str(path)[-23:-19] for path in PLM_result]
PLM_di_pfam_list = [str(path)[-18:-13] for path in PLM_result]
PLM_pdb_path_str = ['%s%s/pdb%s.ent.gz' % (pdb_path, pdb[1:3], pdb) for pdb in PLM_di_pdb_list]
print('PLM di: ', len(PLM_di_pdb_list))
print(PLM_di_pdb_list[:5])
print(PLM_di_pfam_list[:5])

PMF_result = list(Path(dir_path).rglob("*PMF_di.npy"))
PMF_di_pdb_list = [str(path)[-23:-19] for path in PMF_result]
PMF_di_pfam_list = [str(path)[-18:-13] for path in PMF_result]
PMF_pdb_path_str = ['%s%s/pdb%s.ent.gz' % (pdb_path, pdb[1:3], pdb) for pdb in PMF_di_pdb_list]
print('PMF di: ', len(PMF_di_pdb_list))
print(PMF_di_pdb_list[:5])
print(PMF_di_pfam_list[:5], '\n\n')


comparison_pdb_str_set = set.intersection (set(ER_di_pdb_list), set(PLM_di_pdb_list), set(PMF_di_pdb_list), set(MF_di_pdb_list))
print(len(comparison_pdb_str_set))
# print(comparison_pdb_str_set)
pdb_comparison_paths = ['%s%s/pdb%s.ent.gz' % (pdb_path,  pdb[1:3], pdb) for pdb in comparison_pdb_str_set]

f = open('comparison_pdb.swarm','w')
for pdb_id  in comparison_pdb_str_set:    
    f.write('source /data/cresswellclayec/conda/etc/profile.d/conda.sh; ')
    f.write('conda activate DCA_ER; ')
    f.write('python run_auc_allMethods.py %s $SLURM_CPUS_PER_TASK\n'%(pdb_id))
    #f.write('module load singularity; ')
    #f.write('singularity exec -B /data/cresswellclayec/DCA_ER/biowulf/ /data/cresswellclayec/DCA_ER/dca_er.simg python 1main_ER.py %s\n'%(pdb))
f.close()
 


