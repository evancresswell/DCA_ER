import glob
import numpy as np
from pathlib import Path
from math import floor
import timeit
import os
import random

blocking = False

if blocking:
    pdb_path = "/pdb/pdb/"
    result = list(Path(pdb_path).rglob("*.[eE][nN][tT].[gG][zZ]"))
    pdb_path_str = [str(path) for path in result]
else:
    pdb_path = "/pdb/pdb/"
    dir_path = "/data/cresswellclayec/DCA_ER/biowulf_full/protein_data/di/"
    result = list(Path(dir_path).rglob("*ER_di.npy"))
    pdb_str_list = [str(path)[-22:-18] for path in result]
    pdb_path_str = ['%s%s/pdb%s.ent.gz' % (pdb_path, pdb[1:3], pdb) for pdb in pdb_str_list]
    print(len(pdb_path_str), pdb_path_str[:10])
short_list = random.sample(pdb_path_str, 200)

if 0:	
    pfam_id_list = ['PF00011','PF00014','PF00017','PF00018','PF00025','PF00027','PF00028','PF00035',\
		       'PF00041','PF00043','PF00044','PF00046','PF00056','PF00059','PF00071','PF00073',\
		       'PF00076','PF00081','PF00084','PF00085','PF00091','PF00092','PF00105']

    f = open('pfam_list.txt','w')
    for pfam_id in pfam_id_list:    
        f.write('%s\n'%(pfam_id))
    f.close() 


start_time = timeit.default_timer()
from data_processing import pdb2msa, data_processing_pdb2msa

import gzip, shutil
def gunzip(file_path, output_path):
    print('Unzipping %s to %s' % (file_path, output_path))
    with gzip.open(file_path,"rb") as f_in, open(output_path,"wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


f = open('pdb_short_list.txt','w')
for pdb_file in short_list:
    f.write('%s\n'  % pdb_file)
f.close() 

if blocking:
    # break list into chunnks of 20000 for biowulf simulation
    block_num = 20
    block_size = floor(len(pdb_path_str)/block_num)
    block_size = 20000
    pdb_blocks = [pdb_path_str[x:x+block_size] for x in range(0, len(pdb_path_str), block_size)]
    
    for i, pdb_id_list in enumerate(pdb_blocks):
        f = open('pdb_list_%d.txt' % i, 'w')
        for pdb_file in pdb_id_list:
            f.write('%s\n'  % pdb_file)
        f.close() 
else:
    
    f = open('pdb_ER_list.txt','w')
    for pdb_file in pdb_path_str:
        f.write('%s\n'  % pdb_file)
    f.close() 
    

run_time = timeit.default_timer() - start_time
print('run time:',run_time)
