import glob
import numpy as np
from pathlib import Path
from math import floor
import timeit
import os
import random

pdb_path = "/pdb/pdb/"
result = list(Path(pdb_path).rglob("*.[eE][nN][tT].[gG][zZ]"))
pdb_path_str = [str(path) for path in result]

short_list = random.sample(pdb_path_str, 200)
print(short_list)

if 0:	
    pfam_id_list = ['PF00011','PF00014','PF00017','PF00018','PF00025','PF00027','PF00028','PF00035',\
		       'PF00041','PF00043','PF00044','PF00046','PF00056','PF00059','PF00071','PF00073',\
		       'PF00076','PF00081','PF00084','PF00085','PF00091','PF00092','PF00105']

    f = open('pfam_list.txt','w')
    for pfam_id in pfam_id_list:    
        f.write('%s\n'%(pfam_id))
    f.close() 



f = open('pdb_list.txt','w')
for pdb_file in short_list:
    f.write('%s\n'  % pdb_file)
f.close() 

