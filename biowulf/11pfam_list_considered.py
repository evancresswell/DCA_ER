import glob
import numpy as np
from math import floor
import timeit

short_list = ['PF00011','PF00014','PF00017','PF00018','PF00025','PF00027','PF00028','PF00035',\
		       'PF00041','PF00043','PF00044','PF00046','PF00056','PF00059','PF00071','PF00073',\
		       'PF00076','PF00081','PF00084','PF00085','PF00091','PF00092','PF00105']

if 0:	
    pfam_id_list = ['PF00011','PF00014','PF00017','PF00018','PF00025','PF00027','PF00028','PF00035',\
		       'PF00041','PF00043','PF00044','PF00046','PF00056','PF00059','PF00071','PF00073',\
		       'PF00076','PF00081','PF00084','PF00085','PF00091','PF00092','PF00105']

    f = open('pfam_list.txt','w')
    for pfam_id in pfam_id_list:    
        f.write('%s\n'%(pfam_id))
    f.close() 



path_to_pfam = "/data/cresswellclayec/DCA_ER/Pfam-A.full/"
path_to_pfam = "/data/cresswellclayec/Pfam-A.full/"

from data_processing import load_msa

msa_sizes = {}
# loop through all non-empty msa.npy files in Pfam-A.full

start_time = timeit.default_timer()
f = open('full_pfam_list.txt','w')
f_big = open('large_enough_pfam_list.txt', 'w')
full_pfam_id_list = []
for pfam_path in glob.glob('%s*/' % path_to_pfam):    
    pfam_id = pfam_path.split('Pfam-A.full/')[1].split('/')[0]
    try:
        s = load_msa(path_to_pfam, pfam_id)
    except(FileNotFoundError, ValueError):
        continue
    try: 
        msa_sizes[pfam_id] = s.shape
    except(AttributeError):
        continue

    print('%s size: ' % pfam_id, s.shape)
    if s.shape[0] > 300:
        f_big.write('%s\n'%(pfam_id))
        f.write('%s\n'%(pfam_id))
        full_pfam_id_list.append(pfam_id)
    else:
        f.write('%s\n'%(pfam_id))
f.close() 
f_big.close()


np.save('msa_sizes_dict.npy', msa_sizes)
print(msa_sizes)


msa_sizes_ordered = {k: v for k, v in sorted(msa_sizes.items(), key=lambda item: item[1][0])}
print(msa_sizes_ordered)

block_num = 20
block_size = floor(len(msa_sizes_ordered)/block_num)
full_pfam_id_list = list(msa_sizes_ordered.keys())
pfam_blocks = [full_pfam_id_list[x:x+block_size] for x in range(0, len(full_pfam_id_list), block_size)]

for i, pfam_id_list in enumerate(pfam_blocks):
    f = open('pfam_list_%d.txt' % i, 'w')
    for pfam_id in pfam_id_list:
       f.write('%s\n'  % pfam_id)
    f.close() 
run_time = timeit.default_timer() - start_time
print('run time:',run_time)

