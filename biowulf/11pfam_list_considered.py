import glob
if 0:	
    pfam_id_list = ['PF00011','PF00014','PF00017','PF00018','PF00025','PF00027','PF00028','PF00035',\
		       'PF00041','PF00043','PF00044','PF00046','PF00056','PF00059','PF00071','PF00073',\
		       'PF00076','PF00081','PF00084','PF00085','PF00091','PF00092','PF00105']

    f = open('pfam_list.txt','w')
    for pfam_id in pfam_id_list:    
        f.write('%s\n'%(pfam_id))
    f.close() 



path_to_pfam = "/data/cresswellclayec/DCA_ER/Pfam-A.full/"

from data_processing import load_msa

msa_sizes = {}
# loop through all non-empty msa.npy files in Pfam-A.full

f = open('full_pfam_list.txt','w')
for pfam_path in glob.glob('%s*/' % path_to_pfam):    
    pfam_id = pfam_path.split('Pfam-A.full/')[1].split('/')[0]
    s = load_msa(path_to_pfam, pfam_id)
    msa_sizes[pfam_id] = s.shape
    print('%s size: ' % pfam_id, s.shape)
    f.write('%s\n'%(pfam_id))
f.close() 

np.save('msa_sizes.npy', msa_sizes)
