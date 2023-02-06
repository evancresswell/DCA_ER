import pickle
import itertools
from joblib import Parallel, delayed
import expectation_reflection as ER
from scipy.spatial import distance

def E(i1i2, s, w):
    E = 0
    s_len = len(i1i2)
    for i in range(s_len):
        i1,i2 = i1i2[i,0],i1i2[i,1]
        si_vec = s[i1:i2]
        for j in range(s_len):
            j1,j2 = i1i2[j,0],i1i2[j,1]
            sj_vec = s[j1:j2]
            E += np.dot(si_vec, np.dot(w[i1:i2,j1:j2], np.transpose(sj_vec)))
    return E


def energy_diff(i1i2, s1, s2, w):
    e_diff = 0.
    s_len = len(i1i2)

    E1 = E(i1i2, s1, w)
    E2 = E(i1i2, s2, w)
    #print(E1)
    #print(E2)
    
    e_diff1 = 0
    for i in range(s_len):
        i1,i2 = i1i2[i,0],i1i2[i,1]
        si_vec = s1[i1:i2]
        for j in range(s_len):
            j1,j2 = i1i2[j,0],i1i2[j,1]
            sj_vec = s2[j1:j2]
            e_diff1 += np.dot(si_vec, np.dot(w[i1:i2,j1:j2], np.transpose(sj_vec)))
    
    e_diff2 = 0
    for i in range(s_len):
        i1,i2 = i1i2[i,0],i1i2[i,1]
        si_vec = s2[i1:i2]
        for j in range(s_len):
            j1,j2 = i1i2[j,0],i1i2[j,1]
            sj_vec = s1[j1:j2]
            e_diff2 += np.dot(si_vec, np.dot(w[i1:i2,j1:j2], np.transpose(sj_vec)))
    return E1 + E2 - e_diff1 - e_diff2   


pdb_id = sys.argv[1]
n_cpus = int(sys.argv[2])
#pdb_id = "1zdr"


# Define data directories
pdb_path = "/pdb/pdb/%s/pdb%s.ent.gz" % (pdb_id[1:3], pdb_id)
data_path = Path('/data/cresswellclayec/Pfam-A.full')

# Define data directories
DCA_ER_dir = '/data/cresswellclayec/DCA_ER' # Set DCA_ER directory
biowulf_dir = '%s/biowulf_full' % DCA_ER_dir


out_dir = '%s/protein_data/di/' % biowulf_dir
out_metric_dir = '%s/protein_data/metrics/' % biowulf_dir

processed_data_dir = "%s/protein_data/data_processing_output/" % biowulf_dir
pdb_dir = '%s/protein_data/pdb_data/' % biowulf_dir

pdb_data_file = '%s/%s_pdb_df.csv' % (pdb_dir, pdb_id)

if os.path.exists(pdb_data_file):
    prody_df = pd.read_csv(pdb_data_file)
    pdb_file = "%spdb%s.ent" % (pdb_dir,pdb_id)


    pdb2msa_row = prody_df.iloc[0]
    pfam_id = pdb2msa_row['Pfam']
    pdb_chain = pdb2msa_row['Chain']
    print(pdb2msa_row)
else:
    unzipped_pdb_filename = os.path.basename(pdb_path).replace(".gz", "")

    pdb_out_path = "%s%s" % (pdb_dir, unzipped_pdb_filename)
    print('Unzipping %s to %s' % (pdb_path, pdb_out_path))

    gunzip(pdb_path, pdb_out_path)
    pdb2msa_results = pdb2msa(pdb_out_path, pdb_dir, create_new=False)

    if len(pdb2msa_results) > 1:                                                                         
        fasta_file = pdb2msa_results[0]
        prody_df = pdb2msa_results[1]
    else:                                                                                                
        prody_df = pdb2msa_results[0]



pdb2msa_row  = prody_df.iloc[0]
print('\n\nGetting msa with following pdb2msa entry:\n', pdb2msa_row)
#try:
print(pdb2msa_row)
pfam_id = pdb2msa_row['Pfam']
pdb_id = pdb2msa_row['PDB ID']
pdb_chain = pdb2msa_row['Chain']

ref_outfile = Path(processed_data_dir, '%s_ref.fa' % pfam_id)

pfam_dimensions_file = "%s/%s_%s_pfam_dimensions.npy" % (processed_data_dir, pdb_id, pfam_id)
pfam_dimensions = np.load(pfam_dimensions_file)
if len(pfam_dimensions)==7:
    [n_col, n_seq, m_eff, ct_ER, ct_MF, ct_PMF, ct_PLM] = pfam_dimensions
elif len(pfam_dimensions)==6: # new pfam_dimensions created in run_method_comparison. we dont need MF..
    [n_col, n_seq, m_eff, ct_ER, ct_PMF, ct_PLM] = pfam_dimensions
elif len(pfam_dimensions)==3:
    [n_col, n_seq, m_eff] = pfam_dimensions


if not os.path.exists('%s/%s_processed_data.npy' % (pdb_dir, pdb_id)):
    # we need mx.cumsum
    # also this needs to be rerunn since we deleted processed pdb2msa data.
    for ir, pdb2msa_row in enumerate(prody_df.iterrows()):
        print('\n\nGetting msa with following pdb2msa entry:\n', pdb2msa_row)
        try:
            dp_result =  data_processing_pdb2msa(data_path, prody_df.iloc[pdb2msa_row[0]], gap_seqs=0.2, gap_cols=0.2, prob_low=0.004,
                                   conserved_cols=0.8, printing=True, out_dir=processed_data_dir, pdb_dir=pdb_dir, letter_format=False,
                                   remove_cols=True, create_new=True)
            if dp_result is not None:
                [s0, removed_cols, s_index, tpdb, pdb_s_index] = dp_result
                break
            else:
                rows_to_drop.append(ir)
                continue
        except Exception as e:
            print('row %d got exception: ' % ir , e)
            print('moving on.. ')
            pass

    pdb_id = pdb2msa_row[1]['PDB ID']
    pfam_id = pdb2msa_row[1]['Pfam']
    # update Prody search DF (use same filename as pdb2msa() in data_processing 
    if not os.path.exists(pdb_data_file):
        prody_df = prody_df.drop(rows_to_drop)                                                     
    print("\nSaving updated Prody Search DataFrame:", prody_df.head())
    prody_df.to_csv('%s/%s_pdb_df.csv' % (pdb_dir, pdb_id))
    with open('%s/%s_processed_data.npy' % (pdb_dir, pdb_id), 'wb') as f:
        pickle.dump(dp_result, f)
    f.close()
else:
    with open('%s/%s_processed_data.npy' % (pdb_dir, pdb_id), 'rb') as f:
        dp_result = pickle.load(f)
    f.close()
    [s0, removed_cols, s_index, tpdb, pdb_s_index] = dp_result


# Load Expectation Reflection w
w_file = "%s/%s_%s_w.npy" % (processed_data_dir, pdb_id, pfam_id)
w_ER = np.load(w_file)
print(w_ER.shape)


# number of positions
n_var = s0.shape[1]
n_seq = s0.shape[0]

print("Number of residue positions:",n_var)
print("Number of sequences:",n_seq)

# number of aminoacids at each position
mx = np.array([len(np.unique(s0[:,i])) for i in range(n_var)])
#mx = np.array([m for i in range(n_var)])
print("Number of different amino acids at each position",mx)

mx_cumsum = np.insert(mx.cumsum(),0,0)
i1i2 = np.stack([mx_cumsum[:-1],mx_cumsum[1:]]).T
# print("(Sanity Check) Column indices of first and (",i1i2[0],") and last (",i1i2[-1],") positions")
# print("(Sanity Check) Column indices of second and (",i1i2[1],") and second to last (",i1i2[-2],") positions")


# number of variables
mx_sum = mx.sum()
print("Total number of variables",mx_sum)

# number of bias term
n_linear = mx_sum - n_var

onehot_encoder = OneHotEncoder(sparse=False,categories='auto')
# s is OneHot encoder format, s0 is original sequnce matrix
s = onehot_encoder.fit_transform(s0)






def energy_diff_row(i1i2, s, w, i):
    s_E_row = np.zeros(len(s))
    for j in range(i):
        s_E_row[j] = energy_diff(i1i2,s[i,:],s[j,:],w)   
    return s_E_row



res = Parallel(n_jobs = n_cpus-2)(delayed(energy_diff_row)\
        (i1i2, s_reordered, w_ER, i0)\
        for i0 in range(len(s_alt)))


s_E_dist = np.zeros((len(s),len(s)))
 in range(len(s)):
    for j in range(i):
        s_E_dist[i,j] = res[i][j]
s_E_dist += np.array(np.transpose(s_E_dist_temp))



E_dist_File = "%s/%s_%s_Edist.npy" % (processed_data_dir, pdb_id, pfam_id)
np.save(E_dist_file, s_E_dist)


