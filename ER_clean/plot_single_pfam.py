import os,sys
import datetime
import numpy as np
on_pc = False
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import pickle

from direct_info import sort_di

import ecc_tools as tools
import data_processing as dp

# import inference_dca for mfDCA
from inference_dca import direct_info_dca

"""
NOTES:
PYDCA was altered for these plots. These alterations are local to hurricane
"""
# import pydca for plmDCA
from pydca.plmdca import plmdca
from pydca.meanfield_dca import meanfield_dca
from pydca.sequence_backmapper import sequence_backmapper
from pydca.msa_trimmer import msa_trimmer
from pydca.contact_visualizer import contact_visualizer
from pydca.dca_utilities import dca_utilities

data_path = '../../Pfam-A.full'
data_path = '/data/cresswellclayec/hoangd2_data/Pfam-A.full'

er_directory = './DI/ER/'
mf_directory = './DI/MF/'
plm_directory = './DI/PLM/'


pfam_id = sys.argv[1]

# Get number of data files
#num_files = len([name for name in os.listdir(er_directory) if name.endswith(".pickle")])
#print("Plotting analysis for %d Proteins"% num_files)


print ('Plotting Protein Famility ', pfam_id)
# Load PDB structure 
pdb = np.load('%s/%s/pdb_refs.npy'%(data_path,pfam_id))

#---------- Pre-Process Structure Data ----------------#
# delete 'b' in front of letters (python 2 --> python 3)
pdb = np.array([pdb[t,i].decode('UTF-8') for t in range(pdb.shape[0]) \
for i in range(pdb.shape[1])]).reshape(pdb.shape[0],pdb.shape[1])

# data processing THESE SHOULD BE CREATED DURING DATA GENERATION
ipdb = 0
print(len(pdb))
# Loop Through all pdb structures in Pfam
#for ipdb,pdb_entry in enumerate(pdb):
#	pdb_id = pdb_entry[5]
#	print(pdb)
#	print(pdb_id)

input_data_file = "pfam_ecc/%s_DP.pickle"%(pfam_id)
with open(input_data_file,"rb") as f:
	pfam_dict = pickle.load(f)
f.close()
#s0,cols_removed,s_index,s_ipdb = dp.data_processing(data_path,pfam_id,ipdb,\
#				gap_seqs=0.2,gap_cols=0.2,prob_low=0.004,conserved_cols=0.9)
s0 = pfam_dict['s0']	
s_index = pfam_dict['s_index']	
s_ipdb = pfam_dict['s_ipdb']	
cols_removed = pfam_dict['cols_removed']

# number of positions
n_var = s0.shape[1]

# Save processed data
msa_outfile, ref_outfile = dp.write_FASTA(s0,pfam_id,s_ipdb)
#-------------------------------------------------------#

# Plot Contact Map
ct = tools.contact_map(pdb,ipdb,cols_removed,s_index)
ct_distal = tools.distance_restr(ct,s_index,make_large=True)

#---------------------- Load DI -------------------------------------#
print("Unpickling DI pickle files for %s"%(pfam_id))
with open("%ser_DI_%s.pickle"%(er_directory,pfam_id),"rb") as f:
	DI_er = pickle.load(f)
f.close()
with open("%ser_cov_couplings_DI_%s.pickle"%(er_directory,pfam_id),"rb") as f:
	DI_covER = pickle.load(f)
f.close()
with open("%ser_couplings_DI_%s.pickle"%(er_directory,pfam_id),"rb") as f:
	DI_coupER = pickle.load(f)
f.close()
with open("%splm_DI_%s.pickle"%(plm_directory,pfam_id),"rb") as f:
	DI_plm = pickle.load(f)
f.close()
with open("%smf_DI_%s.pickle"%(mf_directory,pfam_id),"rb") as f:
	DI_mf = pickle.load(f)
f.close()

#DI_er = pickle.load(open("%ser_DI_%s.pickle"%(er_directory,pfam_id),"rb"))
#DI_mf = pickle.load(open("%smf_DI_%s.pickle"%(mf_directory,pfam_id),"rb"))
#DI_plm = pickle.load(open("%splm_DI_%s.pickle"%(plm_directory,pfam_id),"rb"))

DI_er_dup = dp.delete_sorted_DI_duplicates(DI_er)	
DI_cov_er_dup = dp.delete_sorted_DI_duplicates(DI_covER)	
DI_coup_er_dup = dp.delete_sorted_DI_duplicates(DI_coupER)	
DI_plm_dup = dp.delete_sorted_DI_duplicates(DI_plm)	
DI_mf_dup = dp.delete_sorted_DI_duplicates(DI_mf)	

sorted_DI_er = tools.distance_restr_sortedDI(DI_er_dup)
sorted_DI_cov_er = tools.distance_restr_sortedDI(DI_cov_er_dup)
sorted_DI_coup_er = tools.distance_restr_sortedDI(DI_coup_er_dup)
sorted_DI_plm = tools.distance_restr_sortedDI(DI_plm_dup)
sorted_DI_mf = tools.distance_restr_sortedDI(DI_mf_dup)

print("\nPrint top 10 Non-Redundant pairs")
for x in sorted_DI_er[:10]:
	print(x)
#--------------------------------------------------------------------#

#--------------------- Load DI Matrix -------------------------------#
distance_enforced = False
if distance_enforced:
	for coupling in DI_er_dup:
		di_er[coupling[0][0],coupling[0][1]] = coupling[1]
		di_er[coupling[0][1],coupling[0][0]] = coupling[1]
	for coupling in DI_mf_dup:
		di_mf[coupling[0][0],coupling[0][1]] = coupling[1]
		di_mf[coupling[0][1],coupling[0][0]] = coupling[1]
	for coupling in DI_plm_dup:
		di_plm[coupling[0][0],coupling[0][1]] = coupling[1]
		di_plm[coupling[0][1],coupling[0][0]] = coupling[1]
else:
	n_seq = max([coupling[0][0] for coupling in sorted_DI_er]) 
	di_er = np.zeros((n_var,n_var))
	di_cov_er = np.zeros((n_var,n_var))
	di_coup_er = np.zeros((n_var,n_var))
	di_mf = np.zeros((n_var,n_var))
	di_plm = np.zeros((n_var,n_var))
	for coupling in DI_er_dup:
		#print(coupling[1])
		di_er[coupling[0][0],coupling[0][1]] = coupling[1]
		di_er[coupling[0][1],coupling[0][0]] = coupling[1]
	for coupling in DI_cov_er_dup:
		#print(coupling[1])
		di_cov_er[coupling[0][0],coupling[0][1]] = coupling[1]
		di_cov_er[coupling[0][1],coupling[0][0]] = coupling[1]
	for coupling in DI_coup_er_dup:
		#print(coupling[1])
		di_coup_er[coupling[0][0],coupling[0][1]] = coupling[1]
		di_coup_er[coupling[0][1],coupling[0][0]] = coupling[1]
	for coupling in DI_mf_dup:
		di_mf[coupling[0][0],coupling[0][1]] = coupling[1]
		di_mf[coupling[0][1],coupling[0][0]] = coupling[1]
	for coupling in DI_plm_dup:
		di_plm[coupling[0][0],coupling[0][1]] = coupling[1]
		di_plm[coupling[0][1],coupling[0][0]] = coupling[1]

#--------------------------------------------------------------------#


#----------------- Generate Optimal ROC Curve -----------------------#
# find optimal threshold of distance for both DCA and ER
ct_thres = np.linspace(1.5,10.,18,endpoint=True)
n = ct_thres.shape[0]

auc_mf = np.zeros(n)
auc_er = np.zeros(n)
auc_cov_er = np.zeros(n)
auc_coup_er = np.zeros(n)
auc_plm = np.zeros(n)

for i in range(n):
	p,tp,fp = tools.roc_curve(ct_distal,di_mf,ct_thres[i])
	auc_mf[i] = tp.sum()/tp.shape[0]
	
	################################3 need to update singularity container p,tp,fp = tools.roc_curve(ct_distal,di_er,ct_thres[i])
	p,tp,fp = tools.roc_curve(ct_distal,di_er,ct_thres[i])
	auc_er[i] = tp.sum()/tp.shape[0]

	p,tp,fp = tools.roc_curve(ct_distal,di_cov_er,ct_thres[i])
	auc_cov_er[i] = tp.sum()/tp.shape[0]
	
	p,tp,fp = tools.roc_curve(ct_distal,di_coup_er,ct_thres[i])
	auc_coup_er[i] = tp.sum()/tp.shape[0]
	
	p,tp,fp = tools.roc_curve(ct_distal,di_plm,ct_thres[i])
	auc_plm[i] = tp.sum()/tp.shape[0]

i0_mf = np.argmax(auc_mf)
i0_er = np.argmax(auc_er)
i0_coup_er = np.argmax(auc_coup_er)
i0_cov_er = np.argmax(auc_cov_er)
i0_plm = np.argmax(auc_plm)


p0_mf,tp0_mf,fp0_mf = tools.roc_curve(ct_distal,di_mf,ct_thres[i0_mf])
##################################### need to update singularity container   p0_er,tp0_er,fp0_er = tools.roc_curve(ct_distal,di_er,ct_thres[i0_er])
p0_er,tp0_er,fp0_er = tools.roc_curve(ct_distal,di_er,ct_thres[i0_er])
p0_cov_er,tp0_cov_er,fp0_cov_er = tools.roc_curve(ct_distal,di_cov_er,ct_thres[i0_cov_er])
p0_coup_er,tp0_coup_er,fp0_coup_er = tools.roc_curve(ct_distal,di_coup_er,ct_thres[i0_coup_er])
p0_plm,tp0_plm,fp0_plm = tools.roc_curve(ct_distal,di_plm,ct_thres[i0_plm])

#------------------ Plot ROC for optimal DCA vs optimal ER ------------------#
#print("Optimal Contact threshold for (mf, er, plm) = (%f, %f, %f)"%(ct_thres[i0_mf],ct_thres[i0_er],ct_thres[i0_plm]))
#print("Maximal AUC for (mf, er, plm) = (%f, %f, %f)"%(auc_mf[i0_mf], auc_er[i0_er], auc_plm[i0_plm]))

with PdfPages("./Pfam_Plots/%s.pdf"%pfam_id) as pdf:

	#---------------------- Plot Contact Map ----------------------------#

	#plt.title('Contact Map')
	plt.imshow(ct_distal,cmap='rainbow_r',origin='lower')
	plt.xlabel('i')
	plt.ylabel('j')
	plt.title(pfam_id)
	plt.colorbar(fraction=0.045, pad=0.05)
	pdf.attach_note("Contact Map")  # you can add a pdf note to
	pdf.savefig()  # saves the current figure into a pdf page
	plt.close()
	#--------------------------------------------------------------------#

	plt.subplot2grid((1,3),(0,0))
	plt.title('ROC ')
	plt.plot(fp0_er,tp0_er,'b-',label="er")
	plt.plot(fp0_coup_er,tp0_coup_er,'y-',label="coup_er")
	plt.plot(fp0_cov_er,tp0_cov_er,'m-',label="cov_er")
	plt.plot(fp0_mf,tp0_mf,'r-',label="mf")
	plt.plot(fp0_plm,tp0_plm,'g-',label="plm")
	plt.plot([0,1],[0,1],'k--')
	plt.xlim([0,1])
	plt.ylim([0,1])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.legend()

	# Plot AUC for DCA and ER
	plt.subplot2grid((1,3),(0,1))
	plt.title('AUC')
	plt.plot([ct_thres.min(),ct_thres.max()],[0.5,0.5],'k--')
	plt.plot(ct_thres,auc_er,'b-',label="er")
	plt.plot(ct_thres,auc_coup_er,'y-',label="coup_er")
	plt.plot(ct_thres,auc_cov_er,'m-',label="cov_er")
	plt.plot(ct_thres,auc_mf,'r-',label="mf")
	plt.plot(ct_thres,auc_plm,'g-',label="plm")
	print(auc_er)
	print(auc_mf)
	print(auc_plm)
	plt.ylim([min(auc_cov_er.min(),auc_coup_er.min(),auc_er.min(),auc_mf.min(),auc_plm.min())-0.05,max(auc_cov_er.max(),auc_coup_er.max(),auc_er.max(),auc_mf.max(),auc_plm.max())+0.05])
	plt.xlim([ct_thres.min(),ct_thres.max()])
	plt.xlabel('distance threshold')
	plt.ylabel('AUC')
	plt.legend()

	# Plot Precision of optimal DCA and ER
	plt.subplot2grid((1,3),(0,2))
	plt.title('Precision')
	plt.plot( p0_er,tp0_er / (tp0_er + fp0_er),'b-',label='er')
	plt.plot( p0_coup_er,tp0_coup_er / (tp0_coup_er + fp0_coup_er),'y-',label='coup_er')
	plt.plot( p0_cov_er,tp0_cov_er / (tp0_cov_er + fp0_cov_er),'m-',label='cov_er')
	plt.plot( p0_mf,tp0_mf / (tp0_mf + fp0_mf),'r-',label='mf')
	plt.plot( p0_plm,tp0_plm / (tp0_plm + fp0_plm),'g-',label='plm')
	plt.xlim([0,1])
	plt.ylim([0,1])
	plt.ylim([.4,.8])
	plt.xlabel('Recall (Sensitivity - P)')
	plt.ylabel('Precision (PPV)')
	plt.legend()

	plt.tight_layout(h_pad=.25, w_pad=.1)
	pdf.attach_note("ROC")  # you can add a pdf note to
	pdf.savefig()  # saves the current figure into a pdf page
	plt.close()
	#----------------------------------------------------------------------------#

	#----------------------------------------------------------------------------#
	# Using PYDCA contact mapping module
	print("Dimensions of DI Pairs:")
	print("ER: ",len(sorted_DI_er))
	##print("PLM: ",len(sorted_DI_plm))
	####print("MF: ",len(sorted_DI_mf))


	#""" NEED MF and PLM FIRST
	cov_erdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb[ipdb,6], pdb[ipdb,5],
		refseq_file = ref_outfile,
		sorted_dca_scores = sorted_DI_cov_er,
		linear_dist = 4,
		contact_dist = 8.0,
	)
	coup_erdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb[ipdb,6], pdb[ipdb,5],
		refseq_file = ref_outfile,
		sorted_dca_scores = sorted_DI_coup_er,
		linear_dist = 4,
		contact_dist = 8.0,
	)
	erdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb[ipdb,6], pdb[ipdb,5],
		refseq_file = ref_outfile,
		sorted_dca_scores = sorted_DI_er,
		linear_dist = 4,
		contact_dist = 8.0,
	)
	mfdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb[ipdb,6], pdb[ipdb,5],
		refseq_file = ref_outfile,
		sorted_dca_scores = sorted_DI_mf,
		linear_dist = 4,
		contact_dist = 8.0,
	)

	plmdca_visualizer = contact_visualizer.DCAVisualizer('protein', pdb[ipdb,6], pdb[ipdb,5],
		refseq_file = ref_outfile,
		sorted_dca_scores = sorted_DI_plm,
		linear_dist = 4,
		contact_dist = 8.0,
	)

	#"""
	# Define a list of contact visualizers to plot 3 methods
	slick_contact_maps = [ cov_erdca_visualizer, coup_erdca_visualizer, erdca_visualizer, mfdca_visualizer, plmdca_visualizer]
	slick_titles = [ 'ER', 'MF', 'PLM']

	# Create subplots
	fig, axes = plt.subplots(nrows=1,ncols=len(slick_contact_maps), sharex='all',figsize=(15,5))

	# Plot
	for i,slick_map in enumerate(slick_contact_maps):
		contact_map_data = slick_map.plot_contact_map(axes[i])
		axes[i].set_title(slick_titles[i]+'\n'+axes[i].get_title())

	pdf.savefig()  # saves the current figure into a pdf page
	plt.close()

	#er_tp_rate_data, er_tp_ax = erdca_visualizer.plot_true_positive_rates()
	#mf_tp_rate_data, mf_tp_ax = mfdca_visualizer.plot_true_positive_rates()
	#plm_tp_rate_data, plm_tp_ax = plmdca_visualizer.plot_true_positive_rates()
	#pdf.attach_note("True Positive Rate")  # you can add a pdf note to
	#pdf.savefig()  # saves the current figure into a pdf page


	#----------------------------------------------------------------------------#
	# We can also set the file's metadata via the PdfPages object:
	d = pdf.infodict()
	d['Title'] = 'Pfam: %s'%(pfam_id)
	d['Author'] = 'Evan Cresswell\xe4nen'
	d['Subject'] = 'Contact inference of Pfam Proteins'
	d['Keywords'] = 'Pfam Contact Map PDB Expectation Reflection Mean-Field DCA Pseudoliklihood'
	d['CreationDate'] = datetime.datetime.today()

