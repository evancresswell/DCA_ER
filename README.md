# DCA_ER
Implementation of DCA using Expectation of ER

#----- Run Jupyter notebook throught Singularity ------#

# start intervative session with sufficient mem/cpu and tunneling from compute node to biowulf login
sinteractive --mem=50g --cpus-per-task=32 --tunnel

# create tunnel from biowulf login to local computer
ex: ssh  -L 41119:localhost:41119 cresswellclayec@biowulf.nih.gov
# # where 41119 is the port given when requesting the ssh tunner (--tunnel options)

# run singularity shell with anaconda envionrment
singularity shell --bind /data/cresswellclayec/DCA_ER /path/to/<erdca-container>.simg 

# activate anaconda in singularity shell
source /opt/conda/etc/profile.d/conda.sh

# activate enviornment
conda activate dca_er  
# or whatever the enviornment in <erdca-container>.simg with the required modules is called

# run notebook
jupyter notebook --ip localhost --port $PORT1 --no-browser
# Rock and Roll..
#------------------------------------------------------#


