# protein-emachine
# github token: 4de5804a0bb1014bfacfc2466187741403f41b6c

#------------------------------------------------------#
#----- Generate Singularity Container to Run Code -----#
## In DCA_ER/er_images/ (location of Dockerfile)
### Make sure that you are not ssh or vpn connected
#	$ sudo docker build --no-cache --rm -t erdca-muscle .

## In DCA_ER/ (where the .simg files must be stored)
#	$ sudo docker tag <IMAGE ID> evancresswell/<REPO>:<TAG>
#	$ sudo docker push evancresswell/<REPO>:<TAG>

	#--- Singularity: Build .simg file from Docker ---#
	#	$ sudo singularity pull docker://evancresswell/#REPO#:#TAG#
	#	$ sudo singularity build dca_er-DCA.simg #REPO#_#TAG#.sif 
	#		ex: $ sudo singularity build erdca-muscle.simg erdca_muscle.sif 
	#-------------------------------------------------#

#------------------------------------------------------#
#----- Run Jupyter notebook throught Singularity ------#

# start intervative session with sufficient mem/cpu and tunneling from compute node to biowulf login
sinteractive --mem=50g --cpus-per-task=32 --tunnel

# create tunnel from biowulf login to local computer
ex: ssh  -L 41119:localhost:41119 cresswellclayec@biowulf.nih.gov

# run singularity shell with anaconda envionrment
singularity shell --bind /data/cresswellclayec/DCA_ER /path/to/<pydca-container>.simg 

# activate anaconda in singularity shell
source ~/.bashrc

# activate enviornment
conda activate pydca

# run notebook
jupyter notebook --ip localhost --port $PORT1 --no-browser
#------------------------------------------------------#
#------------------------------------------------------#
