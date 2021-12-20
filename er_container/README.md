# protein-emachine
# FOR FULLY UPDATED enviornment run in DCA_ER conda enviornment:
# $ conda list --explicit > spec-file.txt
# 		--> this is used in Dockerfile to generate conda enviornment for DCA_ER


#------------------------------------------------------#
#----- Generate Singularity Container to Run Code -----#
## In DCA_ER/er_images/ (location of Dockerfile)
### Make sure that you are not ssh or vpn connected
#	$ sudo docker build --no-cache --rm -t erdca-container .

## In DCA_ER/ (where the .simg files must be stored)
#	$ sudo docker tag <IMAGE ID> evancresswell/<REPO>:<TAG>
#	$ sudo docker push evancresswell/<REPO>:<TAG>

	#--- Singularity: Build .simg file from Docker ---#
	#	$ sudo singularity pull docker://evancresswell/#REPO#:#TAG#
	#	$ sudo singularity build dca_er-DCA.simg #REPO#_#TAG#.sif 
	#		ex: $ sudo singularity build erdca-muscle.simg erdca_muscle.sif 
	#-------------------------------------------------#

#------------------------------------------------------#
#------------------------------------------------------#
