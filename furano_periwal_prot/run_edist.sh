#! /bin/bash

# example run
# sbatch --mem=400g --time=1-00:00:00 --partition=largemem --cpus-per-task=50 run_edist.sh
source /data/cresswellclayec/conda/etc/profile.d/conda.sh
conda activate DCA_ER
# python run_edist.py $SLURM_CPUS_PER_TASK
python run_edist_family.py $SLURM_CPUS_PER_TASK
