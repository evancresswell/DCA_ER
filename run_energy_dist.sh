#! /bin/bash
# example run sbatch --mem=400g --time=1-00:00:00 --cpus-per-task=60 run_energy_dist.sh
source /data/cresswellclayec/conda/etc/profile.d/conda.sh
conda activate DCA_ER
python generate_energy_dist.py 1zdr $SLURM_CPUS_PER_TASK
