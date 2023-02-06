#!/bin/bash

sbatch --dependency=afterok:jobid --cpus-per-task=2 --mem=10g --partition=gpu --gres=gpu:v100x:1 vp0_perp_part2.sh
sbatch --dependency=afterok:jobid --cpus-per-task=2 --mem=10g --partition=gpu --gres=gpu:v100x:1 vp1_perp_part2.sh
sbatch --dependency=afterok:jobid --cpus-per-task=2 --mem=10g --partition=gpu --gres=gpu:v100x:1 vp2_perp_part2.sh
sbatch --dependency=afterok:jobid --cpus-per-task=2 --mem=10g --partition=gpu --gres=gpu:v100x:1 vp3_perp_part2.sh
sbatch --dependency=afterok:jobid --cpus-per-task=2 --mem=10g --partition=gpu --gres=gpu:v100x:1 vp4_perp_part2.sh
sbatch --dependency=afterok:jobid --cpus-per-task=2 --mem=10g --partition=gpu --gres=gpu:v100x:1 vp5_perp_part2.sh
sbatch --dependency=afterok:jobid --cpus-per-task=2 --mem=10g --partition=gpu --gres=gpu:v100x:1 vp6_perp_part2.sh
sbatch --dependency=afterok:jobid --cpus-per-task=2 --mem=10g --partition=gpu --gres=gpu:v100x:1 vp7_perp_part2.sh
sbatch --dependency=afterok:jobid --cpus-per-task=2 --mem=10g --partition=gpu --gres=gpu:v100x:1 vp8_perp_part2.sh
sbatch --dependency=afterok:jobid --cpus-per-task=2 --mem=10g --partition=gpu --gres=gpu:v100x:1 vp9_perp_part2.sh
sbatch --dependency=afterok:jobid --cpus-per-task=2 --mem=10g --partition=gpu --gres=gpu:v100x:1 vp10_perp_part2.sh
sbatch --dependency=afterok:jobid --cpus-per-task=2 --mem=10g --partition=gpu --gres=gpu:v100x:1 vp11_perp_part2.sh
sbatch --dependency=afterok:jobid --cpus-per-task=2 --mem=10g --partition=gpu --gres=gpu:v100x:1 vp12_perp_part2.sh
sbatch --dependency=afterok:jobid --cpus-per-task=2 --mem=10g --partition=gpu --gres=gpu:v100x:1 vp13_perp_part2.sh
