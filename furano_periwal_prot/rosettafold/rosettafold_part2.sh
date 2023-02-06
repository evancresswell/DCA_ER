#!/bin/bash

# submit with
# sbatch --dependency=afterok:jobid --cpus-per-task=2 --mem=10g --partition=gpu --gres=gpu:v100x:1 rosettafold_part2.sh
 
module load RoseTTAFold
run_e2e_ver_part2.sh vp0.fasta e2e_out_vp0
run_e2e_ver_part2.sh vp1.fasta e2e_out_vp1
run_e2e_ver_part2.sh vp2.fasta e2e_out_vp2
run_e2e_ver_part2.sh vp3.fasta e2e_out_vp3
run_e2e_ver_part2.sh vp4.fasta e2e_out_vp4
run_e2e_ver_part2.sh vp5.fasta e2e_out_vp5
run_e2e_ver_part2.sh vp6.fasta e2e_out_vp6
run_e2e_ver_part2.sh vp7.fasta e2e_out_vp7
run_e2e_ver_part2.sh vp8.fasta e2e_out_vp8
run_e2e_ver_part2.sh vp9.fasta e2e_out_vp9
run_e2e_ver_part2.sh vp10.fasta e2e_out_vp10
run_e2e_ver_part2.sh vp11.fasta e2e_out_vp11
run_e2e_ver_part2.sh vp12.fasta e2e_out_vp12
run_e2e_ver_part2.sh vp13.fasta e2e_out_vp13

run_pyrosetta_ver_part2.sh vp0.fast pyrosetta_out_vp0
run_pyrosetta_ver_part2.sh vp1.fast pyrosetta_out_vp1
run_pyrosetta_ver_part2.sh vp2.fast pyrosetta_out_vp2
run_pyrosetta_ver_part2.sh vp3.fast pyrosetta_out_vp3
run_pyrosetta_ver_part2.sh vp4.fast pyrosetta_out_vp4
run_pyrosetta_ver_part2.sh vp5.fast pyrosetta_out_vp5
run_pyrosetta_ver_part2.sh vp6.fast pyrosetta_out_vp6
run_pyrosetta_ver_part2.sh vp7.fast pyrosetta_out_vp7
run_pyrosetta_ver_part2.sh vp8.fast pyrosetta_out_vp8
run_pyrosetta_ver_part2.sh vp9.fast pyrosetta_out_vp9
run_pyrosetta_ver_part2.sh vp10.fast pyrosetta_out_vp10
run_pyrosetta_ver_part2.sh vp11.fast pyrosetta_out_vp11
run_pyrosetta_ver_part2.sh vp12.fast pyrosetta_out_vp12
run_pyrosetta_ver_part2.sh vp13.fast pyrosetta_out_vp13
