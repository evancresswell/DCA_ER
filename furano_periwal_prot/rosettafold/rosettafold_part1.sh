#!/bin/bash

# submit with
# sbatch --cpus-per-task=10 --mem=60G --time=1-00:00:00 --partition=norm rosettafold_part1.sh
 
module load RoseTTAFold
# run_e2e_ver_part1.sh vp0.fasta e2e_out_vp0
# run_e2e_ver_part1.sh vp1.fasta e2e_out_vp1
# run_e2e_ver_part1.sh vp2.fasta e2e_out_vp2
# run_e2e_ver_part1.sh vp3.fasta e2e_out_vp3
# run_e2e_ver_part1.sh vp4.fasta e2e_out_vp4
# run_e2e_ver_part1.sh vp5.fasta e2e_out_vp5
# run_e2e_ver_part1.sh vp6.fasta e2e_out_vp6
# run_e2e_ver_part1.sh vp7.fasta e2e_out_vp7
# run_e2e_ver_part1.sh vp8.fasta e2e_out_vp8
# run_e2e_ver_part1.sh vp9.fasta e2e_out_vp9
# run_e2e_ver_part1.sh vp10.fasta e2e_out_vp10
# run_e2e_ver_part1.sh vp11.fasta e2e_out_vp11
# run_e2e_ver_part1.sh vp12.fasta e2e_out_vp12
# run_e2e_ver_part1.sh vp13.fasta e2e_out_vp13

run_pyrosetta_ver_part1.sh vp0.fasta pyrosetta_out_vp0
run_pyrosetta_ver_part1.sh vp1.fasta pyrosetta_out_vp1
run_pyrosetta_ver_part1.sh vp2.fasta pyrosetta_out_vp2
run_pyrosetta_ver_part1.sh vp3.fasta pyrosetta_out_vp3
run_pyrosetta_ver_part1.sh vp4.fasta pyrosetta_out_vp4
run_pyrosetta_ver_part1.sh vp5.fasta pyrosetta_out_vp5
run_pyrosetta_ver_part1.sh vp6.fasta pyrosetta_out_vp6
run_pyrosetta_ver_part1.sh vp7.fasta pyrosetta_out_vp7
run_pyrosetta_ver_part1.sh vp8.fasta pyrosetta_out_vp8
run_pyrosetta_ver_part1.sh vp9.fasta pyrosetta_out_vp9
run_pyrosetta_ver_part1.sh vp10.fasta pyrosetta_out_vp10
run_pyrosetta_ver_part1.sh vp11.fasta pyrosetta_out_vp11
run_pyrosetta_ver_part1.sh vp12.fasta pyrosetta_out_vp12
run_pyrosetta_ver_part1.sh vp13.fasta pyrosetta_out_vp13
