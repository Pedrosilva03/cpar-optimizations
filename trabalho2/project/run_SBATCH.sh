#!/bin/bash
#SBATCH --job-name=fluid_sim
#SBATCH --partition=cpar
#SBATCH --exclusive
#SBATCH --cpus-per-task=40
#SBATCH --time=00:10:00           
#SBATCH --output=fluid_sim_output.txt  

# Load any necessary modules (if required)
module load gcc/11.2.0

make par > /dev/null 2>&1

# run app
time ./fluid_sim

