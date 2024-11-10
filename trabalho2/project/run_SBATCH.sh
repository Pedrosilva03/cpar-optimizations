#!/bin/bash
#SBATCH --job-name=fluid_sim
#SBATCH --partition=cpar
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40         
#SBATCH --output=fluid_sim_output.txt  

# Load any necessary modules (if required)
module load gcc/11.2.0

make par > /dev/null 2>&1

# run app
./fluid_sim

echo "Starting Job $SLURM_ARRAY_TASK_ID"


