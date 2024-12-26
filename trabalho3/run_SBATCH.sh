#!/bin/bash
#SBATCH --job-name=fluid_sim
#SBATCH --partition=day
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20        
#SBATCH --output=fluid_sim_output.txt  

# Load any necessary modules (if required)
module load gcc/7.2.0
module load cuda/11.3.1

make cuda > /dev/null 2>&1

# run app
srun --partition=day --exclusive ./fluid_sim_cuda



