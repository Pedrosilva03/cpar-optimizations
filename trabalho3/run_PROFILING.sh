#!/bin/bash
#SBATCH --job-name=nvprof
#SBATCH --partition=cpar
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20        
#SBATCH --output=profile.txt  
#SBATCH --constraint=k20

# Add time limit if needed

# Load any necessary modules (if required)
module load gcc/7.2.0
module load cuda/11.3.1

make cuda > /dev/null 2>&1

# run profiling
make profile_cuda