#!/bin/bash
#SBATCH --job-name=testing
#SBATCH --partition=cpar
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20        
#SBATCH --output=tests.txt  
#SBATCH --constraint=k20

# Add time limit if needed

# Load any necessary modules (if required)
module load gcc/7.2.0
module load cuda/11.3.1
module load python

make cuda > /dev/null 2>&1

# run tests
make tests