#!/bin/bash
#SBATCH --job-name=fluid_sim
#SBATCH --partition=day
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20        
#SBATCH --output=fluid_sim_output.txt  
#SBATCH --constraint=k20

# Add time limit if needed

# Load any necessary modules (if required)
module load gcc/7.2.0
module load cuda/11.3.1

make cuda > /dev/null 2>&1

# run app
srun --partition=cpar --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim_cuda



