#!/bin/bash
#SBATCH --job-name=fluid_sim_par
#SBATCH --partition=cpar
#SBATCH --exclusive
#SBATCH --cpus-per-task=$(grep 'processor' /proc/cpuinfo | wc -l) 
#SBATCH --time=00:10:00           
#SBATCH --output=fluid_sim_output.txt  

# Load any necessary modules (if required)
module load gcc/11.2.0

# Set OMP_NUM_THREADS to the number of CPUs per task allocated by Slurm
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the application with `perf` for performance metrics
srun perf stat -r 3 -M cpi,instructions \
     -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores \
     ./fluid_sim
