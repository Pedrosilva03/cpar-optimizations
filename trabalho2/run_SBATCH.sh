#!/bin/bash
#SBATCH --job-name=fluid_sim
#SBATCH --partition=cpar
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00         
#SBATCH --output=fluid_sim_output.txt  

# Load any necessary modules (if required)
module load gcc/11.2.0

make par > /dev/null 2>&1

# run app
srun --partition=cpar --exclusive perf stat -r 3 -M cpi,instructions -e branch-misses,L1-dcache-loads,L1-dcache-load-misses,cycles,duration_time,mem-loads,mem-stores ./fluid_sim

export OMP_NUM_THREADS=1
echo "OMP_NUM_THREADS=1 below"
time ./fluid_sim
export OMP_NUM_THREADS=2
echo "OMP_NUM_THREADS=2 below"
time ./fluid_sim
export OMP_NUM_THREADS=4
echo "OMP_NUM_THREADS=4 below"
time ./fluid_sim
export OMP_NUM_THREADS=8
echo "OMP_NUM_THREADS=8 below"
time ./fluid_sim
export OMP_NUM_THREADS=16
echo "OMP_NUM_THREADS=16 below"
time ./fluid_sim
export OMP_NUM_THREADS=32
echo "OMP_NUM_THREADS=32 below"
time ./fluid_sim
export OMP_NUM_THREADS=40
echo "OMP_NUM_THREADS=40 below"
time ./fluid_sim



