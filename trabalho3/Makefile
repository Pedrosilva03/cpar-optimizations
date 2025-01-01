CPP = g++ -Wall -Wextra -lm
SRCS_PAR = project/main.cpp project/fluid_solver.cpp project/EventManager.cpp
SRCS_SEQ = project/main.cpp project/fluid_solver_seq.cpp project/EventManager.cpp
SRCS_CUDA = project/main.cpp project/fluid_solver_cuda.cu project/EventManager.cpp
OUTPUT = fluid_sim
NVCC = nvcc

# Flags de otimização
CFLAGS = -march=native -funroll-loops -ffast-math -fopt-info -Ofast -ftree-vectorize -mavx -pg
CFLAG_PAR = -fopenmp -msse4.1
CUDAFLAGS = -O3 -g -use_fast_math -std=c++11 -arch=sm_35 -Wno-deprecated-gpu-targets

OMP_NUM_THREADS ?= 40
export OMP_NUM_THREADS

all: par seq cuda

par:
	$(CPP) $(CFLAGS) $(CFLAG_PAR) $(SRCS_PAR) -o $(OUTPUT)

seq:
	$(CPP) $(CFLAGS) $(SRCS_SEQ) -o fluid_sim_seq

cuda:
	$(NVCC) $(CUDAFLAGS) $(SRCS_CUDA) -o fluid_sim_cuda

# Run the program
runpar: $(OUTPUT)
	./$(OUTPUT)

runseq: fluid_sim_seq
	./fluid_sim_seq

runcuda: fluid_sim_cuda
	./fluid_sim_cuda

# Profile the program
profile_par: runpar
	gprof $(OUTPUT) gmon.out | gprof2dot | dot -Tpng -o callgraph.png
	@echo Profile report generated: callgraph.png

profile_seq: runseq
	gprof fluid_sim_seq gmon.out | gprof2dot | dot -Tpng -o callgraph_seq.png
	@echo Profile report generated: callgraph_seq.png

profile_cuda:
	nvprof --print-gpu-trace ./fluid_sim_cuda
	@echo Profile report generated for nvprof

# Limpeza
clean:
	@echo Cleaning up...
	@rm -f $(OUTPUT) fluid_sim_seq fluid_sim_cuda gmon.out callgraph.png callgraph_seq.png fluid_sim_output.txt profile.txt
	@echo Done.
