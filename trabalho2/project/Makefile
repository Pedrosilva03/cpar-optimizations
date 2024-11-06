CPP = g++ -Wall -Wextra -lm -pg 
SRCS_PAR = main.cpp fluid_solver.cpp EventManager.cpp
SRCS_SEQ = main.cpp fluid_solver_seq.cpp EventManager.cpp
OUTPUT = fluid_sim


# Flags de otimização
CFLAGS = -Wall -Wextra -lm -lstdc++ -O3 -Ofast -ffast-math -ftree-vectorize -march=native -mavx -funroll-loops -flto -fopt-info
CFLAG_PAR = -fopenmp -msse4.1

all: par seq

par:
	$(CPP) $(CFLAGS) $(CFLAG_PAR) $ $(SRCS_PAR) -o $(OUTPUT)

seq:
	$(CPP) $(CFLAGS) $ $(SRCS_SEQ) -o fluid_sim_seq

# Run the program
runpar: $(OUTPUT)
	export OMP_NUM_THREADS=`grep 'processor' /proc/cpuinfo | wc -l `;\
	./$(OUTPUT)

runseq: fluid_sim_seq
	./fluid_sim_seq

# Profile the program
profile_par: runpar
	gprof $(OUTPUT) gmon.out | gprof2dot | dot -Tpng -o callgraph.png
	@echo Profile report generated: callgraph.png

profile_seq: runseq
	gprof fluid_sim_seq gmon.out | gprof2dot | dot -Tpng -o callgraph_seq.png
	@echo Profile report generated: callgraph_seq.png

# Limpeza
clean:
	@echo Cleaning up...
	@rm -f $(OUTPUT) fluid_sim_seq gmon.out callgraph.png callgraph_seq.png 
	@echo Done.