CPP = g++ -Wall -Wextra -lm
SRCS = src/main.cpp src/fluid_solver.cpp src/EventManager.cpp
OUTPUT = fluid_sim

# Flags de otimização
CFLAGS = -march=native -funroll-loops -ffast-math -fopt-info -Ofast -ftree-vectorize -mavx -pg
CFLAG_PAR = -fopenmp -msse4.1

OMP_NUM_THREADS ?= 40
export OMP_NUM_THREADS

all: compile

compile:
	$(CPP) $(CFLAGS) $(CFLAG_PAR) $(SRCS) -o $(OUTPUT)

# Run the program
run: $(OUTPUT)
	./$(OUTPUT)

# Profile the program
profile: run
	gprof $(OUTPUT) gmon.out | gprof2dot | dot -Tpng -o callgraph.png
	@echo Profile report generated: callgraph.png

# Limpeza
clean:
	@echo Cleaning up...
	@rm -f $(OUTPUT) gmon.out callgraph.png fluid_sim_output.txt  
	@echo Done.
