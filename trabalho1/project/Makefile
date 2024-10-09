CPP = g++ -Wall -Wextra -lm -pg # -fexcess-precision=standard -ffloat-store

# Optimization Flags
CFLAG_DEFAULT = -O2 -ffast-math
CFLAG_O2 = -O2 -ffast-math -ftree-vectorize -funroll-loops -march=native 
CFLAG_O3 = -O3 -ffast-math -ftree-vectorize -funroll-loops -march=native
CFLAG_OFAST = -Ofast -ffast-math -ftree-vectorize -funroll-loops -march=native

# Vectorization and Loop Optimization Flags
CFLAG_MSSE = -msse4
CFLAG_DEBUG = -g

SRCS = main.cpp fluid_solver.cpp EventManager.cpp
OUTPUT = fluid_sim

# Main target
all: $(OUTPUT)

$(OUTPUT): $(SRCS)
	$(CPP) $(CFLAG_DEFAULT) $(SRCS) -o $(OUTPUT)  # Default compilation without optimizations or profiling

# Specific optimization targets
o2: $(SRCS)
	$(CPP) $(CFLAG_O2) $(SRCS) -o $(OUTPUT)

o3: $(SRCS)
	$(CPP) $(CFLAG_O3) $(SRCS) -o $(OUTPUT)

ofast: $(SRCS)
	$(CPP) $(CFLAG_OFAST) $(SRCS) -o $(OUTPUT)

# Run the program
run: $(OUTPUT)
	./$(OUTPUT)

# Profile the program
profile: run
	gprof $(OUTPUT) gmon.out | gprof2dot | dot -Tpng -o callgraph.png
	@echo Profile report generated: callgraph.png

# Clean up build artifacts
clean:
	@echo Cleaning up...
	@rm -f $(OUTPUT) gmon.out callgraph.png
	@echo Done.