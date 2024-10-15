CPP = g++ -Wall -Wextra -lm -pg 

# Optimization Flags
CFLAG_DEFAULT = -O2 -ffast-math
CFLAG_VEC = -ftree-vectorize -march=native
CFLAG_LOOP = -funroll-loops
CFLAG_O3 = -O3 -ffast-math 
CFLAG_OFAST = -Ofast -ffast-math 
CFLAG_ALIGN = -falign-functions=16 -falign-loops=16
CFLAG_LTO = -flto

# Vectorization and Loop Optimization Flags
CFLAG_MSSE = -msse4
CFLAG_DEBUG = -g

SRCS = main.cpp fluid_solver.cpp EventManager.cpp
OUTPUT = fluid_sim

# Main target
all: $(OUTPUT)

$(OUTPUT): $(SRCS)
	$(CPP) $(CFLAG_DEFAULT) $(SRCS) -o $(OUTPUT)  # Default compilation without optimizations or profiling

# O2 flags
o2_vec: $(SRCS)
	$(CPP) $(CFLAG_O2) $(CFLAG_VEC) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)

o2_loop: $(SRCS)
	$(CPP) $(CFLAG_O2) $(CFLAG_LOOP) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)

o2_ofast: $(SRCS)
	$(CPP) $(CFLAG_O2) $(CFLAG_OFAST) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)

o2_all: $(SRCS)
	$(CPP) $(CFLAG_O2) $(CFLAG_VEC) $(CFLAG_LOOP) $(CFLAG_OFAST) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)

# O3 flags
o3: $(SRCS)
	$(CPP) $(CFLAG_O3) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)

o3_vec: $(SRCS)
	$(CPP) $(CFLAG_O3) $(CFLAG_VEC) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)

o3_loop: $(SRCS)
	$(CPP) $(CFLAG_O3) $(CFLAG_LOOP) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)

o3_ofast: $(SRCS)
	$(CPP) $(CFLAG_O3) $(CFLAG_OFAST) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)

o3_all: $(SRCS)
	$(CPP) $(CFLAG_O3) $(CFLAG_VEC) $(CFLAG_LOOP) $(CFLAG_OFAST) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)

# Ofast flags
ofast: $(SRCS)
	$(CPP) $(CFLAG_OFAST) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)

ofast_vec: $(SRCS)
	$(CPP) $(CFLAG_OFAST) $(CFLAG_VEC) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)

ofast_loop: $(SRCS)
	$(CPP) $(CFLAG_OFAST) $(CFLAG_LOOP) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)

ofast_all: $(SRCS)
	$(CPP) $(CFLAG_OFAST) $(CFLAG_VEC) $(CFLAG_LOOP) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)

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
