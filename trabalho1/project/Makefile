CPP = g++ -Wall -Wextra -lm -pg 

# Optimization Flags
CFLAG_DEFAULT = -O2 -ffast-math
CFLAG_VEC = -ftree-vectorize -march=native
CFLAG_LOOP = -funroll-loops
CFLAG_O2 = -O2 -ffast-math
CFLAG_O3 = -O3 -ffast-math 
CFLAG_OFAST = -Ofast -ffast-math 
CFLAG_ALIGN = -falign-functions=16 -falign-loops=16
CFLAG_LTO = -flto

# Vectorization and Loop Optimization Flags
CFLAG_MSSE = -msse4
CFLAG_DEBUG = -g

SRCS = main.cpp fluid_solver.cpp EventManager.cpp
OUTPUT = fluid_sim

# Main target: 5,7638 s +- 0,0802 s | 4,08% | 81981,2 | validate.sh : 9,370638579 s ; 6,711913000 seconds user; 81981,2

all: $(OUTPUT)

$(OUTPUT): $(SRCS)
	$(CPP) $(CFLAG_DEFAULT) $(SRCS) -o $(OUTPUT)  # Default compilation without optimizations or profiling

# O2 flags
o2_vec: $(SRCS)
	$(CPP) $(CFLAG_O2) $(CFLAG_VEC) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)  #5.849 s  +- 0.158 s | 4.20% | 81981.5

o2_loop: $(SRCS)
	$(CPP) $(CFLAG_O2) $(CFLAG_LOOP) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)  #5.55 s +- 0.0476 s | 4.04% | 81981.2

o2_ofast: $(SRCS)
	$(CPP) $(CFLAG_O2) $(CFLAG_OFAST) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)  #4.60 s +- 0.0853 s | 4.11% | 81981.5

o2_all: $(SRCS)
	$(CPP) $(CFLAG_O2) $(CFLAG_VEC) $(CFLAG_LOOP) $(CFLAG_OFAST) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)  #4.50 s +- 0.0635 s | 4.92% | 81981.5

# O3 flags
o3: $(SRCS)
	$(CPP) $(CFLAG_O3) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)  #4.5150 s +- 0.0537 s | 4.12% | 81981.5

o3_vec: $(SRCS)
	$(CPP) $(CFLAG_O3) $(CFLAG_VEC) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)  #4.5644 s +- 0.0528 s | 4.11% | 81981.5

o3_loop: $(SRCS)
	$(CPP) $(CFLAG_O3) $(CFLAG_LOOP) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)  #4.4907 s +- 0.0653 s | 4.99% | 81981.5

o3_ofast: $(SRCS)
	$(CPP) $(CFLAG_O3) $(CFLAG_OFAST) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)  #4.6357 s +- 0.0232 s | 4.13% | 81981.5

o3_all: $(SRCS)
	$(CPP) $(CFLAG_O3) $(CFLAG_VEC) $(CFLAG_LOOP) $(CFLAG_OFAST) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)  #4.4990 s +- 0.0630 s | 4.93% | 81981.5

# Ofast flags
ofast: $(SRCS)
	$(CPP) $(CFLAG_OFAST) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)  #4.6503 s +- 0.0352 s | 4.11% | 81981.5

ofast_vec: $(SRCS)
	$(CPP) $(CFLAG_OFAST) $(CFLAG_VEC) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)  #4.5250 s +- 0.0642 s | 4.10% | 81981.5

ofast_loop: $(SRCS)
	$(CPP) $(CFLAG_OFAST) $(CFLAG_LOOP) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)  #4.4984 s +- 0.0652 s | 5% | 81981.5

ofast_all: $(SRCS)
	$(CPP) $(CFLAG_OFAST) $(CFLAG_VEC) $(CFLAG_LOOP) $(CFLAG_LTO) $(CFLAG_ALIGN) $(SRCS) -o $(OUTPUT)  #4.499 s +- 0.117 s | 4.95% | 81981.5

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
