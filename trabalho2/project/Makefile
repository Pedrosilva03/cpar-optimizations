CPP = g++ -Wall -Wextra -lm -pg 
SRCS = main.cpp fluid_solver.cpp EventManager.cpp
OUTPUT = fluid_sim
# Flags de otimização
CFLAGS = -O3 -fopenmp -march=native -funroll-loops -ffast-math

all: $(OUTPUT)

$(OUTPUT): $(SRCS)
	$(CPP) $(CFLAGS) $(SRCS) -o $(OUTPUT)

# Limpeza
clean:
	@echo Cleaning up...
	@rm -f $(TARGET)
	@echo Done.