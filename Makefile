# make a.out from test.c and mpi_alltoallv_3stage.c

# Compiler
CC = mpicc

# Compiler flags
CFLAGS = -Wall -O3 -std=c99

# Linker flags
LDFLAGS = -lm

# Source files
SOURCES = test.c mpi_alltoallv_3stage.c

# Object files
OBJECTS = $(SOURCES:.c=.o)

# Executable
EXECUTABLE = a.out

# Default target
all: $(SOURCES) $(EXECUTABLE)

# Link
$(EXECUTABLE): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@ $(LDFLAGS)

# Compile
.c.o:
	$(CC) $(CFLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

# Run test
run: $(EXECUTABLE)
	mpirun -np 4 ./$(EXECUTABLE)

# Dependencies
test.o: mpi_alltoallv_3stage.h
mpi_alltoallv_3stage.o: mpi_alltoallv_3stage.h



