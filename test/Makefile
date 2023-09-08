# generate osu_alltoallv from osu_alltoallv.c and test from test.c

# Compiler
CC = mpicc

# Compiler flags
CFLAGS = -Wall -O3 -g

# Linker flags
LDFLAGS = -lm

# Source files
SRC = ../mpi_alltoallv_3stage.c test.c osu_alltoallv.c osu_alltoallv_random.c osu_alltoallv_sparse.c \
 osu_alltoall.c osu_util_mpi.c osu_util.c osu_util_graph.c osu_util_papi.c osu_util_validation.c

# Object files
OBJ = $(SRC:.c=.o)

# Executable
EXE = osu_alltoall test.out osu_alltoallv  osu_alltoallv_random osu_alltoallv_sparse

# Default target
all: $(EXE)

# Link
test.out: ../mpi_alltoallv_3stage.o test.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

osu_alltoallv: ../mpi_alltoallv_3stage.o osu_alltoallv.o osu_util_mpi.o osu_util.o osu_util_graph.o osu_util_papi.o osu_util_validation.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

osu_alltoallv_random: ../mpi_alltoallv_3stage.o osu_alltoallv_random.o osu_util_mpi.o osu_util.o osu_util_graph.o osu_util_papi.o osu_util_validation.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	
osu_alltoallv_sparse: ../mpi_alltoallv_3stage.o osu_alltoallv_sparse.o osu_util_mpi.o osu_util.o osu_util_graph.o osu_util_papi.o osu_util_validation.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

osu_alltoall: ../mpi_alltoallv_3stage.o osu_alltoall.o osu_util_mpi.o osu_util.o osu_util_graph.o osu_util_papi.o osu_util_validation.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

# Compile
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean
clean:
	rm -f $(OBJ) $(EXE)

# Dependencies
../mpi_alltoallv_3stage.o: ../mpi_alltoallv_3stage.c ../mpi_alltoallv_3stage.h
test.o: test.c ../mpi_alltoallv_3stage.h osu_util.h osu_util_mpi.h
osu_alltoallv.o: osu_alltoallv.c osu_util_mpi.h
osu_alltoallv_random.o: osu_alltoallv_random.c osu_util_mpi.h
osu_alltoallv_sparse.o: osu_alltoallv_sparse.c osu_util_mpi.h
osu_alltoall.o: osu_alltoall.c osu_util_mpi.h
osu_util_mpi.o: osu_util_mpi.c osu_util_mpi.h
osu_util.o: osu_util.c osu_util.h osu_util_options.h
osu_util_graph.o: osu_util_graph.c osu_util_graph.h
osu_util_papi.o: osu_util_papi.c osu_util_papi.h
osu_util_validation.o: osu_util_validation.c osu_util.h osu_util_mpi.h

