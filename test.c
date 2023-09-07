#include <stdio.h>
#include <stdlib.h>

#include "mpi_alltoallv_3stage.h"

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n           = 5;
  int *sendcounts = (int *)malloc(sizeof(int) * size);
  int *recvcounts = (int *)malloc(sizeof(int) * size);
  int *sdispls    = (int *)malloc(sizeof(int) * size);
  int *rdispls    = (int *)malloc(sizeof(int) * size);

  for(int i = 0; i < size; ++i)
    {
      sendcounts[i] = n;
      recvcounts[i] = n;
      sdispls[i]    = i * n;
      rdispls[i]    = i * n;
    }

  int *sendbuf = (int *)malloc(sizeof(int) * n * size);
  int *recvbuf = (int *)malloc(sizeof(int) * n * size);

  for(int i = 0; i < size; ++i)
    {
      for(int j = 0; j < n; ++j)
        {
          sendbuf[i * n + j] = rank * n + i;
        }
    }

  double t1 = MPI_Wtime();
  MPI_Alltoallv_3stage(sendbuf, sendcounts, sdispls, MPI_INT, recvbuf, recvcounts, rdispls, MPI_INT, MPI_COMM_WORLD);
  double t2 = MPI_Wtime();

  for(int i = 0; i < size; ++i)
    {
      for(int j = 0; j < n; ++j)
        {
          if (recvbuf[i * n + j] != i * n + rank)
            {
              printf("rank %d: recvbuf[%d] = %d, expected %d\n", rank, i * n + j, recvbuf[i * n + j], i * n + rank);
            }
        }
    }

    double latency = (t2 - t1) / (size * n);
    double bandwidth = (size * n * sizeof(int)) / (t2 - t1) / 1e6;
    printf("rank %d: latency = %e, bandwidth = %e\n", rank, latency, bandwidth);

  MPI_Finalize();

  return 0;
}