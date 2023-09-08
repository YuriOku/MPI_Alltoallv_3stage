/* BSD 3-Clause License
 *
 * Copyright (c) 2023, YuriOku
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* test whether data is delivered to target rank */

#include <stdio.h>
#include <stdlib.h>

#include "../mpi_alltoallv_3stage.h"

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int use_3stage = 1;
  if(argc > 1)
    {
      use_3stage = atoi(argv[1]);
    }
  if(rank == 0)
    printf("use_3stage = %d\n", use_3stage);

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
  if(use_3stage)
    MPI_Alltoallv_3stage(sendbuf, sendcounts, sdispls, MPI_INT, recvbuf, recvcounts, rdispls, MPI_INT, MPI_COMM_WORLD);
  else
    MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_INT, recvbuf, recvcounts, rdispls, MPI_INT, MPI_COMM_WORLD);
  double t2 = MPI_Wtime();

  int success = 1;
  for(int i = 0; i < size; ++i)
    {
      for(int j = 0; j < n; ++j)
        {
          if(recvbuf[i * n + j] != i * n + rank)
            {
              printf("rank %d: recvbuf[%d] = %d, expected %d\n", rank, i * n + j, recvbuf[i * n + j], i * n + rank);
              success = 0;
            }
        }
    }

  MPI_Allreduce(MPI_IN_PLACE, &success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if(success == 1 && rank == 0)
    {
      printf("test passed\n");
    }
  else if(success == 0 && rank == 0)
    {
      printf("test failed\n");
    }

  double latency   = (t2 - t1) / (size * n);
  double bandwidth = (size * n * sizeof(int)) / (t2 - t1) / 1e6;
  if(rank == 0)
    printf("rank %d: latency = %e, bandwidth = %e\n", rank, latency, bandwidth);

  MPI_Finalize();

  return 0;
}