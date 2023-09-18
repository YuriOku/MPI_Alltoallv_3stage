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

#define LEVELMAX 15
#define ITERS 100

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int i, j;

  int use_3stage = 1;
  if(argc > 1)
    {
      use_3stage = atoi(argv[1]);
    }

  if(rank == 0)
    {
      printf("# use_3stage = %d\n", use_3stage);
      printf("# size = %d\n", size);
      printf("#\n# uniform data size\n");
      printf("# data size    latency [us]    bandwidth [MB/s]\n");
    }

  int *sendcounts = (int *)malloc(sizeof(int) * size);
  int *recvcounts = (int *)malloc(sizeof(int) * size);
  int *sdispls    = (int *)malloc(sizeof(int) * size);
  int *rdispls    = (int *)malloc(sizeof(int) * size);
  int *sendbuf    = NULL;
  int *recvbuf    = NULL;
  int success_all = 1;

  int level = 0;
  for(level = 0; level < LEVELMAX; level++)
    {
      int n = 1 << level;

      for(i = 0; i < size; ++i)
        {
          sendcounts[i] = n;
          recvcounts[i] = n;
          sdispls[i]    = i * n;
          rdispls[i]    = i * n;
        }

      sendbuf = (int *)malloc(sizeof(int) * n * size);
      recvbuf = (int *)malloc(sizeof(int) * n * size);

      int iter;
      double time = 0.0;
      for(iter = 0; iter < ITERS; iter++)
        {
          for(i = 0; i < size; ++i)
            {
              for(j = 0; j < n; ++j)
                {
                  sendbuf[i * n + j] = rank * size + i;
                }
            }

          double t1 = MPI_Wtime();
          if(use_3stage == 1)
            MPI_Alltoallv_3stage(sendbuf, sendcounts, sdispls, MPI_INT, recvbuf, recvcounts, rdispls, MPI_INT, MPI_COMM_WORLD);
          else if(use_3stage == 2)
            MPI_Alltoallv_3stage_shared(sendbuf, sendcounts, sdispls, MPI_INT, recvbuf, recvcounts, rdispls, MPI_INT, MPI_COMM_WORLD);
          else if(use_3stage == 3)
            MPI_Alltoallv_3stage2(sendbuf, sendcounts, sdispls, MPI_INT, recvbuf, recvcounts, rdispls, MPI_INT, MPI_COMM_WORLD);
          else
            MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_INT, recvbuf, recvcounts, rdispls, MPI_INT, MPI_COMM_WORLD);
          double t2 = MPI_Wtime();

          time += t2 - t1;

          int success = 1;
          for(i = 0; i < size; ++i)
            {
              for(j = 0; j < n; ++j)
                {
                  if(recvbuf[i * n + j] != i * size + rank)
                    {
                      printf("rank %d: recvbuf[%d] = %d, expected %d\n", rank, i * n + j, recvbuf[i * n + j], i * size + rank);
                      success = 0;
                    }
                }
            }

          MPI_Allreduce(MPI_IN_PLACE, &success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

          if(success == 0 && rank == 0)
            {
              printf("test failed\n");
              success_all = 0;
            }
        }

      free(sendbuf);
      free(recvbuf);

      time /= ITERS;

      double latency   = time * 1e6;
      double bandwidth = (size * n * sizeof(int)) / time / 1e6;
      if(rank == 0)
        printf("%7lu %18.5f %18.5f\n", n * sizeof(int), latency, bandwidth);
    }

  if(success_all == 1 && rank == 0)
    printf("passed all tests\n");

  if(rank == 0)
    {
      printf("#\n# non-uniform data size\n");
      printf("# data size    latency [us]    bandwidth [MB/s]   latency (Alltoall) [us]\n");
    }
  success_all = 1;

  for(level = 0; level < LEVELMAX; level++)
    {
      int n = 1 << level;

      int iter;
      double time = 0.0, time2 = 0.0;
      ;
      for(iter = 0; iter < ITERS; iter++)
        {
          int sdisp = 0, rdisp = 0;
          for(i = 0; i < size; ++i)
            {
              double r      = (double)rand() / RAND_MAX;
              sendcounts[i] = (r > 0.5) ? (int)(n * 4 * (r - 0.5)) : 0;
              sdispls[i]    = sdisp;
              sdisp += sendcounts[i];
            }
          double ta = MPI_Wtime();
          MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);
          double tb = MPI_Wtime();

          time2 += tb - ta;

          for(i = 0; i < size; ++i)
            {
              rdispls[i] = rdisp;
              rdisp += recvcounts[i];
            }

          sendbuf = (int *)malloc(sizeof(int) * sdisp);
          recvbuf = (int *)malloc(sizeof(int) * rdisp);

          for(i = 0; i < size; ++i)
            {
              for(j = 0; j < sendcounts[i]; ++j)
                {
                  sendbuf[sdispls[i] + j] = rank * size + i;
                }
            }

          double t1 = MPI_Wtime();
          if(use_3stage == 1)
            MPI_Alltoallv_3stage(sendbuf, sendcounts, sdispls, MPI_INT, recvbuf, recvcounts, rdispls, MPI_INT, MPI_COMM_WORLD);
          else if(use_3stage == 2)
            MPI_Alltoallv_3stage_shared(sendbuf, sendcounts, sdispls, MPI_INT, recvbuf, recvcounts, rdispls, MPI_INT, MPI_COMM_WORLD);
          else if(use_3stage == 3)
            MPI_Alltoallv_3stage2(sendbuf, sendcounts, sdispls, MPI_INT, recvbuf, recvcounts, rdispls, MPI_INT, MPI_COMM_WORLD);
          else
            MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_INT, recvbuf, recvcounts, rdispls, MPI_INT, MPI_COMM_WORLD);
          double t2 = MPI_Wtime();

          time += t2 - t1;

          int success = 1;
          for(i = 0; i < size; ++i)
            {
              for(j = 0; j < recvcounts[i]; ++j)
                {
                  if(recvbuf[rdispls[i] + j] != i * size + rank)
                    {
                      printf("rank %d: recvbuf[%d] = %d, expected %d\n", rank, rdispls[i] + j, recvbuf[rdispls[i] + j],
                             i * size + rank);
                      success = 0;
                    }
                }
            }

          MPI_Allreduce(MPI_IN_PLACE, &success, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

          if(success == 0 && rank == 0)
            {
              printf("test failed\n");
              success_all = 0;
            }
        }

      free(sendbuf);
      free(recvbuf);

      time /= ITERS;
      time2 /= ITERS;

      double latency   = time * 1e6;
      double latency2  = time2 * 1e6;
      double bandwidth = (size * n * sizeof(int)) / time / 1e6;
      if(rank == 0)
        printf("%7lu %18.5f %18.5f %18.5f\n", n * sizeof(int), latency, bandwidth, latency2);
    }

  free(sendcounts);
  free(recvcounts);
  free(sdispls);
  free(rdispls);

  if(success_all == 1 && rank == 0)
    printf("passed all tests\n");

  MPI_Finalize();

  return 0;
}
