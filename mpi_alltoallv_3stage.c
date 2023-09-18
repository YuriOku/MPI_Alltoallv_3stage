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

#include "mpi_alltoallv_3stage.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PCHAR(x) ((char *)(x))
#define COLLECTIVE_ISEND_IRECV_THROTTLE 32
#define MAX_NTASK_NODE 4

#define PRINT_TIMER 0

void alltoallv_isend_irecv(const void *sendbuf, const size_t *sendcounts, const size_t *sdispls, MPI_Datatype sendtype, void *recvbuf,
                           const size_t *recvcounts, const size_t *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  int ntask, thistask;
  MPI_Comm_size(comm, &ntask);
  MPI_Comm_rank(comm, &thistask);

  int lptask = 1;
  while(lptask < ntask)
    lptask <<= 1;

  int nloop = (lptask - 1) / COLLECTIVE_ISEND_IRECV_THROTTLE + 1;

  MPI_Request *requests = (MPI_Request *)malloc(sizeof(MPI_Request) * COLLECTIVE_ISEND_IRECV_THROTTLE * 2);
  MPI_Status *statuses  = (MPI_Status *)malloc(sizeof(MPI_Status) * COLLECTIVE_ISEND_IRECV_THROTTLE * 2);

  int typesize_send, typesize_recv;
  MPI_Type_size(sendtype, &typesize_send);
  MPI_Type_size(recvtype, &typesize_recv);

  if(recvcounts[thistask] > 0)  // local communication
    memcpy(PCHAR(recvbuf) + rdispls[thistask] * typesize_recv, PCHAR(sendbuf) + sdispls[thistask] * typesize_send,
           recvcounts[thistask] * typesize_recv);

  int iloop, ngrp;
  for(iloop = 0; iloop < nloop; iloop++)
    {
      int n_requests = 0;
      int ngrp_start = iloop * COLLECTIVE_ISEND_IRECV_THROTTLE + 1;
      int ngrp_end   = (iloop + 1) * COLLECTIVE_ISEND_IRECV_THROTTLE + 1;
      if(ngrp_end > lptask)
        ngrp_end = lptask;

      for(ngrp = ngrp_start; ngrp < ngrp_end; ngrp++)
        {
          int otask = thistask ^ ngrp;
          if(otask < ntask)
            if(recvcounts[otask] > 0)
              MPI_Irecv(PCHAR(recvbuf) + rdispls[otask] * typesize_recv, recvcounts[otask] * typesize_recv, MPI_BYTE, otask, 0, comm,
                        &requests[n_requests++]);
        }

      for(ngrp = ngrp_start; ngrp < ngrp_end; ngrp++)
        {
          int otask = thistask ^ ngrp;
          if(otask < ntask)
            if(sendcounts[otask] > 0)
              MPI_Isend(PCHAR(sendbuf) + sdispls[otask] * typesize_send, sendcounts[otask] * typesize_send, MPI_BYTE, otask, 0, comm,
                        &requests[n_requests++]);
        }

      MPI_Waitall(n_requests, requests, statuses);
    }

  free(statuses);
  free(requests);
}

int MPI_Alltoallv_3stage_s_shared(const void *sendbuf, const size_t *sendcounts, const size_t *sdispls, MPI_Datatype sendtype,
                                  void *recvbuf, const size_t *recvcounts, const size_t *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  /* check if sendtype==recvtype, the number of task on each node is the same, and we have enough memory */
  int flag_type = 0, flag_comm = 0, flag_task = 0;
  int i, j;

  int cnttmr = 0, showtmr = PRINT_TIMER;
  double tmr[10];
  int linetmr[10];

  linetmr[cnttmr] = __LINE__;
  tmr[cnttmr++]   = MPI_Wtime();

  if(sendtype == recvtype)
    flag_type = 1;

  int typesize;
  MPI_Type_size(sendtype, &typesize);

  int ntask_all, thistask_all;
  MPI_Comm_size(comm, &ntask_all);
  MPI_Comm_rank(comm, &thistask_all);

  MPI_Comm comm_node0, comm_node;
  int ntask_node, thistask_node;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, thistask_all, MPI_INFO_NULL, &comm_node0);
  MPI_Comm_size(comm_node0, &ntask_node);
  MPI_Comm_rank(comm_node0, &thistask_node);

  int max_ntask_node = MAX_NTASK_NODE;
  while(ntask_node % max_ntask_node != 0)
    max_ntask_node--;

  /* further split */
  if(ntask_node > max_ntask_node)
    {
      int color = thistask_node / max_ntask_node;
      MPI_Comm_split(comm_node0, color, thistask_node, &comm_node);
      MPI_Comm_rank(comm_node, &thistask_node);
      MPI_Comm_size(comm_node, &ntask_node);
    }
  else
    {
      comm_node = comm_node0;
    }

  MPI_Comm comm_inter;
  int ntask_inter, thistask_inter;
  if(thistask_node == 0)
    {
      MPI_Comm_split(comm, 0, thistask_all, &comm_inter);
      MPI_Comm_size(comm_inter, &ntask_inter);
      MPI_Comm_rank(comm_inter, &thistask_inter);
      MPI_Bcast(&ntask_inter, 1, MPI_INT, 0, comm_node);
    }
  else
    {
      MPI_Comm_split(comm, MPI_UNDEFINED, thistask_all, &comm_inter);
      MPI_Bcast(&ntask_inter, 1, MPI_INT, 0, comm_node);
    }

  int nmax, nmin;
  MPI_Allreduce(&ntask_node, &nmax, 1, MPI_INT, MPI_MAX, comm);
  MPI_Allreduce(&ntask_node, &nmin, 1, MPI_INT, MPI_MIN, comm);
  if(nmax == nmin)
    flag_task = 1;

  size_t totsend = 0, totrecv = 0;
  for(i = 0; i < ntask_all; i++)
    {
      totsend += sendcounts[i];
      totrecv += recvcounts[i];
    }

  size_t totsend_node, totrecv_node;
  MPI_Allreduce(&totsend, &totsend_node, 1, MPI_LONG_LONG_INT, MPI_SUM, comm_node);
  MPI_Allreduce(&totrecv, &totrecv_node, 1, MPI_LONG_LONG_INT, MPI_SUM, comm_node);

  MPI_Aint bufsize = 0;
  if(thistask_node == 0)
    bufsize = (totsend_node + totrecv_node) * typesize + 4 * ntask_inter * sizeof(size_t) +
              4 * ntask_node * ntask_all * sizeof(size_t) + 4 * ntask_node * sizeof(size_t);

  char *Base0;
  MPI_Win win;
  if(MPI_Win_allocate_shared(bufsize, 1, MPI_INFO_NULL, comm_node, &Base0, &win) == MPI_SUCCESS)
    flag_comm = 1;

  char *Base;
  MPI_Aint size;
  int disp_unit;
  MPI_Win_shared_query(win, 0, &size, &disp_unit, &Base);

  MPI_Allreduce(MPI_IN_PLACE, &flag_type, 1, MPI_INT, MPI_MIN, comm);
  MPI_Allreduce(MPI_IN_PLACE, &flag_comm, 1, MPI_INT, MPI_MIN, comm);
  MPI_Allreduce(MPI_IN_PLACE, &flag_task, 1, MPI_INT, MPI_MIN, comm);

  if(flag_type == 0 || flag_comm == 0 || flag_task == 0)
    {
      if(flag_comm == 1)
        MPI_Win_free(&win);

      MPI_Comm_free(&comm_node0);
      if(ntask_node > max_ntask_node)
        MPI_Comm_free(&comm_node);
      if(thistask_node == 0)
        MPI_Comm_free(&comm_inter);

      alltoallv_isend_irecv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
      return MPI_SUCCESS;
    }

  size_t offset       = 0;
  char *sendbuf_inter = Base + offset;
  offset += totsend_node * typesize;

  char *recvbuf_inter = Base + offset;
  offset += totrecv_node * typesize;

  size_t *sendcounts_inter = (size_t *)(Base + offset);
  offset += ntask_inter * sizeof(size_t);

  size_t *recvcounts_inter = (size_t *)(Base + offset);
  offset += ntask_inter * sizeof(size_t);

  size_t *sdispls_inter = (size_t *)(Base + offset);
  offset += ntask_inter * sizeof(size_t);

  size_t *rdispls_inter = (size_t *)(Base + offset);
  offset += ntask_inter * sizeof(size_t);

  size_t *sendcounts_node = (size_t *)(Base + offset);
  offset += ntask_node * ntask_all * sizeof(size_t);

  size_t *recvcounts_node = (size_t *)(Base + offset);
  offset += ntask_node * ntask_all * sizeof(size_t);

  size_t *sdispls_node = (size_t *)(Base + offset);
  offset += ntask_node * ntask_all * sizeof(size_t);

  size_t *rdispls_node = (size_t *)(Base + offset);
  offset += ntask_node * ntask_all * sizeof(size_t);

  size_t *totsendcounts_node = (size_t *)(Base + offset);
  offset += ntask_node * sizeof(size_t);

  size_t *totrecvcounts_node = (size_t *)(Base + offset);
  offset += ntask_node * sizeof(size_t);

  size_t *totsdispls_node = (size_t *)(Base + offset);
  offset += ntask_node * sizeof(size_t);

  size_t *totrdispls_node = (size_t *)(Base + offset);
  offset += ntask_node * sizeof(size_t);

  memcpy(sendcounts_node + thistask_node * ntask_all, sendcounts, ntask_all * sizeof(size_t));
  memcpy(recvcounts_node + thistask_node * ntask_all, recvcounts, ntask_all * sizeof(size_t));
  memcpy(sdispls_node + thistask_node * ntask_all, sdispls, ntask_all * sizeof(size_t));
  memcpy(rdispls_node + thistask_node * ntask_all, rdispls, ntask_all * sizeof(size_t));
  totsendcounts_node[thistask_node] = totsend;
  totrecvcounts_node[thistask_node] = totrecv;

  MPI_Barrier(comm_node);
  linetmr[cnttmr] = __LINE__;
  tmr[cnttmr++]   = MPI_Wtime();

  if(thistask_node == 0)
    {
      for(i = 0; i < ntask_inter; i++)
        {
          sendcounts_inter[i] = 0;
          recvcounts_inter[i] = 0;
          sdispls_inter[i]    = 0;
          rdispls_inter[i]    = 0;
        }

      for(j = 0; j < ntask_node; j++)
        {
          if(j == 0)
            {
              totsdispls_node[j] = 0;
              totrdispls_node[j] = 0;
            }
          else
            {
              totsdispls_node[j] = totsdispls_node[j - 1] + totsendcounts_node[j - 1];
              totrdispls_node[j] = totrdispls_node[j - 1] + totrecvcounts_node[j - 1];
            }

          for(i = 0; i < ntask_all; i++)
            {
              sendcounts_inter[i / ntask_node] += sendcounts_node[i + j * ntask_all];
              recvcounts_inter[i / ntask_node] += recvcounts_node[i + j * ntask_all];
            }
        }

      for(i = 0; i < ntask_all; i++)
        {
          for(j = 0; j < ntask_node; j++)
            {
              if(i == 0 && j == 0)
                {
                  sdispls_node[0] = 0;
                }
              else if(j == 0)
                {
                  sdispls_node[i * ntask_node] =
                      sdispls_node[i * ntask_node - 1] + sendcounts_node[i - 1 + (ntask_node - 1) * ntask_all];
                }
              else
                {
                  sdispls_node[i * ntask_node + j] = sdispls_node[i * ntask_node + j - 1] + sendcounts_node[i + (j - 1) * ntask_all];
                }
            }
        }

      int ind;
      for(ind = 0; ind < ntask_all * ntask_node; ind++)
        {
          if(ind == 0)
            rdispls_node[0] = 0;
          else
            {
              int i0            = (ind - 1) % ntask_node;
              int j             = ((ind - 1) / ntask_node) % ntask_node;
              int i1            = (ind - 1) / ntask_node / ntask_node;
              int i             = i0 + i1 * ntask_node;
              rdispls_node[ind] = rdispls_node[ind - 1] + recvcounts_node[i + j * ntask_all];
            }
        }

      for(j = 0; j < ntask_inter; j++)
        {
          if(j == 0)
            {
              sdispls_inter[j] = 0;
              rdispls_inter[j] = 0;
            }
          else
            {
              sdispls_inter[j] = sdispls_inter[j - 1] + sendcounts_inter[j - 1];
              rdispls_inter[j] = rdispls_inter[j - 1] + recvcounts_inter[j - 1];
            }
        }
    }

  MPI_Barrier(comm_node);
  linetmr[cnttmr] = __LINE__;
  tmr[cnttmr++]   = MPI_Wtime();

  for(i = 0; i < ntask_all; i++)
    {
      memcpy(sendbuf_inter + sdispls_node[i * ntask_node + thistask_node] * typesize, PCHAR(sendbuf) + sdispls[i] * typesize,
             sendcounts[i] * typesize);
    }

  MPI_Barrier(comm_node);
  linetmr[cnttmr] = __LINE__;
  tmr[cnttmr++]   = MPI_Wtime();

  /*alltoallv*/
  if(thistask_node == 0)
    alltoallv_isend_irecv(sendbuf_inter, sendcounts_inter, sdispls_inter, sendtype, recvbuf_inter, recvcounts_inter, rdispls_inter,
                          recvtype, comm_inter);

  MPI_Barrier(comm_node);
  linetmr[cnttmr] = __LINE__;
  tmr[cnttmr++]   = MPI_Wtime();

  /*scatter*/
  for(i = 0; i < ntask_all; i++)
    memcpy(PCHAR(recvbuf) + rdispls[i] * typesize,
           recvbuf_inter +
               rdispls_node[i % ntask_node + thistask_node * ntask_node + (i / ntask_node) * ntask_node * ntask_node] * typesize,
           recvcounts_node[i + thistask_node * ntask_all] * typesize);

  MPI_Barrier(comm_node);
  linetmr[cnttmr] = __LINE__;
  tmr[cnttmr++]   = MPI_Wtime();

  MPI_Win_free(&win);

  MPI_Comm_free(&comm_node0);
  if(ntask_node > max_ntask_node)
    MPI_Comm_free(&comm_node);
  if(thistask_node == 0)
    MPI_Comm_free(&comm_inter);

  if(thistask_all == 0 && showtmr == 1)
    {
      for(i = 0; i < cnttmr; i++)
        printf("time %d: L.%d %lf\n", i, linetmr[i], tmr[i] - tmr[0]);
    }

  return MPI_SUCCESS;
}

int MPI_Alltoallv_3stage_s2(const void *sendbuf, const size_t *sendcounts, const size_t *sdispls, MPI_Datatype sendtype, void *recvbuf,
                            const size_t *recvcounts, const size_t *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  /* check if sendtype==recvtype, the number of task on each node is the same, and we have enough memory */
  int flag_type = 0, flag_mem = 0, flag_task = 0;
  int i, j;

  int cnttmr = 0, showtmr = PRINT_TIMER;
  double tmr[10];
  int linetmr[10];

  linetmr[cnttmr] = __LINE__;
  tmr[cnttmr++]   = MPI_Wtime();

  if(sendtype == recvtype)
    flag_type = 1;

  int typesize;
  MPI_Type_size(sendtype, &typesize);

  int ntask_all, thistask_all;
  MPI_Comm_size(comm, &ntask_all);
  MPI_Comm_rank(comm, &thistask_all);

  int max_ntask_node = MAX_NTASK_NODE;
  while(ntask_all % max_ntask_node != 0)
    max_ntask_node--;

  /* split */
  MPI_Comm comm_node;
  int ntask_node, thistask_node;
  int color = thistask_all / max_ntask_node;
  MPI_Comm_split(comm, color, thistask_all, &comm_node);
  MPI_Comm_rank(comm_node, &thistask_node);
  MPI_Comm_size(comm_node, &ntask_node);

  int nmax, nmin;
  MPI_Allreduce(&ntask_node, &nmax, 1, MPI_INT, MPI_MAX, comm);
  MPI_Allreduce(&ntask_node, &nmin, 1, MPI_INT, MPI_MIN, comm);
  if(nmax == nmin && nmin > 1)
    flag_task = 1;

  int ntask_inter    = ntask_all / ntask_node;
  int thistask_inter = thistask_all / ntask_node;

  size_t totsend = 0, totrecv = 0;
  for(i = 0; i < ntask_all; i++)
    {
      totsend += sendcounts[i];
      totrecv += recvcounts[i];
    }

  size_t totsend_node, totrecv_node;
  MPI_Allreduce(&totsend, &totsend_node, 1, MPI_LONG_LONG_INT, MPI_SUM, comm_node);
  MPI_Allreduce(&totrecv, &totrecv_node, 1, MPI_LONG_LONG_INT, MPI_SUM, comm_node);

  size_t bufsize = (4 * ntask_inter + 4 * ntask_node * ntask_all) * sizeof(size_t);
  if(thistask_node == 0)
    bufsize += totsend_node * typesize;
  if(thistask_node == 1)
    bufsize += totrecv_node * typesize;

  MPI_Request *requests = (MPI_Request *)malloc(sizeof(MPI_Request) * ntask_node * COLLECTIVE_ISEND_IRECV_THROTTLE * 2);
  MPI_Status *statuses  = (MPI_Status *)malloc(sizeof(MPI_Status) * ntask_node * COLLECTIVE_ISEND_IRECV_THROTTLE * 2);

  char *Base = (char *)malloc(bufsize);
  if(Base != NULL)
    flag_mem = 1;

  MPI_Allreduce(MPI_IN_PLACE, &flag_type, 1, MPI_INT, MPI_MIN, comm);
  MPI_Allreduce(MPI_IN_PLACE, &flag_mem, 1, MPI_INT, MPI_MIN, comm);
  MPI_Allreduce(MPI_IN_PLACE, &flag_task, 1, MPI_INT, MPI_MIN, comm);

  /* if conditions are not met, do normal alltoallv */
  if(flag_type == 0 || flag_mem == 0 || flag_task == 0)
    {
      if(flag_mem == 1)
        free(Base);

      free(statuses);
      free(requests);

      MPI_Comm_free(&comm_node);

      alltoallv_isend_irecv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
      return MPI_SUCCESS;
    }

  linetmr[cnttmr] = __LINE__;
  tmr[cnttmr++]   = MPI_Wtime();

  size_t offset = 0;

  size_t *sendcounts_inter = (size_t *)(Base + offset);
  offset += ntask_inter * sizeof(size_t);

  size_t *recvcounts_inter = (size_t *)(Base + offset);
  offset += ntask_inter * sizeof(size_t);

  size_t *sdispls_inter = (size_t *)(Base + offset);
  offset += ntask_inter * sizeof(size_t);

  size_t *rdispls_inter = (size_t *)(Base + offset);
  offset += ntask_inter * sizeof(size_t);

  size_t *sendcounts_node = (size_t *)(Base + offset);
  offset += ntask_node * ntask_all * sizeof(size_t);

  size_t *recvcounts_node = (size_t *)(Base + offset);
  offset += ntask_node * ntask_all * sizeof(size_t);

  size_t *sdispls_node = (size_t *)(Base + offset);
  offset += ntask_node * ntask_all * sizeof(size_t);

  size_t *rdispls_node = (size_t *)(Base + offset);
  offset += ntask_node * ntask_all * sizeof(size_t);

  char *sendbuf_inter = NULL, *recvbuf_inter = NULL;
  if(thistask_node == 0)
    {
      sendbuf_inter = Base + offset;
      offset += totsend_node * typesize;
      sendbuf_inter[0] = 0;
    }
  if(thistask_node == 1)
    {
      recvbuf_inter = Base + offset;
      offset += totrecv_node * typesize;
      recvbuf_inter[0] = 0;
    }

  MPI_Allgather(sendcounts, ntask_all, MPI_LONG_LONG_INT, sendcounts_node, ntask_all, MPI_LONG_LONG_INT, comm_node);
  MPI_Allgather(recvcounts, ntask_all, MPI_LONG_LONG_INT, recvcounts_node, ntask_all, MPI_LONG_LONG_INT, comm_node);
  MPI_Allgather(sdispls, ntask_all, MPI_LONG_LONG_INT, sdispls_node, ntask_all, MPI_LONG_LONG_INT, comm_node);
  MPI_Allgather(rdispls, ntask_all, MPI_LONG_LONG_INT, rdispls_node, ntask_all, MPI_LONG_LONG_INT, comm_node);

  linetmr[cnttmr] = __LINE__;
  tmr[cnttmr++]   = MPI_Wtime();

  /* make tables of send/recv */

  for(i = 0; i < ntask_inter; i++)
    {
      sendcounts_inter[i] = 0;
      recvcounts_inter[i] = 0;
      sdispls_inter[i]    = 0;
      rdispls_inter[i]    = 0;
    }

  for(j = 0; j < ntask_node; j++)
    {
      for(i = 0; i < ntask_all; i++)
        {
          sendcounts_inter[i / ntask_node] += sendcounts_node[i + j * ntask_all];
          recvcounts_inter[i / ntask_node] += recvcounts_node[i + j * ntask_all];
        }
    }

  for(i = 0; i < ntask_all; i++)
    {
      for(j = 0; j < ntask_node; j++)
        {
          if(i == 0 && j == 0)
            {
              sdispls_node[0] = 0;
            }
          else if(j == 0)
            {
              sdispls_node[i * ntask_node] = sdispls_node[i * ntask_node - 1] + sendcounts_node[i - 1 + (ntask_node - 1) * ntask_all];
            }
          else
            {
              sdispls_node[i * ntask_node + j] = sdispls_node[i * ntask_node + j - 1] + sendcounts_node[i + (j - 1) * ntask_all];
            }
        }
    }

  int ind;
  for(ind = 0; ind < ntask_all * ntask_node; ind++)
    {
      if(ind == 0)
        rdispls_node[0] = 0;
      else
        {
          int i0            = (ind - 1) % ntask_node;
          int j             = ((ind - 1) / ntask_node) % ntask_node;
          int i1            = (ind - 1) / ntask_node / ntask_node;
          int i             = i0 + i1 * ntask_node;
          rdispls_node[ind] = rdispls_node[ind - 1] + recvcounts_node[i + j * ntask_all];
        }
    }

  for(j = 0; j < ntask_inter; j++)
    {
      if(j == 0)
        {
          sdispls_inter[j] = 0;
          rdispls_inter[j] = 0;
        }
      else
        {
          sdispls_inter[j] = sdispls_inter[j - 1] + sendcounts_inter[j - 1];
          rdispls_inter[j] = rdispls_inter[j - 1] + recvcounts_inter[j - 1];
        }
    }

  MPI_Barrier(comm_node);
  linetmr[cnttmr] = __LINE__;
  tmr[cnttmr++]   = MPI_Wtime();

  /* -- gather to task0 on node -- */
  int nloop = ntask_all / COLLECTIVE_ISEND_IRECV_THROTTLE + 1;
  int iloop, target;
  for(iloop = 0; iloop < nloop; iloop++)
    {
      int n_requests   = 0;
      int target_begin = iloop * COLLECTIVE_ISEND_IRECV_THROTTLE;
      int target_end   = (iloop + 1) * COLLECTIVE_ISEND_IRECV_THROTTLE;
      if(target_end > ntask_all)
        target_end = ntask_all;

      if(thistask_node == 0)
        for(target = target_begin; target < target_end; target++)
          {
            for(i = 0; i < ntask_node; i++)
              if(sendcounts_node[i * ntask_all + target] > 0)
                MPI_Irecv(sendbuf_inter + sdispls_node[target * ntask_node + i] * typesize,
                          sendcounts_node[i * ntask_all + target] * typesize, MPI_BYTE, i, target, comm_node, &requests[n_requests++]);
          }

      for(target = target_begin; target < target_end; target++)
        {
          if(sendcounts[target] > 0)
            MPI_Isend(PCHAR(sendbuf) + sdispls[target] * typesize, sendcounts[target] * typesize, MPI_BYTE, 0, target, comm_node,
                      &requests[n_requests++]);
        }

      MPI_Waitall(n_requests, requests, statuses);
    }

  MPI_Barrier(comm_node);
  linetmr[cnttmr] = __LINE__;
  tmr[cnttmr++]   = MPI_Wtime();

  /* all-to-all among nodes */

  if(recvcounts_inter[thistask_inter] > 0)  // local communication
    {
      if(thistask_node == 1)
        MPI_Recv(recvbuf_inter + rdispls_inter[thistask_inter] * typesize, recvcounts_inter[thistask_inter] * typesize, MPI_BYTE, 0,
                 thistask_inter, comm_node, MPI_STATUS_IGNORE);
      if(thistask_node == 0)
        MPI_Send(sendbuf_inter + sdispls_inter[thistask_inter] * typesize, recvcounts_inter[thistask_inter] * typesize, MPI_BYTE, 1,
                 thistask_inter, comm_node);
    }

  int lptask = 1;
  while(lptask < ntask_inter)
    lptask <<= 1;
  nloop = (lptask - 1) / COLLECTIVE_ISEND_IRECV_THROTTLE + 1;
  int ngrp;

  for(iloop = 0; iloop < nloop; iloop++)
    {
      int n_requests = 0;
      int ngrp_begin = iloop * COLLECTIVE_ISEND_IRECV_THROTTLE + 1;
      int ngrp_end   = (iloop + 1) * COLLECTIVE_ISEND_IRECV_THROTTLE + 1;
      if(ngrp_end > lptask)
        ngrp_end = lptask;

      if(thistask_node == 1)  // receiver
        for(ngrp = ngrp_begin; ngrp < ngrp_end; ngrp++)
          {
            target = thistask_inter ^ ngrp;
            if(target < ntask_inter)
              if(recvcounts_inter[target] > 0)
                MPI_Irecv(recvbuf_inter + rdispls_inter[target] * typesize, recvcounts_inter[target] * typesize, MPI_BYTE,
                          target * ntask_node, 0, comm, &requests[n_requests++]);  // real target is target*ntask_node on comm
          }

      if(thistask_node == 0)  // sender
        for(ngrp = ngrp_begin; ngrp < ngrp_end; ngrp++)
          {
            target = thistask_inter ^ ngrp;

            if(target < ntask_inter)
              if(sendcounts_inter[target] > 0)
                MPI_Isend(sendbuf_inter + sdispls_inter[target] * typesize, sendcounts_inter[target] * typesize, MPI_BYTE,
                          target * ntask_node + 1, 0, comm, &requests[n_requests++]);  // real target is target*ntask_node+1 on comm
          }

      MPI_Waitall(n_requests, requests, statuses);
    }

  MPI_Barrier(comm_node);
  linetmr[cnttmr] = __LINE__;
  tmr[cnttmr++]   = MPI_Wtime();

  /* -- scatter from task1 on node -- */
  nloop = ntask_all / COLLECTIVE_ISEND_IRECV_THROTTLE + 1;
  for(iloop = 0; iloop < nloop; iloop++)
    {
      int n_requests   = 0;
      int target_begin = iloop * COLLECTIVE_ISEND_IRECV_THROTTLE;
      int target_end   = (iloop + 1) * COLLECTIVE_ISEND_IRECV_THROTTLE;
      if(target_end > ntask_all)
        target_end = ntask_all;

      if(thistask_node == 1)
        for(target = target_begin; target < target_end; target++)
          {
            for(i = 0; i < ntask_node; i++)
              if(recvcounts_node[i * ntask_all + target] > 0)
                MPI_Isend(recvbuf_inter +
                              rdispls_node[target % ntask_node + i * ntask_node + (target / ntask_node) * ntask_node * ntask_node] *
                                  typesize,
                          recvcounts_node[i * ntask_all + target] * typesize, MPI_BYTE, i, target, comm_node, &requests[n_requests++]);
          }

      for(target = target_begin; target < target_end; target++)
        {
          if(recvcounts[target] > 0)
            MPI_Irecv(PCHAR(recvbuf) + rdispls[target] * typesize, recvcounts[target] * typesize, MPI_BYTE, 1, target, comm_node,
                      &requests[n_requests++]);
        }

      MPI_Waitall(n_requests, requests, statuses);
    }

  MPI_Barrier(comm_node);
  linetmr[cnttmr] = __LINE__;
  tmr[cnttmr++]   = MPI_Wtime();

  free(Base);
  free(statuses);
  free(requests);

  MPI_Comm_free(&comm_node);

  if(thistask_all == 0 && showtmr == 1)
    {
      for(i = 0; i < cnttmr; i++)
        printf("time %d: L.%d %lf\n", i, linetmr[i], tmr[i] - tmr[0]);
    }

  return MPI_SUCCESS;
}

int MPI_Alltoallv_3stage_s(const void *sendbuf, const size_t *sendcounts, const size_t *sdispls, MPI_Datatype sendtype, void *recvbuf,
                           const size_t *recvcounts, const size_t *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  /* check if sendtype==recvtype, the number of task on each node is the same, and we have enough memory */
  int flag_type = 0, flag_mem = 0, flag_task = 0;
  int i, j;
  int ip, target_node, target_task;

  int cnttmr = 0, showtmr = PRINT_TIMER;
  double tmr[10];
  int linetmr[10];

  linetmr[cnttmr] = __LINE__;
  tmr[cnttmr++]   = MPI_Wtime();

  if(sendtype == recvtype)
    flag_type = 1;

  int typesize;
  MPI_Type_size(sendtype, &typesize);

  int ntask_all, thistask_all;
  MPI_Comm_size(comm, &ntask_all);
  MPI_Comm_rank(comm, &thistask_all);

  int max_ntask_node = MAX_NTASK_NODE;
  while(ntask_all % max_ntask_node != 0)
    max_ntask_node--;

  /* split */
  MPI_Comm comm_node;
  int ntask_node, thistask_node;
  int color = thistask_all / max_ntask_node;
  MPI_Comm_split(comm, color, thistask_all, &comm_node);
  MPI_Comm_rank(comm_node, &thistask_node);
  MPI_Comm_size(comm_node, &ntask_node);

  int nmax, nmin;
  MPI_Allreduce(&ntask_node, &nmax, 1, MPI_INT, MPI_MAX, comm);
  MPI_Allreduce(&ntask_node, &nmin, 1, MPI_INT, MPI_MIN, comm);
  if(nmax == nmin && nmin > 1)
    flag_task = 1;

  int ntask_inter    = ntask_all / ntask_node;
  int thistask_inter = thistask_all / ntask_node;

  int ptask = 1;
  while(ptask < ntask_inter)
    ptask <<= 1;

  size_t totsend = 0, totrecv = 0;
  for(i = 0; i < ntask_all; i++)
    {
      totsend += sendcounts[i];
      totrecv += recvcounts[i];
    }

  size_t totsend_node, totrecv_node;
  MPI_Allreduce(&totsend, &totsend_node, 1, MPI_LONG_LONG_INT, MPI_SUM, comm_node);
  MPI_Allreduce(&totrecv, &totrecv_node, 1, MPI_LONG_LONG_INT, MPI_SUM, comm_node);

  size_t bufsize = 2 * (sizeof(MPI_Request) + sizeof(MPI_Status)) * ntask_node * ntask_node * COLLECTIVE_ISEND_IRECV_THROTTLE;
  if(thistask_node == 0)
    bufsize += totsend_node * typesize + (2 * ptask + ntask_node * ntask_all + ntask_node * ntask_node * ptask) * sizeof(size_t);
  if(thistask_node == ntask_node - 1)
    bufsize += totrecv_node * typesize + (2 * ptask + ntask_node * ntask_all + ntask_node * ntask_node * ptask) * sizeof(size_t);

  char *Base = (char *)malloc(bufsize);
  if(Base != NULL)
    flag_mem = 1;

  MPI_Allreduce(MPI_IN_PLACE, &flag_type, 1, MPI_INT, MPI_MIN, comm);
  MPI_Allreduce(MPI_IN_PLACE, &flag_mem, 1, MPI_INT, MPI_MIN, comm);
  MPI_Allreduce(MPI_IN_PLACE, &flag_task, 1, MPI_INT, MPI_MIN, comm);

  /* if conditions are not met, do normal alltoallv */
  if(flag_type == 0 || flag_mem == 0 || flag_task == 0)
    {
      if(flag_mem == 1)
        free(Base);

      MPI_Comm_free(&comm_node);

      alltoallv_isend_irecv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
      return MPI_SUCCESS;
    }

  linetmr[cnttmr] = __LINE__;
  tmr[cnttmr++]   = MPI_Wtime();

  size_t offset = 0, *sendcounts_inter = NULL, *recvcounts_inter = NULL, *sdispls_inter = NULL, *rdispls_inter = NULL,
         *sendcounts_node = NULL, *recvcounts_node = NULL, *sdispls_node = NULL, *rdispls_node = NULL;
  char *sendbuf_inter = NULL, *recvbuf_inter = NULL;

  if(thistask_node == 0)
    {
      sendcounts_inter = (size_t *)(Base + offset);
      offset += ptask * sizeof(size_t);

      sdispls_inter = (size_t *)(Base + offset);
      offset += ptask * sizeof(size_t);

      sendcounts_node = (size_t *)(Base + offset);
      offset += ntask_node * ntask_all * sizeof(size_t);

      sdispls_node = (size_t *)(Base + offset);
      offset += ntask_node * ntask_node * ptask * sizeof(size_t);

      sendbuf_inter = Base + offset;
      offset += totsend_node * typesize;
    }
  if(thistask_node == ntask_node - 1)
    {
      recvcounts_inter = (size_t *)(Base + offset);
      offset += ptask * sizeof(size_t);

      rdispls_inter = (size_t *)(Base + offset);
      offset += ptask * sizeof(size_t);

      recvcounts_node = (size_t *)(Base + offset);
      offset += ntask_node * ntask_all * sizeof(size_t);

      rdispls_node = (size_t *)(Base + offset);
      offset += ntask_node * ntask_node * ptask * sizeof(size_t);

      recvbuf_inter = Base + offset;
      offset += totrecv_node * typesize;
    }

  MPI_Request *requests = (MPI_Request *)(Base + offset);
  offset += sizeof(MPI_Request) * ntask_node * ntask_node * COLLECTIVE_ISEND_IRECV_THROTTLE;

  MPI_Status *statuses = (MPI_Status *)(Base + offset);
  offset += sizeof(MPI_Status) * ntask_node * ntask_node * COLLECTIVE_ISEND_IRECV_THROTTLE;

  MPI_Request *requests2 = (MPI_Request *)(Base + offset);
  offset += sizeof(MPI_Request) * ntask_node * ntask_node * COLLECTIVE_ISEND_IRECV_THROTTLE;

  MPI_Status *statuses2 = (MPI_Status *)(Base + offset);
  offset += sizeof(MPI_Status) * ntask_node * ntask_node * COLLECTIVE_ISEND_IRECV_THROTTLE;

  MPI_Gather(sendcounts, ntask_all, MPI_LONG_LONG_INT, sendcounts_node, ntask_all, MPI_LONG_LONG_INT, 0, comm_node);
  MPI_Gather(recvcounts, ntask_all, MPI_LONG_LONG_INT, recvcounts_node, ntask_all, MPI_LONG_LONG_INT, ntask_node - 1, comm_node);

  linetmr[cnttmr] = __LINE__;
  tmr[cnttmr++]   = MPI_Wtime();

  /* make tables of send/recv */

  if(thistask_node == 0)
    {
      for(i = 0; i < ptask; i++)
        {
          sendcounts_inter[i] = 0;
          sdispls_inter[i]    = 0;
        }

      for(ip = 0; ip < ptask; ip++)
        {
          target_node = thistask_inter ^ ip;
          if(target_node >= ntask_inter)
            continue;

          for(i = 0; i < ntask_node; i++)
            {
              target_task = target_node * ntask_node + i;
              for(j = 0; j < ntask_node; j++)
                {
                  /*counts of elements from j-th task on this node to target task, which is on ip-th node in hypercube pattern */
                  sendcounts_inter[ip] += sendcounts_node[j * ntask_all + target_task];
                }
            }

          if(ip > 0)
            {
              sdispls_inter[ip] = sdispls_inter[ip - 1] + sendcounts_inter[ip - 1];
            }
        }

      int idx, target_task_prev, first = 1;
      size_t cnt;
      sdispls_node[0] = 0;
      for(ip = 0; ip < ptask; ip++)
        {
          target_node = thistask_inter ^ ip;

          for(i = 0; i < ntask_node; i++)
            {
              target_task = target_node * ntask_node + i;
              idx         = ip * ntask_node + i;

              for(j = 0; j < ntask_node; j++)
                {
                  if(first)  // first element
                    {
                      sdispls_node[idx * ntask_node + j] = 0;
                      if(target_node < ntask_inter)
                        first = 0;
                    }
                  else if(target_node >= ntask_inter)
                    {
                      sdispls_node[idx * ntask_node + j] = sdispls_node[idx * ntask_node + j - 1];
                    }
                  else
                    {
                      if(j == 0)
                        cnt = sendcounts_node[target_task_prev + (ntask_node - 1) * ntask_all];
                      else
                        cnt = sendcounts_node[target_task_prev + (j - 1) * ntask_all];

                      sdispls_node[idx * ntask_node + j] = sdispls_node[idx * ntask_node + j - 1] + cnt;
                    }
                  target_task_prev = target_task;
                }
            }
        }
    }

  if(thistask_node == ntask_node - 1)
    {
      for(i = 0; i < ptask; i++)
        {
          recvcounts_inter[i] = 0;
          rdispls_inter[i]    = 0;
        }

      for(ip = 0; ip < ptask; ip++)
        {
          target_node = thistask_inter ^ ip;
          if(target_node >= ntask_inter)
            continue;

          for(i = 0; i < ntask_node; i++)
            {
              target_task = target_node * ntask_node + i;
              for(j = 0; j < ntask_node; j++)
                {
                  /*counts of elements from j-th task on this node to target task, which is on ip-th node in hypercube pattern */
                  recvcounts_inter[ip] += recvcounts_node[j * ntask_all + target_task];
                }
            }

          if(ip > 0)
            {
              rdispls_inter[ip] = rdispls_inter[ip - 1] + recvcounts_inter[ip - 1];
            }
        }

      int ind;
      size_t cnt;
      rdispls_node[0] = 0;
      for(ind = 1; ind < ptask * ntask_node * ntask_node; ind++)
        {
          int i0 = (ind - 1) % ntask_node;
          int j  = ((ind - 1) / ntask_node) % ntask_node;
          int i1 = (ind - 1) / ntask_node / ntask_node;
          int i  = i0 + (thistask_inter ^ i1) * ntask_node;
          if(i < ntask_all)
            cnt = recvcounts_node[i + j * ntask_all];
          else
            cnt = 0;

          rdispls_node[ind] = rdispls_node[ind - 1] + cnt;
        }
    }

  MPI_Barrier(comm_node);
  linetmr[cnttmr] = __LINE__;
  tmr[cnttmr++]   = MPI_Wtime();

  int iloop, nloop = ptask / COLLECTIVE_ISEND_IRECV_THROTTLE + 1;

  for(iloop = 0; iloop < nloop; iloop++)
    {
      int n_requests = 0, n_requests2 = 0;
      int ip_begin = iloop * COLLECTIVE_ISEND_IRECV_THROTTLE;
      int ip_end   = (iloop + 1) * COLLECTIVE_ISEND_IRECV_THROTTLE;
      if(ip_end > ptask)
        ip_end = ptask;

      if(thistask_node == 0)
        {
          /* reuse the buffer to inprove cache hit */
          char *sendbuf_inter2 = sendbuf_inter - sdispls_inter[ip_begin] * typesize;
          // char *sendbuf_inter2 = sendbuf_inter;
          int target_task_ip;

          for(ip = ip_begin; ip < ip_end; ip++)
            {
              target_node = thistask_inter ^ ip;

              if(target_node < ntask_inter)
                for(i = 0; i < ntask_node; i++)
                  {
                    target_task    = target_node * ntask_node + i;
                    target_task_ip = ip * ntask_node + i;

                    /* -- gather -- */
                    for(j = 1; j < ntask_node; j++)
                      if(sendcounts_node[j * ntask_all + target_task] > 0)
                        MPI_Irecv(sendbuf_inter2 + sdispls_node[target_task_ip * ntask_node + j] * typesize,
                                  sendcounts_node[j * ntask_all + target_task] * typesize, MPI_BYTE, j, target_task, comm_node,
                                  &requests[n_requests++]);

                    /* receive from task n-1 on node */
                    if(recvcounts[target_task] > 0)
                      MPI_Irecv(PCHAR(recvbuf) + rdispls[target_task] * typesize, recvcounts[target_task] * typesize, MPI_BYTE,
                                ntask_node - 1, target_task, comm_node, &requests2[n_requests2++]);
                  }
            }

          /* local copy */
          for(ip = ip_begin; ip < ip_end; ip++)
            {
              target_node = thistask_inter ^ ip;

              if(target_node < ntask_inter)
                for(i = 0; i < ntask_node; i++)
                  {
                    target_task    = target_node * ntask_node + i;
                    target_task_ip = ip * ntask_node + i;

                    if(sendcounts[target_task] > 0)
                      memcpy(sendbuf_inter2 + sdispls_node[target_task_ip * ntask_node] * typesize,
                             PCHAR(sendbuf) + sdispls[target_task] * typesize, sendcounts[target_task] * typesize);
                  }
            }

          /* wait to send until gather has done */
          MPI_Waitall(n_requests, requests, statuses);

          /* send to other nodes */
          for(ip = ip_begin; ip < ip_end; ip++)
            {
              target_node = thistask_inter ^ ip;

              if(target_node < ntask_inter)
                if(sendcounts_inter[ip] > 0)
                  {
                    MPI_Isend(sendbuf_inter2 + sdispls_inter[ip] * typesize, sendcounts_inter[ip] * typesize, MPI_BYTE,
                              (target_node + 1) * ntask_node - 1, 0, comm,
                              &requests2[n_requests2++]);  // real target is target*ntask_node+(ntask_node-1) on comm
                  }
            }

          MPI_Waitall(n_requests2, requests2, statuses2);
        }
      else if(thistask_node == ntask_node - 1)
        {
          /* reuse the buffer to inprove cache hit */
          char *recvbuf_inter2 = recvbuf_inter - rdispls_inter[ip_begin] * typesize;
          // char *recvbuf_inter2 = recvbuf_inter;

          for(ip = ip_begin; ip < ip_end; ip++)
            {
              target_node = thistask_inter ^ ip;

              if(target_node < ntask_inter)
                {
                  /* receive from other nodes */
                  if(recvcounts_inter[ip] > 0)
                    MPI_Irecv(recvbuf_inter2 + rdispls_inter[ip] * typesize, recvcounts_inter[ip] * typesize, MPI_BYTE,
                              target_node * ntask_node, 0, comm,
                              &requests[n_requests++]);  // real target is target*ntask_node on comm

                  /* send to task 0 on node */
                  for(i = 0; i < ntask_node; i++)
                    {
                      target_task = target_node * ntask_node + i;

                      if(sendcounts[target_task] > 0)
                        MPI_Isend(PCHAR(sendbuf) + sdispls[target_task] * typesize, sendcounts[target_task] * typesize, MPI_BYTE, 0,
                                  target_task, comm_node, &requests2[n_requests2++]);
                    }
                }
            }

          /* wait to scatter until recv has done */
          MPI_Waitall(n_requests, requests, statuses);

          /* scatter to all tasks on node */
          for(ip = ip_begin; ip < ip_end; ip++)
            {
              target_node = thistask_inter ^ ip;

              if(target_node < ntask_inter)
                for(i = 0; i < ntask_node; i++)
                  {
                    target_task = target_node * ntask_node + i;

                    for(j = 0; j < ntask_node - 1; j++)
                      if(recvcounts_node[j * ntask_all + target_task] > 0)
                        MPI_Isend(recvbuf_inter2 + rdispls_node[i + j * ntask_node + ip * ntask_node * ntask_node] * typesize,
                                  recvcounts_node[j * ntask_all + target_task] * typesize, MPI_BYTE, j, target_task, comm_node,
                                  &requests2[n_requests2++]);
                  }
            }

          /* local copy */
          for(ip = ip_begin; ip < ip_end; ip++)
            {
              target_node = thistask_inter ^ ip;

              if(target_node < ntask_inter)
                for(i = 0; i < ntask_node; i++)
                  {
                    target_task = target_node * ntask_node + i;

                    if(recvcounts[target_task] > 0)
                      memcpy(
                          PCHAR(recvbuf) + rdispls[target_task] * typesize,
                          recvbuf_inter2 + rdispls_node[i + (ntask_node - 1) * ntask_node + ip * ntask_node * ntask_node] * typesize,
                          recvcounts[target_task] * typesize);
                  }
            }

          MPI_Waitall(n_requests2, requests2, statuses2);
        }
      else
        {
          /* send to task 0 and recv from task n-1 */
          for(ip = ip_begin; ip < ip_end; ip++)
            {
              target_node = thistask_inter ^ ip;

              if(target_node < ntask_inter)
                {
                  for(i = 0; i < ntask_node; i++)
                    {
                      target_task = target_node * ntask_node + i;

                      if(sendcounts[target_task] > 0)
                        MPI_Isend(PCHAR(sendbuf) + sdispls[target_task] * typesize, sendcounts[target_task] * typesize, MPI_BYTE, 0,
                                  target_task, comm_node, &requests[n_requests++]);

                      if(recvcounts[target_task] > 0)
                        MPI_Irecv(PCHAR(recvbuf) + rdispls[target_task] * typesize, recvcounts[target_task] * typesize, MPI_BYTE,
                                  ntask_node - 1, target_task, comm_node, &requests[n_requests++]);
                    }
                }
            }
          MPI_Waitall(n_requests, requests, statuses);
        }
    }

  MPI_Barrier(comm_node);
  linetmr[cnttmr] = __LINE__;
  tmr[cnttmr++]   = MPI_Wtime();

  free(Base);

  MPI_Comm_free(&comm_node);

  if(thistask_all == 0 && showtmr == 1)
    {
      for(i = 0; i < cnttmr; i++)
        printf("time %d: L.%d %lf\n", i, linetmr[i], tmr[i] - tmr[0]);
    }

  return MPI_SUCCESS;
}

int MPI_Alltoallv_3stage(const void *sendbuf, const int *sendcounts, const int *sdispls, MPI_Datatype sendtype, void *recvbuf,
                         const int *recvcounts, const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  int ntask_all, thistask_all;
  MPI_Comm_size(comm, &ntask_all);
  MPI_Comm_rank(comm, &thistask_all);

  size_t *sendcounts_s = (size_t *)malloc(ntask_all * sizeof(size_t));
  size_t *recvcounts_s = (size_t *)malloc(ntask_all * sizeof(size_t));
  size_t *sdispls_s    = (size_t *)malloc(ntask_all * sizeof(size_t));
  size_t *rdispls_s    = (size_t *)malloc(ntask_all * sizeof(size_t));

  if(sendcounts_s == NULL || recvcounts_s == NULL || sdispls_s == NULL || rdispls_s == NULL)
    {
      if(rdispls_s != NULL)
        free(rdispls_s);
      if(sdispls_s != NULL)
        free(sdispls_s);
      if(recvcounts_s != NULL)
        free(recvcounts_s);
      if(sendcounts_s != NULL)
        free(sendcounts_s);

      return MPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
    }

  int i;
  for(i = 0; i < ntask_all; i++)
    {
      sendcounts_s[i] = sendcounts[i];
      recvcounts_s[i] = recvcounts[i];
      sdispls_s[i]    = sdispls[i];
      rdispls_s[i]    = rdispls[i];
    }

  int ret = MPI_Alltoallv_3stage_s(sendbuf, sendcounts_s, sdispls_s, sendtype, recvbuf, recvcounts_s, rdispls_s, recvtype, comm);

  free(rdispls_s);
  free(sdispls_s);
  free(recvcounts_s);
  free(sendcounts_s);

  return ret;
}

int MPI_Alltoallv_3stage2(const void *sendbuf, const int *sendcounts, const int *sdispls, MPI_Datatype sendtype, void *recvbuf,
                          const int *recvcounts, const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  int ntask_all, thistask_all;
  MPI_Comm_size(comm, &ntask_all);
  MPI_Comm_rank(comm, &thistask_all);

  size_t *sendcounts_s = (size_t *)malloc(ntask_all * sizeof(size_t));
  size_t *recvcounts_s = (size_t *)malloc(ntask_all * sizeof(size_t));
  size_t *sdispls_s    = (size_t *)malloc(ntask_all * sizeof(size_t));
  size_t *rdispls_s    = (size_t *)malloc(ntask_all * sizeof(size_t));

  if(sendcounts_s == NULL || recvcounts_s == NULL || sdispls_s == NULL || rdispls_s == NULL)
    {
      if(rdispls_s != NULL)
        free(rdispls_s);
      if(sdispls_s != NULL)
        free(sdispls_s);
      if(recvcounts_s != NULL)
        free(recvcounts_s);
      if(sendcounts_s != NULL)
        free(sendcounts_s);

      return MPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
    }

  int i;
  for(i = 0; i < ntask_all; i++)
    {
      sendcounts_s[i] = sendcounts[i];
      recvcounts_s[i] = recvcounts[i];
      sdispls_s[i]    = sdispls[i];
      rdispls_s[i]    = rdispls[i];
    }

  int ret = MPI_Alltoallv_3stage_s2(sendbuf, sendcounts_s, sdispls_s, sendtype, recvbuf, recvcounts_s, rdispls_s, recvtype, comm);

  free(rdispls_s);
  free(sdispls_s);
  free(recvcounts_s);
  free(sendcounts_s);

  return ret;
}

int MPI_Alltoallv_3stage_shared(const void *sendbuf, const int *sendcounts, const int *sdispls, MPI_Datatype sendtype, void *recvbuf,
                                const int *recvcounts, const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  int ntask_all, thistask_all;
  MPI_Comm_size(comm, &ntask_all);
  MPI_Comm_rank(comm, &thistask_all);

  size_t *sendcounts_s = (size_t *)malloc(ntask_all * sizeof(size_t));
  size_t *recvcounts_s = (size_t *)malloc(ntask_all * sizeof(size_t));
  size_t *sdispls_s    = (size_t *)malloc(ntask_all * sizeof(size_t));
  size_t *rdispls_s    = (size_t *)malloc(ntask_all * sizeof(size_t));

  if(sendcounts_s == NULL || recvcounts_s == NULL || sdispls_s == NULL || rdispls_s == NULL)
    {
      if(rdispls_s != NULL)
        free(rdispls_s);
      if(sdispls_s != NULL)
        free(sdispls_s);
      if(recvcounts_s != NULL)
        free(recvcounts_s);
      if(sendcounts_s != NULL)
        free(sendcounts_s);

      return MPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
    }

  int i;
  for(i = 0; i < ntask_all; i++)
    {
      sendcounts_s[i] = sendcounts[i];
      recvcounts_s[i] = recvcounts[i];
      sdispls_s[i]    = sdispls[i];
      rdispls_s[i]    = rdispls[i];
    }

  int ret =
      MPI_Alltoallv_3stage_s_shared(sendbuf, sendcounts_s, sdispls_s, sendtype, recvbuf, recvcounts_s, rdispls_s, recvtype, comm);

  free(rdispls_s);
  free(sdispls_s);
  free(recvcounts_s);
  free(sendcounts_s);

  return ret;
}

int MPI_Alltoall_3stage_s(const void *sendbuf, const size_t sendcount, MPI_Datatype sendtype, void *recvbuf, const size_t recvcount,
                          MPI_Datatype recvtype, MPI_Comm comm)
{
  int ntask_all, thistask_all;
  MPI_Comm_size(comm, &ntask_all);
  MPI_Comm_rank(comm, &thistask_all);

  size_t *sendcounts_s = (size_t *)malloc(ntask_all * sizeof(size_t));
  size_t *recvcounts_s = (size_t *)malloc(ntask_all * sizeof(size_t));
  size_t *sdispls_s    = (size_t *)malloc(ntask_all * sizeof(size_t));
  size_t *rdispls_s    = (size_t *)malloc(ntask_all * sizeof(size_t));

  if(sendcounts_s == NULL || recvcounts_s == NULL || sdispls_s == NULL || rdispls_s == NULL)
    {
      if(rdispls_s != NULL)
        free(rdispls_s);
      if(sdispls_s != NULL)
        free(sdispls_s);
      if(recvcounts_s != NULL)
        free(recvcounts_s);
      if(sendcounts_s != NULL)
        free(sendcounts_s);

      return MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    }

  int i;
  for(i = 0; i < ntask_all; i++)
    {
      sendcounts_s[i] = sendcount;
      recvcounts_s[i] = recvcount;
      sdispls_s[i]    = sendcount * i;
      rdispls_s[i]    = recvcount * i;
    }

  int ret = MPI_Alltoallv_3stage_s(sendbuf, sendcounts_s, sdispls_s, sendtype, recvbuf, recvcounts_s, rdispls_s, recvtype, comm);

  free(rdispls_s);
  free(sdispls_s);
  free(recvcounts_s);
  free(sendcounts_s);

  return ret;
}

int MPI_Alltoall_3stage(const void *sendbuf, const int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcount,
                        MPI_Datatype recvtype, MPI_Comm comm)
{
  int ntask_all, thistask_all;
  MPI_Comm_size(comm, &ntask_all);
  MPI_Comm_rank(comm, &thistask_all);

  size_t *sendcounts_s = (size_t *)malloc(ntask_all * sizeof(size_t));
  size_t *recvcounts_s = (size_t *)malloc(ntask_all * sizeof(size_t));
  size_t *sdispls_s    = (size_t *)malloc(ntask_all * sizeof(size_t));
  size_t *rdispls_s    = (size_t *)malloc(ntask_all * sizeof(size_t));

  if(sendcounts_s == NULL || recvcounts_s == NULL || sdispls_s == NULL || rdispls_s == NULL)
    {
      if(rdispls_s != NULL)
        free(rdispls_s);
      if(sdispls_s != NULL)
        free(sdispls_s);
      if(recvcounts_s != NULL)
        free(recvcounts_s);
      if(sendcounts_s != NULL)
        free(sendcounts_s);

      return MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, comm);
    }

  int i;
  for(i = 0; i < ntask_all; i++)
    {
      sendcounts_s[i] = sendcount;
      recvcounts_s[i] = recvcount;
      sdispls_s[i]    = sendcount * i;
      rdispls_s[i]    = recvcount * i;
    }

  int ret = MPI_Alltoallv_3stage_s(sendbuf, sendcounts_s, sdispls_s, sendtype, recvbuf, recvcounts_s, rdispls_s, recvtype, comm);

  free(rdispls_s);
  free(sdispls_s);
  free(recvcounts_s);
  free(sendcounts_s);

  return ret;
}
