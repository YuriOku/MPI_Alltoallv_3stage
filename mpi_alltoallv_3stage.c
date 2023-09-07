// BSD 3-Clause License

// Copyright (c) 2023, YuriOku

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "mpi_alltoallv_3stage.h"

#include <stdlib.h>
#include <string.h>

#define PCHAR(x) ((char *)(x))
#define COLLECTIVE_ISEND_IRECV_THROTTLE 32

int MPI_Alltoallv_3stage_s(const void *sendbuf, const size_t *sendcounts, const size_t *sdispls, MPI_Datatype sendtype, void *recvbuf,
                           const size_t *recvcounts, const size_t *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  /* check if sendtype==recvtype, the number of task on each node is the same, and we have enough memory */
  int flag_type = 0, flag_comm = 0, flag_task = 0;

  if(sendtype == recvtype)
    flag_type = 1;

  int typesize;
  MPI_Type_size(sendtype, &typesize);

  int ntask_all, thistask_all;
  MPI_Comm_size(comm, &ntask_all);
  MPI_Comm_rank(comm, &thistask_all);

  MPI_Comm comm_node;
  int ntask_node, thistask_node;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, thistask_all, MPI_INFO_NULL, &comm_node);
  MPI_Comm_size(comm_node, &ntask_node);
  MPI_Comm_rank(comm_node, &thistask_node);

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
  for(int i = 0; i < ntask_all; i++)
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

  if(thistask_node == 0)
    {
      for(int i = 0; i < ntask_inter; i++)
        {
          sendcounts_inter[i] = 0;
          recvcounts_inter[i] = 0;
          sdispls_inter[i]    = 0;
          rdispls_inter[i]    = 0;
        }

      for(int j = 0; j < ntask_node; j++)
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

          for(int i = 0; i < ntask_all; i++)
            {
              sendcounts_inter[i / ntask_node] += sendcounts_node[i + j * ntask_all];
              recvcounts_inter[i / ntask_node] += recvcounts_node[i + j * ntask_all];
            }
        }

      for(int i = 0; i < ntask_all; i++)
        {
          for(int j = 0; j < ntask_node; j++)
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

      for(int ind = 0; ind < ntask_all * ntask_node; ind++)
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

      for(int j = 0; j < ntask_inter; j++)
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

  for(int i = 0; i < ntask_all; i++)
    {
      memcpy(sendbuf_inter + sdispls_node[i * ntask_node + thistask_node] * typesize, PCHAR(sendbuf) + sdispls[i] * typesize,
             sendcounts[i] * typesize);
    }

  MPI_Barrier(comm_node);

  /*alltoallv*/
  if(thistask_node == 0)
    alltoallv_isend_irecv(sendbuf_inter, sendcounts_inter, sdispls_inter, sendtype, recvbuf_inter, recvcounts_inter, rdispls_inter,
                          recvtype, comm_inter);

  MPI_Barrier(comm_node);

  /*scatter*/
  for(int i = 0; i < ntask_all; i++)
    memcpy(PCHAR(recvbuf) + rdispls[i] * typesize,
           recvbuf_inter +
               rdispls_node[i % ntask_node + thistask_node * ntask_node + (i / ntask_node) * ntask_node * ntask_node] * typesize,
           recvcounts_node[i + thistask_node * ntask_all] * typesize);

  MPI_Barrier(comm_node);

  MPI_Win_free(&win);

  MPI_Comm_free(&comm_node);
  if(thistask_node == 0)
    MPI_Comm_free(&comm_inter);

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

  for(int i = 0; i < ntask_all; i++)
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

  for(int iloop = 0; iloop < nloop; iloop++)
    {
      int n_requests = 0;
      int ngrp_start = iloop * COLLECTIVE_ISEND_IRECV_THROTTLE + 1;
      int ngrp_end   = (iloop + 1) * COLLECTIVE_ISEND_IRECV_THROTTLE + 1;
      if(ngrp_end > lptask)
        ngrp_end = lptask;

      for(int ngrp = ngrp_start; ngrp < ngrp_end; ngrp++)
        {
          int otask = thistask ^ ngrp;
          if(otask < ntask)
            if(recvcounts[otask] > 0)
              MPI_Irecv(PCHAR(recvbuf) + rdispls[otask] * typesize_recv, recvcounts[otask] * typesize_recv, MPI_BYTE, otask, 0, comm,
                        &requests[n_requests++]);
        }

      for(int ngrp = ngrp_start; ngrp < ngrp_end; ngrp++)
        {
          int otask = thistask ^ ngrp;
          if(otask < ntask)
            if(sendcounts[otask] > 0)
              MPI_Isend(sendbuf + sdispls[otask] * typesize_send, sendcounts[otask] * typesize_send, MPI_BYTE, otask, 0, comm,
                        &requests[n_requests++]);
        }

      MPI_Waitall(n_requests, requests, statuses);
    }

  free(statuses);
  free(requests);
}
