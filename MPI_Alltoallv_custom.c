/* BSD 3-Clause License
 *
 * Copyright (c) 2023, YuriOku
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
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
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "MPI_Alltoallv_custom.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PCHAR(x) ((char *)(x))
#define COLLECTIVE_ISEND_IRECV_THROTTLE 16

/* core routine of alltoallv. use isend and irecv internally, and the number of simultaneous calls are limited by
 * COLLECTIVE_ISEND_IRECV_THROTTLE */
void alltoallv_isend_irecv(const void *sendbuf, const size_t *sendcounts, const size_t *sdispls, MPI_Datatype sendtype, void *recvbuf,
                           const size_t *recvcounts, const size_t *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  int ntask, thistask;
  MPI_Comm_size(comm, &ntask);
  MPI_Comm_rank(comm, &thistask);

  int lptask = 1;
  while(lptask < ntask)
    lptask <<= 1;

  MPI_Request *requests_send = (MPI_Request *)malloc(sizeof(MPI_Request) * COLLECTIVE_ISEND_IRECV_THROTTLE);
  MPI_Status *statuses_send  = (MPI_Status *)malloc(sizeof(MPI_Status) * COLLECTIVE_ISEND_IRECV_THROTTLE);
  MPI_Request *requests_recv = (MPI_Request *)malloc(sizeof(MPI_Request) * COLLECTIVE_ISEND_IRECV_THROTTLE);
  MPI_Status *statuses_recv  = (MPI_Status *)malloc(sizeof(MPI_Status) * COLLECTIVE_ISEND_IRECV_THROTTLE);

  int typesize_send, typesize_recv;
  MPI_Type_size(sendtype, &typesize_send);
  MPI_Type_size(recvtype, &typesize_recv);

  if(recvcounts[thistask] > 0)  // local communication
    memcpy(PCHAR(recvbuf) + rdispls[thistask] * typesize_recv, PCHAR(sendbuf) + sdispls[thistask] * typesize_send,
           recvcounts[thistask] * typesize_recv);

  int i_send = 1, i_recv = 1, j_send, j_recv, k;
  int indices_send[COLLECTIVE_ISEND_IRECV_THROTTLE], indices_recv[COLLECTIVE_ISEND_IRECV_THROTTLE];
  int count_send = COLLECTIVE_ISEND_IRECV_THROTTLE, count_recv = COLLECTIVE_ISEND_IRECV_THROTTLE;
  for(k = 0; k < COLLECTIVE_ISEND_IRECV_THROTTLE; k++)
    indices_send[k] = indices_recv[k] = k;

  while(i_send < lptask || i_recv < lptask)
    {
      j_send = j_recv = 0;

      while(j_recv < count_recv)
        {
          if(i_recv < lptask)
            {
              int otask = thistask ^ i_recv;
              if(otask < ntask)
                if(recvcounts[otask] > 0)
                  MPI_Irecv(PCHAR(recvbuf) + rdispls[otask] * typesize_recv, recvcounts[otask] * typesize_recv, MPI_BYTE, otask, 0,
                            comm, &requests_recv[indices_recv[j_recv++]]);
            }
          else
            {
              MPI_Irecv(NULL, 0, MPI_BYTE, MPI_PROC_NULL, 0, comm, &requests_recv[indices_recv[j_recv++]]);
            }

          i_recv++;
        }

      while(j_send < count_send)
        {
          if(i_send < lptask)
            {
              int otask = thistask ^ i_send;
              if(otask < ntask)
                if(sendcounts[otask] > 0)
                  MPI_Issend(PCHAR(sendbuf) + sdispls[otask] * typesize_send, sendcounts[otask] * typesize_send, MPI_BYTE, otask, 0,
                             comm, &requests_send[indices_send[j_send++]]);
            }
          else
            {
              MPI_Isend(NULL, 0, MPI_BYTE, MPI_PROC_NULL, 0, comm, &requests_send[indices_send[j_send++]]);
            }

          i_send++;
        }

      if(MPI_Testsome(COLLECTIVE_ISEND_IRECV_THROTTLE, requests_recv, &count_recv, indices_recv, statuses_recv) != MPI_SUCCESS)
        count_recv = 0;

      if(MPI_Testsome(COLLECTIVE_ISEND_IRECV_THROTTLE, requests_send, &count_send, indices_send, statuses_send) != MPI_SUCCESS)
        count_send = 0;
    }

  /* fill request buffer by dummy and wait for leftovers */
  j_send = j_recv = 0;
  while(j_recv < count_recv)
    MPI_Irecv(NULL, 0, MPI_BYTE, MPI_PROC_NULL, 0, comm, &requests_recv[indices_recv[j_recv++]]);

  while(j_send < count_send)
    MPI_Isend(NULL, 0, MPI_BYTE, MPI_PROC_NULL, 0, comm, &requests_send[indices_send[j_send++]]);

  MPI_Waitall(COLLECTIVE_ISEND_IRECV_THROTTLE, requests_recv, statuses_recv);
  MPI_Waitall(COLLECTIVE_ISEND_IRECV_THROTTLE, requests_send, statuses_send);

  free(statuses_send);
  free(requests_send);
  free(statuses_recv);
  free(requests_recv);

  return;
}

/* another implementation that executes isends and irecvs block by block. can be faster when the number of MPI ranks is small. */
void alltoallv_isend_irecv2(const void *sendbuf, const size_t *sendcounts, const size_t *sdispls, MPI_Datatype sendtype, void *recvbuf,
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

/* another implementation that throws all irecv requests in the beginning and calls limited number of isend. equivalent performance
 * with alltoallv_isend_irecv() but more buffer consumption */
void alltoallv_isend_irecv3(const void *sendbuf, const size_t *sendcounts, const size_t *sdispls, MPI_Datatype sendtype, void *recvbuf,
                            const size_t *recvcounts, const size_t *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  int ntask, thistask;
  MPI_Comm_size(comm, &ntask);
  MPI_Comm_rank(comm, &thistask);

  int lptask = 1;
  while(lptask < ntask)
    lptask <<= 1;

  MPI_Request *requests_send = (MPI_Request *)malloc(sizeof(MPI_Request) * COLLECTIVE_ISEND_IRECV_THROTTLE);
  MPI_Status *statuses_send  = (MPI_Status *)malloc(sizeof(MPI_Status) * COLLECTIVE_ISEND_IRECV_THROTTLE);
  MPI_Request *requests_recv = (MPI_Request *)malloc(sizeof(MPI_Request) * ntask);
  MPI_Status *statuses_recv  = (MPI_Status *)malloc(sizeof(MPI_Status) * ntask);

  int typesize_send, typesize_recv;
  MPI_Type_size(sendtype, &typesize_send);
  MPI_Type_size(recvtype, &typesize_recv);

  if(recvcounts[thistask] > 0)  // local communication
    memcpy(PCHAR(recvbuf) + rdispls[thistask] * typesize_recv, PCHAR(sendbuf) + sdispls[thistask] * typesize_send,
           recvcounts[thistask] * typesize_recv);

  int i, j;
  int n_requests_recv = 0;
  for(i = 1; i < lptask; i++)
    {
      int otask = thistask ^ i;
      if(otask < ntask)
        if(recvcounts[otask] > 0)
          MPI_Irecv(PCHAR(recvbuf) + rdispls[otask] * typesize_recv, recvcounts[otask] * typesize_recv, MPI_BYTE, otask, 0, comm,
                    &requests_recv[n_requests_recv++]);
    }

  i = 1;
  int indices[COLLECTIVE_ISEND_IRECV_THROTTLE];
  int index_count = COLLECTIVE_ISEND_IRECV_THROTTLE;
  for(j = 0; j < COLLECTIVE_ISEND_IRECV_THROTTLE; j++)
    indices[j] = j;

  while(i < lptask)
    {
      j = 0;
      while(j < index_count)
        {
          int otask = thistask ^ i;
          if(otask < ntask)
            {
              if(sendcounts[otask] > 0)
                {
                  MPI_Issend(PCHAR(sendbuf) + sdispls[otask] * typesize_send, sendcounts[otask] * typesize_send, MPI_BYTE, otask, 0,
                             comm, &requests_send[indices[j++]]);
                }
            }

          i++;
          if(i >= lptask)
            break;
        }
      if(i >= lptask)
        break;

      MPI_Waitsome(COLLECTIVE_ISEND_IRECV_THROTTLE, requests_send, &index_count, indices, statuses_send);
    }

  /* fill request buffer by dummy */
  while(j < index_count)
    {
      MPI_Isend(NULL, 0, MPI_BYTE, MPI_PROC_NULL, 0, comm, &requests_send[indices[j++]]);
    }

  MPI_Waitall(COLLECTIVE_ISEND_IRECV_THROTTLE, requests_send, statuses_send);
  MPI_Waitall(n_requests_recv, requests_recv, statuses_recv);

  free(statuses_send);
  free(requests_send);
  free(statuses_recv);
  free(requests_recv);

  return;
}

int MPI_Alltoall_custom_s(const void *sendbuf, const size_t sendcount, MPI_Datatype sendtype, void *recvbuf, const size_t recvcount,
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

  alltoallv_isend_irecv(sendbuf, sendcounts_s, sdispls_s, sendtype, recvbuf, recvcounts_s, rdispls_s, recvtype, comm);

  free(rdispls_s);
  free(sdispls_s);
  free(recvcounts_s);
  free(sendcounts_s);

  return MPI_SUCCESS;
}

int MPI_Alltoall_custom(const void *sendbuf, const int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcount,
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

  alltoallv_isend_irecv(sendbuf, sendcounts_s, sdispls_s, sendtype, recvbuf, recvcounts_s, rdispls_s, recvtype, comm);

  free(rdispls_s);
  free(sdispls_s);
  free(recvcounts_s);
  free(sendcounts_s);

  return MPI_SUCCESS;
}

int MPI_Alltoallv_custom_s(const void *sendbuf, const size_t *sendcounts, const size_t *sdispls, MPI_Datatype sendtype, void *recvbuf,
                           const size_t *recvcounts, const size_t *rdispls, MPI_Datatype recvtype, MPI_Comm comm)
{
  alltoallv_isend_irecv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts, rdispls, recvtype, comm);
  return MPI_SUCCESS;
}

int MPI_Alltoallv_custom(const void *sendbuf, const int *sendcounts, const int *sdispls, MPI_Datatype sendtype, void *recvbuf,
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

  alltoallv_isend_irecv(sendbuf, sendcounts_s, sdispls_s, sendtype, recvbuf, recvcounts_s, rdispls_s, recvtype, comm);

  free(rdispls_s);
  free(sdispls_s);
  free(recvcounts_s);
  free(sendcounts_s);

  return MPI_SUCCESS;
}

int MPI_Alltoallv_custom2(const void *sendbuf, const int *sendcounts, const int *sdispls, MPI_Datatype sendtype, void *recvbuf,
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

  alltoallv_isend_irecv2(sendbuf, sendcounts_s, sdispls_s, sendtype, recvbuf, recvcounts_s, rdispls_s, recvtype, comm);

  free(rdispls_s);
  free(sdispls_s);
  free(recvcounts_s);
  free(sendcounts_s);

  return MPI_SUCCESS;
}

int MPI_Alltoallv_custom3(const void *sendbuf, const int *sendcounts, const int *sdispls, MPI_Datatype sendtype, void *recvbuf,
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

  alltoallv_isend_irecv3(sendbuf, sendcounts_s, sdispls_s, sendtype, recvbuf, recvcounts_s, rdispls_s, recvtype, comm);

  free(rdispls_s);
  free(sdispls_s);
  free(recvcounts_s);
  free(sendcounts_s);

  return MPI_SUCCESS;
}
