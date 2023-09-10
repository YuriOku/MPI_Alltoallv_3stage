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

#include <mpi.h>
#include <stddef.h>

void alltoallv_isend_irecv(const void *sendbuf, const size_t *sendcounts, const size_t *sdispls, MPI_Datatype sendtype, void *recvbuf,
                           const size_t *recvcounts, const size_t *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int MPI_Alltoallv_3stage_s(const void *sendbuf, const size_t *sendcounts, const size_t *sdispls, MPI_Datatype sendtype, void *recvbuf,
                           const size_t *recvcounts, const size_t *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int MPI_Alltoallv_3stage_s2(const void *sendbuf, const size_t *sendcounts, const size_t *sdispls, MPI_Datatype sendtype, void *recvbuf,
                           const size_t *recvcounts, const size_t *rdispls, MPI_Datatype recvtype, MPI_Comm comm);
                           
int MPI_Alltoallv_3stage_s_shared(const void *sendbuf, const size_t *sendcounts, const size_t *sdispls, MPI_Datatype sendtype, void *recvbuf,
                           const size_t *recvcounts, const size_t *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int MPI_Alltoallv_3stage(const void *sendbuf, const int *sendcounts, const int *sdispls, MPI_Datatype sendtype, void *recvbuf,
                         const int *recvcounts, const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int MPI_Alltoallv_3stage2(const void *sendbuf, const int *sendcounts, const int *sdispls, MPI_Datatype sendtype, void *recvbuf,
                         const int *recvcounts, const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);
                         
int MPI_Alltoallv_3stage_shared(const void *sendbuf, const int *sendcounts, const int *sdispls, MPI_Datatype sendtype, void *recvbuf,
                         const int *recvcounts, const int *rdispls, MPI_Datatype recvtype, MPI_Comm comm);

int MPI_Alltoall_3stage_s(const void *sendbuf, const size_t sendcount, MPI_Datatype sendtype, void *recvbuf, const size_t recvcount,
                          MPI_Datatype recvtype, MPI_Comm comm);

int MPI_Alltoall_3stage(const void *sendbuf, const int sendcount, MPI_Datatype sendtype, void *recvbuf, const int recvcount,
                        MPI_Datatype recvtype, MPI_Comm comm);
