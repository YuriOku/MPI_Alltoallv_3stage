// repeat comm split test

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ITERS 100000
#define COLLECTIVE_ISEND_IRECV_THROTTLE 32
#define MAX_NTASK_NODE 8

#define PRINT_TIMER 0

int main(int argc, char *argv[])
{
  int rank, size, color, key;
  int new_rank, new_size;
  int i, j, k;
  int *ranks;
  MPI_Comm new_comm;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int sum = 0;  // dummy variable to prevent compiler optimization

  for(k = 0; k < ITERS; k++)
    {
      MPI_Datatype sendtype = MPI_INT, recvtype = MPI_INT;
      MPI_Comm comm      = MPI_COMM_WORLD;
      size_t *sendcounts = (size_t *)malloc(sizeof(size_t) * size);
      size_t *recvcounts = (size_t *)malloc(sizeof(size_t) * size);
      for(j = 0; j < size; j++)
        {
          sendcounts[j] = k * 1000000;
          recvcounts[j] = k * 1000000;
        }

      {
        /* check if sendtype==recvtype, the number of task on each node is the same, and we have enough memory */
        int flag_type = 0, flag_mem = 0, flag_task = 0;
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

        size_t bufsize = (4 * ptask + 2 * ntask_node * ntask_all + 2 * ntask_node * ntask_node * ptask) * sizeof(size_t);
        if(thistask_node == 0)
          bufsize += totsend_node * typesize;
        if(thistask_node == ntask_node - 1)
          bufsize += totrecv_node * typesize;

        MPI_Request *requests = (MPI_Request *)malloc(sizeof(MPI_Request) * ntask_node * ntask_node * COLLECTIVE_ISEND_IRECV_THROTTLE);
        MPI_Status *statuses  = (MPI_Status *)malloc(sizeof(MPI_Status) * ntask_node * ntask_node * COLLECTIVE_ISEND_IRECV_THROTTLE);

        MPI_Request *requests2 =
            (MPI_Request *)malloc(sizeof(MPI_Request) * ntask_node * ntask_node * COLLECTIVE_ISEND_IRECV_THROTTLE);
        MPI_Status *statuses2 = (MPI_Status *)malloc(sizeof(MPI_Status) * ntask_node * ntask_node * COLLECTIVE_ISEND_IRECV_THROTTLE);

        char *Base = (char *)malloc(bufsize);
        if(Base != NULL)
          flag_mem = 1;

        MPI_Allreduce(MPI_IN_PLACE, &flag_type, 1, MPI_INT, MPI_MIN, comm);
        MPI_Allreduce(MPI_IN_PLACE, &flag_mem, 1, MPI_INT, MPI_MIN, comm);
        MPI_Allreduce(MPI_IN_PLACE, &flag_task, 1, MPI_INT, MPI_MIN, comm);

        sum += flag_type + flag_mem + flag_task;

        if(flag_mem == 1)
          free(Base);

        free(statuses2);
        free(requests2);
        free(statuses);
        free(requests);

        MPI_Comm_free(&comm_node);
      }
    }

  MPI_Barrier(MPI_COMM_WORLD);

  if(rank == 0)
    {
      printf("sum = %d\n", sum);
    }

  MPI_Finalize();
  return 0;
}
