# MPI_Alltoallv_custom

## Description
This is a custom implementation of MPI_Alltoallv. The all-to-all exchange is executed using a bunch of asynchronous isend/irecv calls. The number of simultaneous isend/irecv calls is limited by `COLLECTIVE_ISEND_IRECV_THROTTLE` to avoid choking the network. 

## Usage
To use this implementation, include the header file "MPI_Alltoallv_custom.h" and link the object file "MPI_Alltoallv_custom.o" to your program. Then, you can call MPI_Alltoallv_custom to replace MPI_Alltoallv.
If your sendcount buffer is size_t type, you can call MPI_Alltoallv_custom_s. MPI_Alltoallv_custom is actually calling MPI_Alltoallv_custom_s. 

## History
I initially developed a custom three-stage MPI_Alltoallv, which is actually saved in the legacy branch. The three-stage MPI_Alltoallv exchanges data in three steps: gather on the local node, exchange among nodes, and scatter on the local node. The implementation works (still some bugs in the legacy branch), but the performance is not good. The performance is limited by the memory load and store in sorting data in the gather and scatter steps. 

The current implementation uses isend and irecv calls as is done in the MPI_Alltoallv implementation in OpenMPI, MPICH, MVAPICH, etc. My implementation using MPI_Testsome() is more flexible for sparse data exchange.
