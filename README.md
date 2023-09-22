# MPI_Alltoallv_custom

## Description
This is a shared-memory-aware three-stage implementation of MPI_Alltoallv. The all-to-all exchange is divided into three stages: intra-node gather, inter-node exchange, and intra-node scatter. The intra-node communication is performed on the MPI-3 shared-memory window, and the inter-node communication is performed by a bunch of MPI_Isend and MPI_Irecv.

## Limitation
There are three limitations of this implementation. First, The current implementation only supports the case where the number of processes on each node is the same. Second, the data type of the send buffer and the receive buffer must be the same. Third, there should be enough memory to store the send buffer and the receive buffer on each node. If these conditions are not satisfied, the program will back to the original MPI_Alltoallv.

## Usage
To use this implementation, you need to include the header file "MPI_Alltoallv_custom.h" and link the object file "MPI_Alltoallv_custom.o" to your program. Then you can call MPI_Alltoallv_custom to replace MPI_Alltoallv.
If your sendcount buffer is size_t type, you can call MPI_Alltoallv_custom_s. MPI_Alltoallv_custom is actually calling MPI_Alltoallv_custom_s. 