#include <cstring>
#include <chrono>
#include <iostream>
#include <mpi.h>

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv);

    // read input values from commandline
    size_t n = atol(argv[1]);

    // buffer initialization
    float* send_buf = new float[n];
    float* recv_buf = new float[n];

    srand(time(NULL)); // random generator
    for(size_t i=0; i<n; i++){ // initialize buffers with values ranging from 0 to 1
	send_buf[i] = (float)rand() / RAND_MAX;
	recv_buf[i] = (float)rand() / RAND_MAX;
    }

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size); // mpi processes
    if(size != 2){ // we are using 2 processes
	std::cout << size << " MPI processes are used! --- Use 2." << std::endl;
	MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // mpi rank
    MPI_Status status; // mpi status when communication
    double t0 = 0;
    double t1 = 0;

    // peer is assigned based on its rank 
    int peer = (rank == 0) ? 1 : 0;
    int tag = 0; // 0 for send and recv tag 

    if(rank == 0){
	auto start = std::chrono::high_resolution_clock::now(); // measure start time
	MPI_Send(send_buf, n, MPI_FLOAT, peer, tag, MPI_COMM_WORLD); // post send to rank 1
	MPI_Recv(recv_buf, n, MPI_FLOAT, peer, tag, MPI_COMM_WORLD, &status); // post receive to rank 1
	auto end = std::chrono::high_resolution_clock::now(); // measure end time
	t0 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count(); // elapsed time in milliseconds

	MPI_Recv(&t1, 1, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD, &status); // post receive for the elapsed time to rank 1
	std::cout << t0 + t1 << std::endl; // print the elapsed time
    }
    else if(rank == 1){
	auto start = std::chrono::high_resolution_clock::now(); // measure start time
	MPI_Recv(recv_buf, n, MPI_FLOAT, peer, tag, MPI_COMM_WORLD, &status); // post receive to rank 0
	MPI_Send(send_buf, n, MPI_FLOAT, peer, tag, MPI_COMM_WORLD); // post send to rank 0
	auto end = std::chrono::high_resolution_clock::now(); // measure end time
	t1 = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count(); // elapsed time in milliseconds

	MPI_Send(&t1, 1, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD); // post send of the elapsed time to rank 0
    }


    delete[] send_buf;
    delete[] recv_buf;

    MPI_Finalize();

    return 0;
}
