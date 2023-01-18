#include "reduce.h"
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#include <mpi.h>

int main(int argc, char* argv[]){
    MPI_Init(&argc, &argv); // initialize mpi
    int n = atoi(argv[1]);
    int t = atoi(argv[2]);

    omp_set_num_threads(t); // set number of threads
    float* arr = new float[n]; // array

    srand(time(NULL)); // random generator
    for(int i=0; i<n; i++) // initialize arr with values [-1,1]
	arr[i] = ((float)rand() / RAND_MAX) * 2 - 1;

    int rank;
    int size;
    float global_res = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Barrier(MPI_COMM_WORLD); // barrier to complete array initialization in all processes

    auto start = std::chrono::high_resolution_clock::now();
    float res = reduce(arr, 0, n); // call reduce function to get local res
    MPI_Reduce(&res, &global_res, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD); // combine local res into global_res
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();

    if(rank == 0){
	std::cout << global_res << std::endl; // print result
	std::cout << elapsed << std::endl; // print elapsed time
    }

    delete[] arr; // cleanup
    MPI_Finalize();

    return 0;
}

