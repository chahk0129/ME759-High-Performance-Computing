#include "matmul.cuh"
#include <cuda.h>

#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]){
    // read arguments from a commandline
    int n = atoi(argv[1]);
    int threads_per_block = atoi(argv[2]);

    // matrix declaration
    float* A;
    float* B;
    float* C;

    // matrix memory allocation managed by unified memory system
    cudaMallocManaged(&A, sizeof(float) * n*n);
    cudaMallocManaged(&B, sizeof(float) * n*n);
    cudaMallocManaged(&C, sizeof(float) * n*n);
    
    // initialize matrix A and B with random numbers between -1 and 1
    srand(time(NULL));
    for(int i=0; i<n*n; i++){
	A[i] = ((float)rand() / RAND_MAX) * 2 - 1;
	B[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
    // initialize matrix C with 0
    memset(C, 0, sizeof(float)*n*n);

    // cuda event initialization to measure the elapsed time
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // record the start time
    cudaEventRecord(start);
    // call matmul function
    matmul(A, B, C, n, threads_per_block);
    // record the end time
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed = 0;
    // get the elapsed time with the recorded event times
    cudaEventElapsedTime(&elapsed, start, end);

    // print the last element
    std::cout << C[n*n-1] << std::endl;
    // print the elapsed time
    std::cout << elapsed << std::endl;

    // destroy the created events
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    // free allocated memory for matrices
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}
