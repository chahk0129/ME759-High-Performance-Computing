#include "mmul.h"
#include <cuda.h>
#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[]){
    // get parameters from commandline
    int n = atoi(argv[1]);
    int n_test = atoi(argv[2]);

    // matrix declaration
    float* A;
    float* B;
    float* C;

    // allocate memory for matrix A, B, and C
    cudaMallocManaged(&A, sizeof(float)*n*n);
    cudaMallocManaged(&B, sizeof(float)*n*n);
    cudaMallocManaged(&C, sizeof(float)*n*n);

    srand(time(NULL)); // random generator
    for(int i=0; i<n*n; i++){ // initialize matrix A and B with random numbers ranging from -1 to 1
	A[i] = ((float)rand() / RAND_MAX) * 2 - 1;
	B[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }

    float total_elapsed = 0; // total elapsed time
    for(int i=0; i<n_test; i++){ // iterate n_test times
	cudaMemset(C, 0, sizeof(float)*n*n); // set output matrix C to 0

	// event creation for timing measurement
	cudaEvent_t start, end; 
	cudaEventCreate(&start);
	cudaEventCreate(&end);

	cublasHandle_t handle;
	cublasCreate(&handle); // create cublas handle

	cudaEventRecord(start); // start recording time
	mmul(handle, A, B, C, n); // call mmul function
	cudaEventRecord(end); // stop recording time
	cudaEventSynchronize(end); 

	cublasDestroy(handle); // destroy cublas handle

	float elapsed = 0; 
	cudaEventElapsedTime(&elapsed, start, end); //get elapsed time for current iteration
	total_elapsed += elapsed; // add it to global elapsed time

	// destroy event for timing measurement
	cudaEventDestroy(start); 
	cudaEventDestroy(end);
    }

    // print the average elapsed time
    std::cout << total_elapsed / n_test << std::endl;

    // free allocated memory of matrix A, B and C
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}




