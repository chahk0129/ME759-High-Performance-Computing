#include "matmul.cuh"
#include <cuda.h>

#include <typeinfo>
#include <iostream>
#include <cstdlib>


// overloaded functions for the wrapper of matmul function
// - int --> matmul_1
// - float --> matmul_2
// - double --> matmul_3

void matmul_func(int* A, int* B, int* C, unsigned int n, unsigned int block_dim){
    matmul_1(A, B, C, n, block_dim);
}

void matmul_func(float* A, float* B, float* C, unsigned int n, unsigned int block_dim){
    matmul_2(A, B, C, n, block_dim);
}

void matmul_func(double* A, double* B, double* C, unsigned int n, unsigned int block_dim){
    matmul_3(A, B, C, n, block_dim);
}

// template function that runs the test
template <typename T>
void test_func(unsigned int n, unsigned int block_dim){
    // declaration of matrix A, B, C
    T* A;
    T* B;
    T* C;

    // memory allocation for matrix A, B, C
    cudaMallocManaged(&A, sizeof(T) * n*n);
    cudaMallocManaged(&B, sizeof(T) * n*n);
    cudaMallocManaged(&C, sizeof(T) * n*n);

    // if the type is int, assign A and B with random numbers ranging from 0 to 1
    if(typeid(T) == typeid(int)){
	for(unsigned int i=0; i<n*n; i++){
	    A[i] = rand() % 2 - 1;
	    B[i] = rand() % 2 - 1;
	}
    }
    else if((typeid(T) == typeid(float)) || (typeid(T) == typeid(double))){ // if the type if float or double, assign A and B with random numbers ranging from -1 to 1
	for(unsigned int i=0; i<n*n; i++){
	    A[i] = ((T)rand() / RAND_MAX) * 2 - 1;
	    B[i] = ((T)rand() / RAND_MAX) * 2 - 1;
	}
    }
    else{ // invalid type
	std::cout << "Invalid type " << typeid(T).name() << " exiting the program ... " << std::endl;
	return;
    }

    // set C to zeros
    cudaMemset(C, 0, sizeof(T) * n*n);

    // cuda event initialization to measure the elapsed time
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // record the start time
    cudaEventRecord(start);
    // call an overloaded matmul function
    matmul_func(A, B, C, n, block_dim);
    // record the end time
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed = 0;
    // get the elapsed time with the recorded event times
    cudaEventElapsedTime(&elapsed, start, end);

    // print the first element of C
    std::cout << C[0] << std::endl;
    // print the last element of C
    std::cout << C[n*n-1] << std::endl;
    // print the elapsed time
    std::cout << elapsed << std::endl;

    // destroy the created events
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    // free allocated memory for matrix A, B and C
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}



int main(int argc, char* argv[]){
    // read arguments from a commandline
    unsigned int n = (unsigned int)atoi(argv[1]);
    unsigned int block_dim = (unsigned int)atoi(argv[2]);

    // call test function that uses int array
    test_func<int>(n, block_dim);

    // call test function that uses float array
    test_func<float>(n, block_dim);

    // call test function that uses double array
    test_func<double>(n, block_dim);

    return 0;
}
