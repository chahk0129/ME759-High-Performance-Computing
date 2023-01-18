#include "matmul.cuh"
#include <cuda.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, size_t n){
    // calculate idx with blockIdx, blockDim and threadIdx
    int loc = blockIdx.x * blockDim.x + threadIdx.x;
    if(loc > n*n-1) // if it is out of range, return
	return;

    int i = loc / n; // calculate row
    int j = loc % n; // calculate column
    float ret = 0;
    for(size_t k=0; k<n; k++){ // calculate the value of matrix multiplication by iterating each row of array A and column of array B
	ret += A[i*n + k] * B[k*n + j];
    }
    C[i*n + j] = ret; // write the output to the array C
}


void matmul(const float* A, const float* B, float* C, size_t n, unsigned int threads_per_block){
    int num_block = n*n / threads_per_block; // calculate the number of blocks needed for the computation
    if(n*n % threads_per_block != 0) // if the remainder is not zero, we need one more block for the multiplication for the last row of A with the last column B
	num_block++;

    matmul_kernel<<<num_block, threads_per_block>>>(A, B, C, n); // call gpu kernel function
    cudaDeviceSynchronize(); // synchronize the kernel function calls
}
