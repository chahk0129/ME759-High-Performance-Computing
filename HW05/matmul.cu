#include "matmul.cuh"
#include <cuda.h>
#include <cstdio>

// template kernel function to support variable types
template <typename T>
__global__ void matmul(const T* A, const T* B, T* C, unsigned int n){
    extern __shared__ unsigned char shared_mem[]; // wrapper shared memory pointer for the template (instantiation needed for compilation)

    int bx = blockIdx.x; // matrix B sub-block column index
    int by = blockIdx.y; // matrix A sub-block row index

    int tx = threadIdx.x; // column index in sub-block
    int ty = threadIdx.y; // row index in sub-block

    int aBegin = n * blockDim.x * by; // index of the first sub-matrix of A processed by the block
    int aEnd = aBegin + n - 1; // index of the last sub-matrix of A processed by the block
    int aStep = blockDim.x; // step size used to iterate through sub-matrices of A

    int bBegin = blockDim.y * bx; // index of the first sub-matrix of B processed by the block
    int bStep = blockDim.y * n; // step size used to iterate through the sub-matrices of B

    int c = n * blockDim.y * by + blockDim.x * bx;
    int c_idx = c + n * ty + tx;
    if(c_idx > n*n-1)
	return;

    T* As = reinterpret_cast<T*>(shared_mem); // shared memory for matrix A
    T* Bs = reinterpret_cast<T*>(&As[blockDim.x * blockDim.y]); // shared memory for matrix B

    T Csub = 0; // element of the block sub-matrix that is computed by the thread
    for(int a=aBegin, b=bBegin; a<=aEnd; a+=aStep, b+=bStep){
	int a_idx = a + n * ty + tx; // idx for matrix A
	int b_idx = b + n * ty + tx; // idx for matrix B

	int s_idx = ty * blockDim.x + tx; // idx for shared array

	As[s_idx] = A[a_idx]; // load sub-matrix A to shared memory
	Bs[s_idx] = B[b_idx]; // load sub-matrix B to shared memory

	__syncthreads(); // synchronize to make sure the matrices are loaded

	for(int k=0; k<blockDim.x; k++){ // each thread computes one element of the block sub-matrix
	    int a_idx = ty * blockDim.y + k; // idx for matrix As
	    int b_idx = tx + blockDim.x * k; // idx for matrix Bs
	    Csub += As[a_idx] * Bs[b_idx]; // compute matrix multiplication
	}

	__syncthreads(); // synchronize to make sure the preceding computation is done before loading new sub-matrices of A and B
    }
	
    // write the block sub-matrix to global memory
    // each thread writes one element
    C[c_idx] = Csub;
}

__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n,
                       unsigned int block_dim){
    // calculate the number of blocks needed
    int block_num = (n + block_dim - 1) / block_dim;

    // get the block and grid dimension
    dim3 dim_block(block_dim, block_dim);
    dim3 dim_grid(block_num, block_num);

    // shared memory size
    size_t size = sizeof(int) * block_dim * block_dim * 2;
    // call kernel function with int type
    matmul<int><<<dim_grid, dim_block, size>>>(A, B, C, n);
    // synchronize to make sure every grid and block is completed
    cudaDeviceSynchronize();
}

__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n,
                       unsigned int block_dim){
    // calculate the number of blocks needed
    int block_num = (n + block_dim - 1) / block_dim;

    // get the block and grid dimension
    dim3 dim_block(block_dim, block_dim);
    dim3 dim_grid(block_num, block_num);

    // shared memory size
    size_t size = sizeof(float) * block_dim * block_dim * 2;
    // call kernel function with int type
    matmul<float><<<dim_grid, dim_block, size>>>(A, B, C, n);
    // synchronize to make sure every grid and block is completed
    cudaDeviceSynchronize();
}

__host__ void matmul_3(const double *A, const double *B, double *C,
                       unsigned int n, unsigned int block_dim){
    // calculate the number of blocks needed
    int block_num = (n + block_dim - 1) / block_dim;

    // get the block and grid dimension
    dim3 dim_block(block_dim, block_dim);
    dim3 dim_grid(block_num, block_num);

    // shared memory size
    size_t size = sizeof(double) * block_dim * block_dim * 2;
    // call kernel function with int type
    matmul<double><<<dim_grid, dim_block, size>>>(A, B, C, n);
    // synchronize to make sure every grid and block is completed
    cudaDeviceSynchronize();
}
