// Author: Nic Olsen, Jason Zhou

#ifndef MATMUL_CUH
#define MATMUL_CUH

// You should implement Tiled Matrix Multiplication discussed in class
// Compute the matrix product C = AB.
// A, B, and C are row-major representations of nxn matrices in 'managed
// memory'. Configures the kernel call using a 2D configuration with blocks of
// dimensions block_dim x block_dim. The function should end in a call to
// cudaDeviceSynchronize for timing purposes.

// HINT: think about why we put '__host__' here
// matmul_1, matmul_2, and matmul_3 are doing the same thing except input data
// types, then what is the best way to handle it?

// You DO NOT have to follow the hint and you can do anything you what
// as long as you DO NOT add additional files, you DO NOT modify this header
// file, and your code CAN compile with provided compile command.
__host__ void matmul_1(const int *A, const int *B, int *C, unsigned int n,
                       unsigned int block_dim);
__host__ void matmul_2(const float *A, const float *B, float *C, unsigned int n,
                       unsigned int block_dim);
__host__ void matmul_3(const double *A, const double *B, double *C,
                       unsigned int n, unsigned int block_dim);

#endif