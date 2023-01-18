#include "mmul.h"

void mmul(cublasHandle_t handle, const float* A, const float* B, float* C, int n){
    // alpha and beta are constant 1
    const float alpha = 1;
    const float beta = 1;

    // call cublas matrix multiplication function
    //	no transpose
    //	number of rows and columns of matrix A, B, and C are all n
    //	scalar 1 (alpha and beta)
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, A, n, B, n, &beta, C, n);
    cudaDeviceSynchronize(); // synchronize to make sure the cublas function is complete
}
