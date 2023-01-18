#include "matmul.h"
#include <cstring>

void mmul(const float* A, const float* B, float* C, const std::size_t n){
    memset(C, 0, sizeof(float) * n*n); // set C zeros

    // this is the parallel version of the mmul2 function in HW02
    // parallelize two nested loops
#pragma omp parallel for collapse(2)
    for(size_t i=0; i<n; i++){
	for(size_t k=0; k<n; k++){
	    for(size_t j=0; j<n; j++){
		C[i*n + j] += A[i*n + k] * B[k*n + j];
	    }
	}
    }
}
