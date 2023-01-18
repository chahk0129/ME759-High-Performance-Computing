#include "matmul.h"
#include <cstdlib>
#include <iostream>
#include <cstring>

int main(int argc, char* argv[]){
    // read size of matrix and number of threads from commandline
    size_t n = atol(argv[1]);
    int t = atoi(argv[2]);

    // matrix allocation
    float* A = new float[n*n];
    float* B = new float[n*n];
    float* C = new float[n*n];

    srand(time(NULL)); // random generator
    for(size_t i=0; i<n*n; i++){ // range is not given in the instruction, we use the same range [-1,1]
	A[i] = ((float)rand() / RAND_MAX) * 2 - 1; 
	B[i] = ((float)rand() / RAND_MAX) * 2 - 1; 
    }

    // set the number of threads
    omp_set_num_threads(t);

    auto start = omp_get_wtime(); // measure the start time
    mmul(A, B, C, n); // call mmul function
    auto end = omp_get_wtime(); // measure the end time

    auto elapsed = (end - start) * 1000; // elapsed time in milliseconds

    std::cout << C[0] << std::endl; // print the first element
    std::cout << C[n*n-1] << std::endl; // print the last element
    std::cout << elapsed << std::endl; // print the elapsed time

    /*
    // test
    float _A[n*n];
    float _B[n*n];
    float _C[n*n];
    memcpy(_A, A, sizeof(float)*n*n);
    memcpy(_B, B, sizeof(float)*n*n);
    memset(_C, 0, sizeof(float)*n*n);
    for(size_t i=0; i<n; i++){
	for(size_t j=0; j<n; j++){
	    for(size_t k=0; k<n; k++){
		_C[i*n + j] += _A[i*n + k] * _B[k*n + j];
	    }
	}
    }
    bool equal = true;
    for(size_t i=0; i<n*n; i++){
	if(_C[i] != C[i]){
	    std::cout << "not equal at " << i << "C(" << C[i] << "), _C(" << _C[i] << ")" << std::endl;
	    equal = false;
	    break;
	}
    }

    if(equal)
	std::cout << "equal" << std::endl;
	*/

    // free the matrices
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
