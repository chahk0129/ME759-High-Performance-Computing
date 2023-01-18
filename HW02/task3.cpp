#include <iostream>
#include <chrono>
#include "matmul.h"

int main(int argc, char* argv[]){
    int n = 1024;
    double* A = new double[n*n];
    double* B = new double[n*n];
    double* C = new double[n*n];
    std::vector<double> _A, _B;
    _A.reserve(n*n);
    _B.reserve(n*n);
    srand(time(NULL));

    for(int i=0; i<n; i++){
	for(int j=0; j<n; j++){
	    A[i*n + j] = ((float)rand() / RAND_MAX) * 2 - 1;
	    _A[i*n + j] = A[i*n + j];
	    B[i*n + j] = ((float)rand() / RAND_MAX) * 2 - 1;
	    _B[i*n + j] = B[i*n + j];
	}
    }

    std::cout << n << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    mmul1(A, B, C, n);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    std::cout << duration.count() << std::endl;
    std::cout << C[n*n-1] << std::endl;

    start = std::chrono::high_resolution_clock::now();
    mmul2(A, B, C, n);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    std::cout << duration.count() << std::endl;
    std::cout << C[n*n-1] << std::endl;

    start = std::chrono::high_resolution_clock::now();
    mmul3(A, B, C, n);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    std::cout << duration.count() << std::endl;
    std::cout << C[n*n-1] << std::endl;

    start = std::chrono::high_resolution_clock::now();
    mmul4(_A, _B, C, n);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    std::cout << duration.count() << std::endl;
    std::cout << C[n*n-1] << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;
    _A.clear();
    _B.clear();
    _A.shrink_to_fit();
    _B.shrink_to_fit();
    return 0;
}
