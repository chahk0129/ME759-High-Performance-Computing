#include "convolution.h"
#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[]){
    // read size of matrix and number of threads from commandline
    size_t n = atol(argv[1]);
    int t = atoi(argv[2]);
    size_t m = 3; // mask size

    // matrix allocation
    float* image = new float[n*n];
    float* output = new float[n*n];
    float* mask = new float[m*m];

    srand(time(NULL)); // random generator
    for(size_t i=0; i<n; i++){ // initialize image with random numbers ranging from -10 to 10
	for(size_t j=0; j<n; j++){
	    image[i*n + j] = ((float)rand() / RAND_MAX) * 20 - 10;
	}
    }

    for(size_t i=0; i<m; i++){ // initialize mask with random numbers ranging from -1 to 1
	for(size_t j=0; j<m; j++){
	    mask[i*m + j] = ((float)rand() / RAND_MAX) * 2 - 1;
	}
    }

    // set the number of threads
    omp_set_num_threads(t);

    auto start = omp_get_wtime(); // measure the start time
    convolve(image, output, n, mask, m); // call convolve function
    auto end = omp_get_wtime(); // measure the end time

    auto elapsed = (end - start) * 1000; // elapsed time in milliseconds

    std::cout << output[0] << std::endl; // print the first element
    std::cout << output[n*n-1] << std::endl; // print the last element
    std::cout << elapsed << std::endl; // print the elapsed time

    // free the matrices
    delete[] image;
    delete[] output;
    delete[] mask;

    return 0;
}
