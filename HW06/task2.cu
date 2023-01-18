#include "scan.cuh"
#include <cuda.h>
#include <cstdlib>
#include <iostream>

int main(int argc, char* argv[]){
    // get parameters from commandline 
    unsigned int n = (unsigned int)atoi(argv[1]);
    unsigned int threads_per_block = (unsigned int)atoi(argv[2]);

    // input and output declaration
    float* input;
    float* output;

    // memory allocation of input and output array
    cudaMallocManaged(&input, sizeof(float)*n);
    cudaMallocManaged(&output, sizeof(float)*n);

    // random number generator
    srand(time(NULL));
    for(unsigned int i=0; i<n; i++){ // initialize input array with random numbers ranging from -1 to 1
	input[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }
    cudaMemset(output, 0, sizeof(float)*n); // set output array 0

    // event creation for time measurement
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start); // start recording time
    scan(input, output, n, threads_per_block); // call scan function
    cudaEventRecord(end); // stop recording time
    cudaEventSynchronize(end);

    float elapsed = 0; 
    cudaEventElapsedTime(&elapsed, start, end); // get the elapsed time in milliseconds

    // print out the last element and elapsed time
    std::cout << output[n-1] << std::endl;
    std::cout << elapsed << std::endl;

    // free allocated memory
    cudaFree(input);
    cudaFree(output);

    return 0;
}




