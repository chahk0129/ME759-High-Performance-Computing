#include "reduce.cuh"
#include <cuda.h>

#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]){
    // read arguments from a commandline
    unsigned int N = (unsigned int)atoi(argv[1]);
    unsigned int threads_per_block = (unsigned int)atoi(argv[2]);

    // memory allocation for host arrays (input, output) with size of N elements
    float* input = new float[N];
    float* output = new float[N];

    // initialize input array with random numbers in the rage [-1,1]
    srand(time(NULL));
    for(unsigned int i=0; i<N; i++){
	input[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }

    // cuda event initialization to measure the elapsed time
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // record the start time
    cudaEventRecord(start);
    // call reduce function
    reduce(&input, &output, N, threads_per_block);
    // record the end time
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed = 0;
    // get the elapsed time with the recorded event times
    cudaEventElapsedTime(&elapsed, start, end);

    // print the resulting sum
    std::cout << input[0] << std::endl;
    // print the elapsed time
    std::cout << elapsed << std::endl;

    // destroy the created events
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    // free allocated memory for arrays 
    delete[] input;
    delete[] output;

    return 0;
}
