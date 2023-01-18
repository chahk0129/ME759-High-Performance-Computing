#include "stencil.cuh"
#include <cuda.h>

#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]){
    // read arguments from a commandline
    int n = atoi(argv[1]);
    int R = atoi(argv[2]);
    int threads_per_block = atoi(argv[3]);

    // array declaration
    float* image;
    float* mask;
    float* output;

    // array memory allocation managed by unified memory system
    cudaMallocManaged(&image, sizeof(float) * n);
    cudaMallocManaged(&mask, sizeof(float) * (R*2+1));
    cudaMallocManaged(&output, sizeof(float) * n);
    
    // initialize array image and mask with random numbers between -1 and 1
    srand(time(NULL));
    for(int i=0; i<n; i++)
	image[i] = ((float)rand() / RAND_MAX) * 2 - 1;

    for(int i=0; i<R*2+1; i++)
	mask[i] = ((float)rand() / RAND_MAX) * 2 - 1;


    // set output to zeros
    cudaMemset(output, 0, sizeof(float)*n);


    // cuda event initialization to measure the elapsed time
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // record the start time
    cudaEventRecord(start);
    // call stencil function
    stencil(image, mask, output, n, R, threads_per_block);
    // record the end time
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float elapsed = 0;
    // get the elapsed time with the recorded event times
    cudaEventElapsedTime(&elapsed, start, end);

    // print the last element
    std::cout << output[n-1] << std::endl;
    // print the elapsed time
    std::cout << elapsed << std::endl;

    // destroy the created events
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    // free allocated memory for arrays 
    cudaFree(image);
    cudaFree(mask);
    cudaFree(output);

    return 0;
}
