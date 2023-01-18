#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cuda.h>
#include "vscale.cuh"

int main(int argc, char* argv[]){
    int n = atoi(argv[1]); // get argument to use it as the size of array
    int num_thread = 512; // use 512 threads
    //int num_thread = 16; // use 16 threads
    int num_block = n/num_thread + 1; // calculate the number of blocks based on the number of threads

    auto _a = new float[n]; // allocate host array a
    auto _b = new float[n]; // allocate host array b

    // random numbers generation
    srand(time(NULL));
    for(int i=0; i<n; i++){
	_a[i] = ((float)rand() / RAND_MAX) * 20 - 10;
	_b[i] = ((float)rand() / RAND_MAX);
    }

    float* a;
    float* b;
    cudaMalloc(&a, sizeof(float)*n); // allocate device array a
    cudaMalloc(&b, sizeof(float)*n); // allocate device array b
    cudaMemcpy(a, _a, sizeof(float)*n, cudaMemcpyHostToDevice); // copy host array a to device array a
    cudaMemcpy(b, _b, sizeof(float)*n, cudaMemcpyHostToDevice); // copy host array b to device array b

    cudaEvent_t start, stop; // to measure time events
    cudaEventCreate(&start); // create events
    cudaEventCreate(&stop);

    cudaEventRecord(start); // measure the start time
    vscale<<<num_block, num_thread>>>(a, b, n); // call gpu kernel function
    cudaDeviceSynchronize(); // synchornize
    cudaEventRecord(stop); // measure the stop time
    cudaEventSynchronize(stop);

    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop); // calculate the elapsed time in msec


    auto ret = new float[n];
    cudaMemcpy(ret, b, sizeof(float)*n, cudaMemcpyDeviceToHost); // copy device array to host

    // print the output
    std::cout << elapsed << std::endl; 
    std::cout << ret[0] << std::endl;
    std::cout << ret[n-1] << std::endl;

    // destroy time measurement events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // free host arrays
    delete[] ret;
    delete[] _a;
    delete[] _b;

    // free device arrays
    cudaFree(&a);
    cudaFree(&b);

    return 0;
}

