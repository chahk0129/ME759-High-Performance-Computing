#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <cuda.h>

__global__ void compute(int* dA, int a){
    // calculate the index location for the array dA
    int loc = blockIdx.x * blockDim.x + threadIdx.x;

    // do the simple math calculation
    dA[loc] = blockIdx.x * a + threadIdx.x;
}

int main(){
    int size = 16; // array size
    int num_block = 2; // use two blocks
    int num_thread = 8; // use 8 threads
    int a = rand() % 101; // random number ranging from 0 - 100

    int* hA = new int[size]; // host array allocation
    int* dA;
    cudaMalloc(&dA, sizeof(int)*size); // device array allocation
    compute<<<num_block, num_thread>>>(dA, a); // call gpu kernel function
    cudaDeviceSynchronize(); // synchornize

    // copy the values of device array to host array
    cudaMemcpy(hA, dA, sizeof(int)*size, cudaMemcpyDeviceToHost);

    // print the output
    for(int i=0; i<size; i++){
	if(i < size-1)
	    std::cout << hA[i] << " ";
	else
	    std::cout << hA[i] << std::endl;
    }

    delete[] hA; // free host array
    cudaFree(&dA); // free device array
    return 0;
}

