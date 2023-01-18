#include <iostream>
#include <cstdio>
#include <cuda.h>

__global__ void cuda_factorial(){
    // calculate thread IDs to use it as the base number of factorial
    int tid = threadIdx.x + 1;
    int ret = 1;

    // calculate the value of factorial
    for(int i=1; i<=tid; i++){
	ret *= i;
    }

    // print the output
    std::printf("%d!=%d\n", tid, ret);
}

int main(){
    int num_block = 1;
    int num_thread = 8;

    // call gpu kernel function
    cuda_factorial<<<num_block, num_thread>>>();
    // synchronize 
    cudaDeviceSynchronize();
    return 0;
}
