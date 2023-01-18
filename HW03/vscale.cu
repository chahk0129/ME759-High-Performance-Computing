#include "vscale.cuh"

__global__ void vscale(float* a, float* b, int n){
    // calculate the location of an index based on block id, the number of threads per block and thread id
    int loc = blockIdx.x * blockDim.x + threadIdx.x;

    if(loc < n) // if the index location resides the array, overwrite b with the computed value
	b[loc] = a[loc] * b[loc];
}
