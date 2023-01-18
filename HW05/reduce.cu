#include "reduce.cuh"
#include <cuda.h>

__global__ void reduce_kernel(float *g_idata, float *g_odata, unsigned int n){
    // dynamic shared memory
    extern __shared__ float sdata[];

    // each thread loads elements from global to shared memory
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    if((idx + blockDim.x) < n) // first add during load
	sdata[tid] = g_idata[idx] + g_idata[idx + blockDim.x]; 
    else if(idx < n) // load
	sdata[tid] = g_idata[idx];
    else // exceeding the range
	sdata[tid] = 0;
    __syncthreads(); // synchronize threads to make sure input data is successfully loaded to shared memory

    // reversed loop and threadID-based indexing
    for(int s=blockDim.x/2; s>0; s>>=1){ // iterate over half of the previous size, e.g., 1/2, 1/4, 1/8, ...
	if(tid < s) // if within range, reduce
	    sdata[tid] += sdata[tid + s];
	__syncthreads(); // synchronize to make sure each reduce is complete before beginning the next iteration
    }

    if(tid == 0) // every first thread in each block writes the summed value to the output array in its block index
	g_odata[blockIdx.x] = sdata[0];
}

__host__ void reduce(float **input, float **output, unsigned int N,
                     unsigned int threads_per_block){
    float* g_idata;
    float* g_odata;

    // memory allocation for input and output array
    cudaMalloc(&g_idata, sizeof(float) * N);
    cudaMalloc(&g_odata, sizeof(float) * (N + threads_per_block - 1) / threads_per_block);
    // copy the input array from host to device memory 
    cudaMemcpy(g_idata, *input, sizeof(float)*N, cudaMemcpyHostToDevice);

    // iterate each block
    for(int i=N; i>1; i=(i+threads_per_block-1)/threads_per_block){
	// calculate the number of block nums for this iteration
	int block_num = (i + threads_per_block - 1) / threads_per_block;
	
	int size = sizeof(float) * threads_per_block; // shared memory size
	reduce_kernel<<<block_num, threads_per_block, size>>>(g_idata, g_odata, i); // call the kernel function

	// copy the output to input data to process the next iteration with the reduced input
	cudaMemcpy(g_idata, g_odata, sizeof(float) * block_num, cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize(); // synchronize to make sure all the kernel finish

    cudaMemcpy(*input, g_idata, sizeof(float), cudaMemcpyDeviceToHost); // copy the last sum to the first element of input
    cudaFree(g_idata); // free the input device memory
    cudaFree(g_odata); // free the output device memory
}

