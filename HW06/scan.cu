#include "scan.cuh"
#include <cuda.h>
#include <cstdio>

// hillis_steele kernel function
__global__ void hs_scan(float* g_idata, float* g_odata, float* g_bdata, unsigned int n, bool is_null){
    extern volatile __shared__ float s[]; // shared data
    int t_idx = threadIdx.x;
    int t_len = blockDim.x;
    int b_idx = blockIdx.x;
    int idx = b_idx * t_len + t_idx; // calculate global index based on block number, thread number, thread idx
    int pout = 0, pin = 1;

    // load input into shared memory
    if(idx < n)
	s[t_idx] = g_idata[idx];
    else
	return;

    // synchronize to make sure all data is loaded to shared memory
    __syncthreads();

    for(int offset=1; offset<t_len; offset*=2){
	pout = 1 - pout; // swap double buffer indices
	pin = 1 - pout;

	if(t_idx >= offset)
	    s[pout * t_len + t_idx] = s[pin * t_len + t_idx] + s[pin * t_len + t_idx - offset];
	else
	    s[pout * t_len + t_idx] = s[pin * t_len + t_idx];
	__syncthreads(); // synchronize before doing the next iteration
    }

    if(pout * t_len + t_idx < t_len) // write output
	g_odata[idx] = s[pout * n + t_idx];

    if(is_null && (t_idx == t_len-1)) // add block data
	g_bdata[b_idx] = s[pout * n + t_idx];
}

// inclusive add kernel function
__global__ void inclusive_add(float* g_od, float* g_os, unsigned int n){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if((idx < n) && (blockIdx.x != 0)) // inclusive add
	g_od[idx] += g_os[blockIdx.x - 1];
}


__host__ void scan(const float* input, float* output, unsigned int n, unsigned int threads_per_block){
    // declare input and output arrays
    float* g_id;
    float* g_od;

    // device memory allocation
    cudaMalloc(&g_id, sizeof(float)*n);
    cudaMalloc(&g_od, sizeof(float)*n);

    // copy input to device input array, and set output to zeros
    cudaMemcpy(g_id, input, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaMemset(g_od, 0, sizeof(float)*n);

    // declare intermediate arrays where sum of each scan iteration will be stored
    float* g_is;
    float* g_os;
    float* g_em;

    // calculate the number of blocks needed 
    int num_block = (n + threads_per_block - 1) / threads_per_block;
    // device memory allocation for the intermediate arrays
    cudaMalloc(&g_is, sizeof(float)*num_block);
    cudaMalloc(&g_os, sizeof(float)*num_block);
    cudaMalloc(&g_em, sizeof(float)*num_block);

    int size = sizeof(float) * threads_per_block * 2; // shared memory size
    // call hillis-steele function
    hs_scan<<<num_block, threads_per_block, size>>>(g_id, g_od, g_is, n, true);
    hs_scan<<<1, threads_per_block, size>>>(g_is, g_os, g_em, num_block, false);

    // do the inclusive add for the intermediate sums
    inclusive_add<<<num_block, threads_per_block>>>(g_od, g_os, n);
    cudaDeviceSynchronize(); // synchronize to make sure all the kernel functions are complete

    // copy the overall sum data to output array
    cudaMemcpy(output, g_od, sizeof(float)*n, cudaMemcpyDeviceToHost);

    // free allocated device memory
    cudaFree(g_id);
    cudaFree(g_od);
    cudaFree(g_is);
    cudaFree(g_os);
    cudaFree(g_em);
}

