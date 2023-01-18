#include "stencil.cuh"
#include <cuda.h>
#include <cstdio>

__global__ void stencil_kernel(const float* image, const float* mask, float* output, unsigned int n, unsigned int R){
    // calculate specific index location based on current block index, block dimenstion, and thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x; 
    
    // dynamic shared memory
    extern __shared__ float shared_mem[];
    float* shared_mask = shared_mem; // shared mask points the start of shared memory
    float* shared_image = &shared_mask[R*2 + 1]; // shared image points to the end of shared_mask+1
    float* shared_output = &shared_image[blockDim.x + R*2]; // shared output points to the start of each block dimension 

    if(threadIdx.x == 0){ // every first thread in the block 

	for(int i=0; i<R*2+1; i++) // reads the entire mask to shared memory
	    shared_mask[i] = mask[i];

	for(int i=0; i<blockDim.x; i++) // set its shared output to zeroes
	    shared_output[i] = 0;
	

	for(int i=0; i<R*2+1; i++){ 
	    int image_idx = idx - R + i;
	    if((image_idx < 0) || (image_idx > n-1)) // if out of range, set shared image to 1
		shared_image[i] = 1;
	    else // else read the image to shared memory
		shared_image[i] = image[idx - R + i];
	}
    }
    else if(threadIdx.x == blockDim.x - 1){ // last thread in each block
	for(int i=0; i<R*2+1; i++){
	    int image_idx = idx + i;
	    if((image_idx) < 0 || (image_idx > n-1)) // if out of range, set shared image to 1
		shared_image[i + R + blockDim.x - 1] = 1;
	    else // else read the image to shared memory
		shared_image[i + R + blockDim.x - 1] = image[image_idx];
	}
    }
    else // threads other than the first or last in each block, read image to shared memory
	shared_image[threadIdx.x + R] = image[idx];
	
    // synchronize threads until all the image values and masks are read into the shared memory
    __syncthreads(); 

    for(int j=0; j<R*2+1; j++){
	int image_idx = threadIdx.x + j;
	shared_output[threadIdx.x] += shared_image[image_idx] * shared_mask[j]; // compute the output based on the convolution formula
    }

    // store the computed shared output to the output device memory
    output[idx] = shared_output[threadIdx.x];
}

__host__ void stencil(const float* image,
                      const float* mask,
                      float* output,
                      unsigned int n,
                      unsigned int R,
                      unsigned int threads_per_block){
    // calculate the number of blocks needed based on the size n and the number of threads per block
    int num_blocks = n / threads_per_block; 
    if(n % threads_per_block != 0) // if there's a remainder, we need one more block 
	num_blocks++;

    // calculate the size of shared memory
    size_t shmem_size = sizeof(float)*(R*2+1) + sizeof(float)*(R*2 + threads_per_block) + sizeof(float)*threads_per_block;

    // call the kernel function with the indicated size of shared memory
    stencil_kernel<<<num_blocks, threads_per_block, shmem_size>>>(image, mask, output, n, R);
    cudaDeviceSynchronize(); // synchornize
}
