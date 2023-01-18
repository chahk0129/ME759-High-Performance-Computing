#define CUB_STDERR // print CUDA runtime errors to console
#include <stdio.h>
#include <cub/util_allocator.cuh>
#include <cub/device/device_reduce.cuh>
#include "cub/util_debug.cuh"
using namespace cub;
CachingDeviceAllocator  g_allocator(true);  // Caching allocator for device memory

int main(int argc, char* argv[]) {
    unsigned int n = (unsigned int)atoi(argv[1]); // read n from commandline

    // Set up host arrays
    float* h_in = new float[n];

    srand(time(NULL)); // random generator
    for(unsigned int i=0; i<n; i++) // initialize h_in array with random numbers ranging from -1 to 1
	h_in[i] = ((float)rand() / RAND_MAX) * 2 - 1;

    /*
    // calculate manual sum to compare it with the result we will get later with device_reduce function
    float sum = 0;
    for (unsigned int i = 0; i < n; i++)
        sum += h_in[i];
	*/

    // Set up device arrays
    float* d_in = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_in, sizeof(float) * n));

    // Initialize device input
    CubDebugExit(cudaMemcpy(d_in, h_in, sizeof(float) * n, cudaMemcpyHostToDevice));

    // Setup device output array
    float* d_sum = NULL;
    CubDebugExit(g_allocator.DeviceAllocate((void**)& d_sum, sizeof(float) * 1));

    // Request and allocate temporary storage
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n));
    CubDebugExit(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

    // event creation for timing measurement
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Do the actual reduce operation
    cudaEventRecord(start); // start recording time
    CubDebugExit(DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_sum, n)); // call sum function
    cudaEventRecord(end); // end recording time
    cudaEventSynchronize(end);

    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, end); // get elapsed time in milliseconds
    cudaEventDestroy(start); // destroy event for timing measurement
    cudaEventDestroy(end);

    float gpu_sum;
    CubDebugExit(cudaMemcpy(&gpu_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

    /*
    // Check for correctness
    printf("\t%s\n", (gpu_sum == sum ? "Test passed." : "Test falied."));
    printf("%f\n", sum);
    */

    printf("%f\n", gpu_sum); // print result
    printf("%f\n", elapsed); // print elapsed time

    // Cleanup
    if (d_in) CubDebugExit(g_allocator.DeviceFree(d_in));
    if (d_sum) CubDebugExit(g_allocator.DeviceFree(d_sum));
    if (d_temp_storage) CubDebugExit(g_allocator.DeviceFree(d_temp_storage));
    
    return 0;
}
