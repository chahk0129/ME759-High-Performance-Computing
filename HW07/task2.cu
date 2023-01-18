#include "count.cuh"
#include <cuda.h>
#include <iostream>
#include <thrust/host_vector.h>

int main(int argc, char* argv[]){
    int n = atoi(argv[1]); // read n from commandline

    thrust::host_vector<int> h_in(n); // host vector of size n
    srand(time(NULL)); // random generator

    int max = 501;
    for(int i=0; i<n; i++) // initialize vector h_in with random numbers ranging from 0 to 500
	h_in[i] = rand() % max;

    thrust::device_vector<int> d_in = h_in; // copy h_in to d_in (device vector)
    thrust::device_vector<int> values(n); // device vector of values (size of n)
    thrust::device_vector<int> counts(n); // device vector of counts (size of n)

    // event creation for timing measurement
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start); // start recording time
    count(d_in, values, counts); // call count function
    cudaEventRecord(end); // stop recodring time
    cudaEventSynchronize(end);
    

    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, end); // get elapsed time in milliseconds
    
    // destroy event for timing measurement
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    std::cout << values.back() << std::endl; // print last element of values
    std::cout << counts.back() << std::endl; // print last element of counts
    std::cout << elapsed << std::endl; // print elapsed time

    return 0;
}
