#include <iostream>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

int main(int argc, char* argv[]){
    int n = atoi(argv[1]); // read n from commandline

    thrust::host_vector<float> h_vec(n); // host vector of size n

    srand(time(NULL)); // random generator
    for(int i=0; i<n; i++) // initialize host vector with random numbers ranging from -1 to 1
	h_vec[i] = ((float)rand() / RAND_MAX) * 2 - 1;

    thrust::device_vector<float> d_vec = h_vec; // device vector copied from host vector 

    // event creation for timing measurement
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start); // start recording time
    float ret = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0, thrust::plus<float>()); // call reduce function
    cudaEventRecord(end); // stop recording time
    cudaEventSynchronize(end);

    float elapsed = 0;
    cudaEventElapsedTime(&elapsed, start, end); // get elapsed time in milliseconds

    // destroy event for timing measurement
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    std::cout << ret << std::endl; // print reduced result
    std::cout << elapsed << std::endl; // print elapsed time

    /*
    float _ret = 0; // manual reduce for test purpose
    for(int i=0; i<n; i++)
	_ret += h_vec[i];
    std::cout << _ret << std::endl;
    */
    return 0;
}


