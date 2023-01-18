#include <cstring>
#include <algorithm>
#include <chrono>
#include <limits>
#include <iostream>
#include "cluster.h"

int main(int argc, char* argv[]){
    // read input from commandline
    size_t n = atol(argv[1]);
    size_t t = atol(argv[2]);

    // allocate arrays
    float* arr = new float[n];
    float* centers = new float[t];
    float* dists = new float[t];

    srand(time(NULL)); // random generator
    for(size_t i=0; i<n; i++) // initialize arr with values ranging from 0 to n
	arr[i] = ((float)rand() / RAND_MAX) * n;
    std::sort(arr, arr+n); // sort arr

    for(size_t i=1; i<=t; i++) // initialize centers with values using the equation
	centers[i-1] = (2.0*i - 1) * n / (2.0 * t);

    omp_set_num_threads(t); // set the number of threads
    int iter = 1; // number of iterations
    double elapsed = 0;

    for(int i=0; i<iter; i++){ // iterate 10 times
	memset(dists, 0, sizeof(float) * t); //set zeros for each iteration to get the exact distance

	auto start = std::chrono::high_resolution_clock::now(); // measure start time
	cluster(n, t, arr, centers, dists); // call cluster function
	auto end = std::chrono::high_resolution_clock::now(); // measure end time
	elapsed += std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count(); // get elapsed time in millisecond
													       
    }

    float max_dist = std::numeric_limits<float>::min();
    size_t max_pos = 0;
    for(size_t i=0; i<t; i++){ // find the maximum distance and its partition ID
	if(max_dist < dists[i]){
	    max_dist = dists[i];
	    max_pos = i;
	}
    }
    std::cout << max_dist << std::endl; // print maximum distance
    std::cout << max_pos << std::endl; // print partition ID that has the maximum distance
    std::cout << elapsed/iter << std::endl; // print elapsed time

    delete[] arr;
    delete[] centers;
    delete[] dists;

    return 0;
}
