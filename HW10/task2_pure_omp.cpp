#include "reduce.h"
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <omp.h>

int main(int argc, char* argv[]){
    int n = atoi(argv[1]); // get input from commandline
    int t = atoi(argv[2]);

    omp_set_num_threads(t); // set number of threads
    float* arr = new float[n]; // array

    srand(time(NULL)); // random generator
    for(int i=0; i<n; i++) // initialize arr with values [-1,1]
	arr[i] = ((float)rand() / RAND_MAX) * 2 - 1;

    auto start = std::chrono::high_resolution_clock::now();
    float res = reduce(arr, 0, n); // call reduce function to res
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();

    std::cout << res << std::endl; // print result
    std::cout << elapsed << std::endl; // print elapsed time

    delete[] arr; // cleanup

    return 0;
}

