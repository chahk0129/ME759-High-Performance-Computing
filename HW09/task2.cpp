#include <cstring>
#include <chrono>
#include <limits>
#include <iostream>
#include "montecarlo.h"

int main(int argc, char* argv[]){
    // read input values from commandline
    size_t n = atol(argv[1]);
    size_t t = atol(argv[2]);

    // x and y initialization
    float* x = new float[n];
    float* y = new float[n];

    float r = 1.0; // radius

    srand(time(NULL)); // random generator
    for(size_t i=0; i<n; i++){ // initialize x and y with values ranging from -r to r
	x[i] = ((float)rand() / RAND_MAX) * 2 - 1;
	y[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }

    omp_set_num_threads(t); // set the number of threads
    
    auto start = std::chrono::high_resolution_clock::now(); // measure start time
    auto pi = 4.0 * montecarlo(n, x, y, r) / n; // call montecarlo function
    auto end = std::chrono::high_resolution_clock::now(); // measure end time

    auto elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count(); // get elapsed time in millisecond

    std::cout << pi << std::endl; // print esimated pi
    std::cout << elapsed << std::endl; // print elapsed time

    delete[] x;
    delete[] y;

    return 0;
}
