#include <iostream>
#include <random>
#include <chrono>
#include "scan.h"

int main(int argc, char* argv[]){
    size_t n = atoi(argv[1]);
    float* arr = new float[n];
    float* output = new float[n];

    srand(time(NULL));
    for(size_t i=0; i<n; i++){
	arr[i] = ((float)rand() / RAND_MAX) * 2 - 1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    scan(arr, output, n);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    std::cout << duration.count() << std::endl;

    std::cout << output[0] << std::endl;
    std::cout << output[n-1] << std::endl;

    delete[] arr;
    delete[] output;

    return 0;
}


