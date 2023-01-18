#include <iostream>
#include <chrono>
#include "convolution.h"

int main(int argc, char* argv[]){
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);

    float* image = new float[n*n];
    float* mask = new float[m*m];
    float* output = new float[n*n];

    srand(time(NULL));
    for(int i=0; i<n; i++){
	for(int j=0; j<n; j++){
	    image[i*n + j] = ((float)rand() / RAND_MAX) * 20 - 10;
	}
    }

    for(int i=0; i<m; i++){
	for(int j=0; j<m; j++){
	    mask[i*m + j] = ((float)rand() / RAND_MAX) * 2 - 1;
	}
    }

    auto start = std::chrono::high_resolution_clock::now();
    convolve(image, output, n, mask, m);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
    std::cout << duration.count() << std::endl;
    std::cout << output[0] << std::endl;
    std::cout << output[n*n - 1] << std::endl;

    delete[] output;
    delete[] mask;
    delete[] image;
    return 0;
}
