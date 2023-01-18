#include "msort.h"
#include <cstdlib>
#include <iostream>
#include <cstring>
#include <algorithm>

int main(int argc, char* argv[]){
    // read input from commandline
    size_t n = atol(argv[1]);
    size_t t = atoi(argv[2]);
    size_t ts = atoi(argv[3]);

    // array allocation
    int* arr = new int[n];

    srand(time(NULL)); // random generator
    for(size_t i=0; i<n; i++) // assign array elements with values ranging [-1000,1000]
	arr[i] = (rand() % 2000) - 1000;

    /*
    // test purpose
    int temp[n];
    memcpy(temp, arr, sizeof(int)*n);
    */

    // set the number of threads
    omp_set_num_threads(t);
    omp_set_nested(1);

    auto start = omp_get_wtime(); // measure the start time
    msort(arr, n, ts); // call msort function
    auto end = omp_get_wtime(); // measure the end time

    auto elapsed = (end - start) * 1000; // elapsed time in milliseconds

    std::cout << arr[0] << std::endl; // print the first element
    std::cout << arr[n-1] << std::endl; // print the last element
    std::cout << elapsed << std::endl; // print the elapsed time

    /*
    // test function
    bool equal = true;
    std::sort(temp, temp+n);
    for(size_t i=0; i<n; i++){
	if(temp[i] != arr[i]){
	    equal = false;
	    break;
	}
    }
    if(!equal)
	std::cout << "something wrong -- not equal" << std::endl;
    else
	std::cout << "equal" << std::endl;
	*/

    // free the array
    delete[] arr;

    return 0;
}
