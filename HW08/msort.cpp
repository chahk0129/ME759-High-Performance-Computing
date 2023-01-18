#include "msort.h"
#include <algorithm>

// this is a function that performs the actual merge sort
void msort(int* arr, const std::size_t n, const std::size_t threshold, int num_threads){
    if(n <= 1) // base case
	return;

    // if the array size goes smaller than the threshold, sort it serially to avoid scheduling overhead
    if(n < threshold){
	for(auto it=arr; it!=arr+n; it++)
	    std::rotate(std::upper_bound(arr, it, *it), it, it+1);
	return;
    }

    if(num_threads == 1){ // task for one thread
	msort(arr, n/2, threshold, num_threads); // sort the first half
	msort(arr+n/2, n-n/2, threshold, num_threads); // sort the rest
    }
    else{ // more than one thread
#pragma omp task
	msort(arr, n/2, threshold, num_threads/2); // sort the first half
#pragma omp task
	msort(arr+n/2, n-n/2, threshold, num_threads-num_threads/2); // sort the rest
#pragma omp taskwait
    }

    std::inplace_merge(arr, arr+n/2, arr+n); // merge the two subarrays
}


void msort(int* arr, const std::size_t n, const std::size_t threshold){
#pragma omp parallel
#pragma omp single
    msort(arr, n, threshold, omp_get_num_threads());
}
