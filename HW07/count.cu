#include "count.cuh"
#include <iostream>

void count(const thrust::device_vector<int>& d_in,
                 thrust::device_vector<int>& values,
                 thrust::device_vector<int>& counts){

    thrust::device_vector<int> d_vec(int(d_in.size())); // create a device vector of the same size with d_in
    thrust::fill(d_vec.begin(), d_vec.end(), 1); // fill the vector with 1

    thrust::device_vector<int> d_val = d_in; // copy values from d_in to d_val
    thrust::sort(d_val.begin(), d_val.end()); // sort the vector

    // call reduce by key to collect the unique values and counts
    auto ret = thrust::reduce_by_key(d_val.begin(), d_val.end(), d_vec.begin(), values.begin(), counts.begin());

    // resize values and counts arrays to the actual reduced elements
    values.resize(ret.first - values.begin());
    counts.resize(ret.second - counts.begin());
}

