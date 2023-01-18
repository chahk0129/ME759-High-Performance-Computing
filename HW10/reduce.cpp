#include "reduce.h"

float reduce(const float* arr, const size_t l, const size_t r){
    float ret = 0;
#pragma omp parallel for simd reduction(+ : ret)
    for(size_t i=l; i<r; i++){
	ret += arr[i];
    }

    return ret;
}
