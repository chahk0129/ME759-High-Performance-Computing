#include "montecarlo.h"

int montecarlo(const size_t n, const float *x, const float *y, const float radius){
    int in_circle = 0;

#ifdef NO_SIMD // reduction without SIMD
#pragma omp parallel for reduction(+ : in_circle)
#else // use SIMD for the reduction
#pragma omp parallel for simd reduction(+ : in_circle)
#endif
    for(size_t i=0; i<n; i++){
	if(x[i]*x[i] + y[i]*y[i] < radius*radius) // if the point is inside the circle, increment the count
	    in_circle++;
    }

    return in_circle;
}
