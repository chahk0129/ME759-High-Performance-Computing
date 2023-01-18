#include "convolution.h"
#include <cstring>

void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m){
    memset(output, 0, sizeof(float) * n*n); // set output to 0

    // parallelize two nested loops
#pragma omp parallel for collapse(2)
    for(size_t x=0; x<n; x++){ // image row iteration
	for(size_t y=0; y<n; y++){ // image column iteration
	    for(size_t i=0; i<m; i++){ // mask row iteration
		for(size_t j=0; j<m; j++){ // mask column iteration
		    size_t image_i = x + i - (m-1)/2; // calculate row index of image
		    size_t image_j = y + j - (m-1)/2; // calculate column index of image
		    float val = 0;
		    if(image_i >= 0 && image_i < n && image_j >= 0 && image_j < n) // if it lies within the boundary, get image value
			val = image[image_i*n + image_j];
		    else if((image_i >= 0 && image_i < n) || (image_j >= 0 && image_j < n)) // if one of the conditions is met, 1
			val = 1;
		    else // if none of the conditions are met, 0
			val = 0;

		    output[x*n + y] += mask[i*m + j] * val; // apply image and mask
		}
	    }
	}
    }
}
