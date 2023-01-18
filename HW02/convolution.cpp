#include "convolution.h"
#include <cstring>


void convolve(const float *image, float *output, std::size_t n, const float *mask, std::size_t m){
    memset(output, 0, sizeof(float)*n*n);
    size_t _m = m / 2;

    for(size_t i=0; i<n; i++){
	for(size_t j=0; j<n; j++){
	    float ret = 0;
	    for(size_t _i=0; _i<m; _i++){
		for(size_t _j=0; _j<m; _j++){
		    size_t image_i = i + _i - _m;
		    size_t image_j = j + _j - _m;
		    float f = 0;

		    if(image_i >= 0 || image_i < n || image_j >= 0 || image_j < n){ // not corners
			f = image[image_i*n + image_j];
		    }

		    ret += f * mask[_i*m + _j];
		}
	    }
	    output[i*n + j] = ret;
	}
    }
}
