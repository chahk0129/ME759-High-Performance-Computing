#include "optimize.h"
#include <iostream>
#include <cstdlib>
#include <chrono>

double my_func(vec* v, data_t& dest, int type){  // runs optimizeX function
    auto start = std::chrono::high_resolution_clock::now(); // start timing
    // call optimizeX function based on the function type variable
    if(type == 1)
	optimize1(v, &dest);
    else if(type == 2)
	optimize2(v, &dest);
    else if(type == 3)
	optimize3(v, &dest);
    else if(type == 4)
	optimize4(v, &dest);
    else if(type == 5)
	optimize5(v, &dest);
    else{
	std::cerr << "not supported function - optimize" << type << std::endl;
	exit(0);
    }
    auto end = std::chrono::high_resolution_clock::now(); // end timing
    double elapsed = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    return elapsed; // return elapsed time
}


int main(int argc, char* argv[]){
    size_t n = atol(argv[1]); // get input from commandline
    int func_num = 5;

    vec* v = new vec(n);
    data_t* d = new data_t[n];

    srand(time(NULL)); // random generator
    for(size_t i=0; i<n; i++) // initialize data with all 1
	d[i] = 1;

    v->data = d; // store data pointer in the vector

    int repeat = 10;
    for(int i=1; i<=func_num; i++){ // iterate over optimizeX functions
	double elapsed = 0;
	data_t dest;

	for(int j=0; j<repeat; j++) // run for 10 times
	    elapsed += my_func(v, dest, i);

	std::cout << dest << std::endl; // print dest
	std::cout << elapsed/repeat << std::endl; // print elapsed time
    }

    delete v;
    delete[] d; // cleanup
    return 0;
}

