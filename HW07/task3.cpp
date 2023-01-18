#include <iostream>
#include <omp.h>

void print(int tid){
    printf("I am thread No. %d\n", tid);
}

int factorial(int num){
    int ret = 1;
    for(int i=2; i<num+1; i++) // no need to start iteration from 1
	ret *= i;

    return ret;
}

int main(int argc, char* argv[]){
    int num_threads = 4; // 4 threads
    omp_set_num_threads(num_threads); // set openmp number of threads

    // print the total number of threads
    printf("Number of threads: %d\n", num_threads);
#pragma omp parallel
    {
	int tid = omp_get_thread_num(); // get the ID for current thread
	print(tid); // each thread prints its TID
    }

    int num = 8; // calculate up to 8 factorial
#pragma omp parallel for num_threads(num_threads) // need to use 4 threads to do the job
    for(int i=1; i<num; i+=2){
	int ret1 = factorial(i); // calculate ith factorial
	int ret2 = factorial(i+1); // calculate (i+1)th factorial
	printf("%d!=%d\n", i, ret1);  // print ith factorial
	printf("%d!=%d\n", i+1, ret2); // print (i+1)th factorial
    }

    return 0;
}
