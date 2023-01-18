#include <iostream>
#include <cstdio>

int main(int argc, char* argv[]){
    // a) takes an argument as a variable
    int n = atoi(argv[1]);

    // b) print 0 to n in ascending order with printf
    for(int i=0; i<=n; i++){
	printf("%d", i);
	if(i < n)
	    printf(" ");
	else
	    printf("\n");
    }

    // c) print n to 0 in descending order with std::cout
    for(int i=n; i>=0; i--){
	std::cout << i;
	if(i > 0)
	    std::cout << " ";
	else
	    std::cout << "\n";
    }

    return 0;
}
