#include <stdio.h>

int exclusiveScan();

int main(){

	exclusiveScan();

	return 0;

}

int exclusiveScan(){
	int N = 11;

	int output[11] = {1,3,5,7,9,11,13,15,17,19, 21};

	for(int twod =1; twod < N; twod*= 2){
		int twod1 	= twod*2;
		printf("\nTwod: %d\t", twod);
		printf("Twod1: %d\n", twod1);

		for(int i = 0; i< N; i += twod1){
			output[i+twod1-1]	+= output[i+twod-1];

			printf("i: %d\t", i);
			printf("output[%d]: %d\t", (i+twod1-1), output[i+twod1-1]);
			printf("output[%d]: %d\n", (i+twod-1), output[i+twod-1]);
		}
		printf("==========================================================\n");

	}


	output[N-1]	=0;
	printf("%d\n", output[N-1]);
	for(int i = 0; i < N; i++){
    	printf("output[%d]:\t%d\n",i, output[i]);
    }

	for (int twod = N/2; twod >= 1; twod /= 2)
    {
     	int twod1 = twod*2;
     	printf("\n\nTwod: %d\t", twod);
		printf("Twod1:%d\n", twod1);
		 for(int i = 0; i < N; i += twod1)
		 {	
		 	printf("i: %d\t", i);
		     int t = output[i+twod-1];
		      printf("t: %d\t", t);
		     printf("output[%d]: %d\n", (i+twod-1), output[i+twod-1]);
		     output[i+twod-1] = output[i+twod1-1];
		    printf("\t\toutput[%d]: %d\t", (i+twod-1), output[i+twod-1]);
			printf("output[%d]: %d\n", (i+twod1-1), output[i+twod1-1]);
		     output[i+twod1-1] += t;

		    
		    printf("\tt: %d\t", t);
			printf("output[%d]: %d\n", (i+twod1-1), output[i+twod1-1]);
			//printf("output[%d]: %d\n", (i+twod-1), output[i+twod-1]);
		 }

		 for(int i = 0; i < N; i++){
    	printf("output[%d]:\t%d\n",i, output[i]);
    }
		printf("==========================================================\n");
    }


    for(int i = 0; i < N; i++){
    	printf("output[%d]:\t%d\n",i, output[i]);
    }

    return 0;
}

