#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include "CycleTimer.h"

extern float toBW(int bytes, float sec);


/* Helper function to round up to a power of 2. 
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}


__global__ void find_repeats_kernel(int* device_input, int length, int* binary_result)//, int* index_result)
{
	int index = (blockIdx.x * blockDim.x + threadIdx.x);
	if(index < length-1){
		 binary_result[index]=(device_input[index] == device_input[index+1]);
//		 printf("device_input[%d] = %d device_input[%d] = %d match = %d\n",index, device_input[index],index+1, device_input[index+1], binary_result[index]); 

        }
}

//device_input comes from binary_check_array
//length is the same as input array length
__global__ void find_matches_kernel(int* device_input, int length_match, int* binary_result)//, int* len_out)
{
	int index 				= (blockIdx.x * blockDim.x + threadIdx.x);
//	printf("find_matches_kernel::::Index = %d\n", index);
//	printf("Device input[%d] :%d, Device input [%d] : %d\n",index,device_input[index], index+1, device_input[index+1]); 
	if ((index < length_match)&&(device_input[index+1] > device_input[index]))
	{
	  binary_result[device_input[index+1]-1] = index; 
//	  printf("Device input[%d] :%d, Device input [%d] : %d\n",index,device_input[index], index+1, device_input[index+1]); 
	}
	//if (index == length_match-1)
	//{
	//  index_result[0] = binary_result[length_match-1];
	//}
}

__global__ void scan_kernel1 (long twod1, long twod, long N, int* device_result)
{
	long index 				= (blockIdx.x * blockDim.x + threadIdx.x)*twod1;
	if(index < N){
//		printf("Index: %d\t Length: %d\n", index, N);
		device_result[index + twod1 -1] += device_result[index + twod - 1]; 
//		printf("Device Result[%d] :%d\t, Device Result [%d] : %d\n", (index + twod1 -1), device_result[index+twod1-1], (index + twod -1), device_result[index+ twod-1]); 
	}
}

__global__ void scan_kernel2 (long twod1, long twod, long N, int* device_result)
{
	long index 		= (blockIdx.x * blockDim.x + threadIdx.x)*twod1;
	if(index < N){
//		printf("Index: %d\t Length: %d\n", index, N);
		int t					= device_result[index + twod -1];
		device_result[index + twod -1]		= device_result[index + twod1 -1];
		device_result[index + twod1 -1] 	+= t;
//		printf("Device Result[%d] :%d\t, Device Result [%d] : %d\n", (index + twod1 -1), device_result[index+twod1-1], (index + twod -1), device_result[index+ twod-1]); 
	}
}

__global__ void scan_kernel5 ( int N, int* device_result)
{
	int index 				= (blockIdx.x * blockDim.x + threadIdx.x);
	if(index < N){
//		printf("Index: %d\t Length: %d\n", index, N);
		device_result[index] = 1; 
//		printf("Device Result[%d] :%d\t, Device Result [%d] : %d\n", (index + twod1 -1), device_result[index+twod1-1], (index + twod -1), device_result[index+ twod-1]); 
	}
}

__global__ void scan_kernel_makezero(long length, int* device_result)
{
	int index		= length-1;
	device_result[index]	= 0;
}

void exclusive_scan(int* device_start, int  length, int* device_result)
{
    /* Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the input and output in device memory,
     * but this is host code -- you will need to declare one or more CUDA 
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the input and the output arrays are sized to accommodate the next
     * power of 2 larger than the input.
     */
  long i;
  {for( i=1;i<length;i*=2);
  length=i;}

  //printf("\n****length = %d****\n",length);	

  int threadsperBlock=512;

//  printf("Entered Scan Function\n");
  
/*
  for (int j = 0; j<length; j++)
  {
    printf("device_result[%d] = \n",j);
  }
*/
  const int N = (length);
  int blocks = ((N-1)/threadsperBlock) + 1;

  cudaThreadSynchronize();
  for(int twod =1; twod < N; twod *= 2)
  {
    int twod1	= 2* twod;
    scan_kernel1<<<blocks, threadsperBlock>>>(twod1, twod, length, device_result);
    cudaThreadSynchronize();
  }
  
//  printf("\n****Upsweep finished****\n");	
  	
  scan_kernel_makezero<<<1, 1>>>(length, device_result);				//TODO optimize
  
  for(int twod = (N/2); twod >= 1; twod /= 2)
  {
    int twod1 = 2* twod;
    scan_kernel2<<<blocks, threadsperBlock>>>(twod1, twod, length, device_result);
    cudaThreadSynchronize();
  }
  
//  printf("\n****Downsweep finished****\n");	
	
}

/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int* inarray, int* end, int* resultarray)
{

  int* device_result;
    int* device_input; 
    // We round the array sizes up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness. 
    // You may have an easier time in your implementation if you assume the 
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    // For convenience, both the input and output vectors on the device are
    // initialized to the input values. This means that you are free to simply
    // implement an in-place scan on the result vector if you wish.
    // If you do this, you will need to keep that fact in mind when calling
    // exclusive_scan from find_repeats.
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, end - inarray, device_result);

    // Wait for any work left over to be completed.
    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);
    return overallDuration;

}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}

int find_repeats(int *device_input, int length, int *device_output) {
    /* Finds all pairs of adjacent repeated elements in the list, storing the
     * indices of the first element of each pair (in order) into device_result.
     * Returns the number of pairs found.
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if 
     * it requires that. However, you must ensure that the results of
     * find_repeats are correct given the original length.
     */ 
	
  int *input_check;//,*binary_check, *temp_out, *temp_out2;
  input_check=(int*)malloc(sizeof(int)*length);
//  binary_check=(int*)malloc(sizeof(int)*length);
//  temp_out=(int*)malloc(sizeof(int)*length);
//  temp_out2=(int*)malloc(sizeof(int)*length);
  cudaMemcpy(input_check,device_input,length*sizeof(int),cudaMemcpyDeviceToHost);


//  printf("\n*****Input Check*****\n");
/*
  for(int i=0;i<length;i++)
  {
    printf("input_check[%d] = %d\n",i, input_check[i]);
  }
*/

//  cudaMemcpy(input_check,device_input,length*sizeof(int),cudaMemcpyDeviceToHost);
//  printf("\nBinary_check\n");

//  cudaError_t err;
  int threadsperBlock	=  512;
  int blocks		=  (threadsperBlock+length-1)/threadsperBlock;
  int threadsperBlock2,blocks2;
  int *binary_check_array;      
  int *len_out_array;

  cudaMalloc((void**)&binary_check_array, (length) * sizeof(int));
  len_out_array =  (int*)malloc(sizeof(int)*1);
  
//  printf("blocks = %d, threadsperBlock = %d\n", blocks, threadsperBlock);
	
//  cudaMemcpy(binary_check,binary_check_array,length,cudaMemcpyDeviceToHost);


  find_repeats_kernel<<<blocks, threadsperBlock>>>(device_input, length, binary_check_array);//,index_check_array);
  cudaThreadSynchronize(); 

//binary_check_array contains 0's and 1's of the form {0 0 1 1 0 1 0 0 }

//  cudaMemcpy(binary_check,binary_check_array,(length-1)*sizeof(int),cudaMemcpyDeviceToHost);

//  printf("\n****printing Binary_check_array*****\n");
/*
  for(int i=0;i<length;i++)
    printf(" %d",binary_check[i]);
  printf("\n");
*/

//  printf("Exclusive sum started\n");	 
//  exclusive_scan(binary_check_array, length, binary_check_array); 	//TODO		

//  cudaScan(binary_check, binary_check+length, temp_out);

/******Dummy Malloc Call*********/
  int* device_result2;
  cudaMalloc((void **)&device_result2, sizeof(int) * 2);

/*
  for(int k = 0; k<length;k++)
  {
    printf("temp_out[%d] = %d\n",k,temp_out[k]);
  }
*/
 
//  printf("Exclusive sum finished\n");	 

  threadsperBlock2=  512;
  blocks2=  (int)(ceil(((float)((length) -1))/threadsperBlock2)); //original

  exclusive_scan(binary_check_array, length, binary_check_array); 	//TODO		
  cudaThreadSynchronize();

  cudaMemcpy(len_out_array,binary_check_array+length-1,(1)*sizeof(int),cudaMemcpyDeviceToHost);

//  printf("\n*****Final Stage to Compute Output Array*****\n");
//  printf("blocks2 = %d, threadsperBlock2 = %d\n", blocks2, threadsperBlock2);
  find_matches_kernel<<<blocks2,threadsperBlock2>>>(binary_check_array,length, device_output);//,len_out);
  cudaThreadSynchronize();

//printf("length of output array = %d\n", len_out_array[0]);

/*
  int *output_check; 
  output_check=(int*)malloc(sizeof(int)*(len_out_array[0])); 
  cudaMemcpy(output_check,device_output,(len_out_array[0])*sizeof(int),cudaMemcpyDeviceToHost);
*/

//  printf("\n************Output**********\n");
/*
  for(int i=0;i<(len_out_array[0]);i++)
  {
    printf("output_check[%d] = %d\n",i,output_check[i]);
  }
  
*/

  cudaFree(binary_check_array);    
  
  return len_out_array[0];

}

/* Timing wrapper around find_repeats. You should not modify this function.
 */
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaThreadSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, result * sizeof(int),
               cudaMemcpyDeviceToHost);
/*
    for (int i=0;i<result;i++)
    {
      printf("output[%d] = %d\n", i, output[i]);
    }
*/
    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}

void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
