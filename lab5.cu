#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCKSIZE 4 // Number of threads in each thread block
 
/* 
 * CUDA kernel to find a global max, each thread process
 * one element.
 * @param values	input of an array of integers in which we search a max number
 * @param max		output of this kernel, the max number in array values 
 * @param reg_maxes	output of this kernel, some regional max number for input array
 * @param num_regions	input of this kernel, number of regions we use to reduce lock contentions
 * @param n		input of this kernel, total number of element in input array
 */
__global__ void global_max(int *values, int *max, int *reg_maxes, int num_regions, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x; 
    int val = values[i];
    int region = i % num_regions; 
    if(atomicMax(&reg_maxes[region],val) < val) 
    { 
        atomicMax(max,val); 
    }
}

// Write the cuda kernel to normal all elements in input values,
// the output is stored back into output array, max is the maximum value in the array
// values, n is the total number of elements in values.
__global__ void normalize(int *values, int *max, float *output, int n)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x; 
    if(i < n)
    {
        output[i] = (float) values[i] / (float)max[0];
    }

}   
 
int main( int argc, char* argv[] )
{
    // Size of vectors
    int i;
    int input[] = {4, 5, 6, 7, 19, 10, 0, 4, 2, 3, 1, 7, 9, 11, 45, 23, 100, 29};
    int n = sizeof(input) / sizeof(float); //careful, this usage only works with statically allocated arrays, NOT dynamic arrays

    // Host input vectors
    int *h_in = input;
    //Host output vector
    float *h_out = (float *) malloc(n * sizeof(float));
 
    // Device input vectors
    int *d_in;;

    //Device output vector
    float *d_out;
    int *d_reg_max;// memory for regional max
    int *d_gl_max;  // memory for global max
 
    // Size, in bytes, of each vector
    int bytes = n * sizeof(int);
    int num_reg = ceil(n / (float)BLOCKSIZE);  //num of regions we will use in calculation of global max
   
    // Allocate memory for each vector on GPU
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMalloc(&d_reg_max, num_reg * sizeof(int) );
    cudaMalloc(&d_gl_max, sizeof(int) );
 
    //PLEASE initialize the values in d_reg_max and d_gl_max to ZERO!!!


 
    // Copy host data to device
    cudaMemcpy( d_in, h_in, bytes, cudaMemcpyHostToDevice);
 
    // Number of threads in each thread block
    int blockSize = BLOCKSIZE;
 
    // Number of thread blocks in grid
    int gridSize = (int)ceil((float)n/blockSize);
 
    //printf("BlockSize: %d, Gridsize: %d", blockSize, gridSize);
    // Execute the kernel
    global_max<<<gridSize, blockSize>>>(d_in, d_gl_max, d_reg_max, num_reg, n); //after this kernel called, *d_gl_max is ready to use
    cudaDeviceSynchronize();
 
    // Execute the second kernel, use the data returned by the first kernel
    normalize<<<gridSize, blockSize>>>(d_in, d_gl_max, d_out, n); 
 
    // Copy array back to host
    cudaMemcpy( h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost );
 
    // Show the result
    printf("The original array is: ");
    for(i = 0; i < n; i ++)
        printf("%6d,", h_in[i] );    
    
    printf("\n\nNormalized   array is: ");
    for(i = 0; i < n; i++)
        printf("%6.2f,", h_out[i] );    
    puts("");
    
    // Release device memory
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_reg_max);
    cudaFree(d_gl_max);
 
    // Release host memory
    free(h_out);
 
    return 0;
}
