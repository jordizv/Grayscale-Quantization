
#include "frame_processing.h"

/**
* @Return cuantization table in device memory
*/
int* allocate_cuantization_table(int* cuant_table, int M) {

  size_t bytes = sizeof(int) * (M+OFFSET);

  //Allocate mamory
  int* d_cuant_table = NULL;
  cudaError_t err = cudaMalloc((void**)&d_cuant_table, bytes);

  // Copy the table from the host to global memory on the device
  err = cudaMemcpy(d_cuant_table, cuant_table, bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "ERROR: Failed to copy to global memory, %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  int verbose = 0;
  if (verbose == 1)     // Print for validation
  {
    int* table_retrieved = (int*)malloc(bytes);
    cudaMemcpy(table_retrieved, d_cuant_table, bytes, cudaMemcpyDeviceToHost);

    printf("Cuantization Table:\n");
    for (int i = 0; i < M + OFFSET; i++) {
      printf("%d\n", table_retrieved[i]);
    }
  }
 
  return d_cuant_table;
}


/**
*  @Brief calculates the cuantization value depending on the value of 
*           the pixel and the cuantization table
* 
*   @Return recovery value
*/
__device__ unsigned char cuantization_value(unsigned char actual_value, int * d_cuant_table)
{
  int M = (int) d_cuant_table[0];

  int interval_amplitude = MAX_INTERVAL / M;
  unsigned char new_value = 0;

  //Calculates the tag in table (which interval belongs)
  int pos = (actual_value / interval_amplitude) + OFFSET; 

  new_value = d_cuant_table[pos] + (interval_amplitude/2);    // recovery value for that interval

  return new_value;
}

/**
* @Brief calculates the cuantization of an entire frame
*/
__global__ void frame_cuantization(unsigned char* input, unsigned char* output, int * cuant_table, int M)
{
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  int idx = row * d_width + col;

  // Uses a shared table in each block for read cuant_table, avoiding global's memory high latency 
  extern __shared__ int table[];

  if (threadIdx.x < (M + OFFSET) && threadIdx.y == 0)
  {
    table[threadIdx.x] = cuant_table[threadIdx.x];
  }

  __syncthreads();

  if (col < d_width && row < d_height) 
  {
    unsigned char actual_value = input[idx];
    output[idx] = cuantization_value(actual_value, table);
  }
}

/**
* @Brief prepares and calls the kernel for calculate the output frame resulting 
*         of cuantization
*/
void execute_kernel_1(unsigned char* input_frame, unsigned char* output_frame, int *cuant_table, int width, int height, int M)
{
  int verbose = 0;
  dim3 dimBlock(8, 8, 1);
  dim3 dimGrid(ceil(width/dimBlock.x), ceil(height/dimBlock.y), 1);

  cudaEvent_t start, stop;
  if (verbose == 1)
  { 
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
  }
  
  frame_cuantization <<< dimGrid, dimBlock, ((M + OFFSET) * sizeof(int)) >>> (input_frame, output_frame, cuant_table, M);
  cudaDeviceSynchronize();

  if (verbose == 1)
  {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::cout << "Kernel execution time: " << elapsedTime << " ms\n";
  }

}

__global__ void frame_reduction_cuantization(unsigned char* input, unsigned char* output, int* cuant_table, int M)
{
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  int idx = row * d_width + col;

  // Uses a shared table in each block for read cuant_table, avoiding global's memory high latency 
  extern __shared__ int table[];

  if (threadIdx.x < (M + OFFSET) && threadIdx.y == 0)
  {
    table[threadIdx.x] = cuant_table[threadIdx.x];
  }
  
  __syncthreads();

  //Each thread will compute each block in the output frame
  if (col <= (d_width/2) && row <= (d_height/2))
  {
    int idx_x = col * 2;
    int idx_y = row * 2;

    int sum = input[idx_y * d_width + idx_x] +
      input[idx_y * d_width + idx_x + 1] +
      input[(idx_y + 1) * d_width + idx_x] +
      input[(idx_y + 1) * d_width + idx_x + 1];

    unsigned char average = sum / 4;

    output[row * (d_width/2) + col] = cuantization_value(average, table);
  }
}


void execute_kernel_2(unsigned char* input_frame, unsigned char* output_frame, int* cuant_table, int width, int height, int M)
{
  int verbose = 0;
  dim3 dimBlock(32, 32, 1);
  dim3 dimGrid(ceil(width / dimBlock.x), ceil(height / dimBlock.y), 1);

  cudaEvent_t start, stop;
  if (verbose == 1)
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
  }

  frame_reduction_cuantization << < dimGrid, dimBlock, ((M + OFFSET) * sizeof(int)) >> > (input_frame, output_frame, cuant_table, M);
  cudaDeviceSynchronize();

  if (verbose == 1)
  {
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::cout << "Kernel execution time: " << elapsedTime << " ms\n";
  }

}

/***
* @Brief copies the input frame data into its memory dir in device, it must to be allocate before
*/
void copy_host2device(unsigned char* h_frame, unsigned char* d_frame, int width, int height)
{
  //std::cout << "Copying from host to device " << std::endl;

  size_t bytes_size = width * height * PIXEL_SIZE;

  //Transferring data to device: from h_frame to d_frame
  cudaError_t err = cudaMemcpy(d_frame, h_frame, bytes_size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "ERROR: Failed trying to copy from Host to Device memory, %s at line %d\n", cudaGetErrorString(err), __LINE__);
    exit(EXIT_FAILURE);
  }

}

/**
* @Brief allocates the memory input and output frame in device.
* @Return tuple: input and ouput device direction
*/
unsigned char * allocate_frame_device_pageable(int width, int height)
{
 // std::cout << "Allocating frame in GPU device memory" << std::endl;
  int n_pixels = width * height;

  size_t frame_bytes = n_pixels * PIXEL_SIZE;
  //Allocate space in the in the device
  unsigned char* d_frame = NULL;

  cudaError_t err = cudaMalloc((void**) &d_frame, frame_bytes);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "ERROR: Failed to allocate device memory frame, %s at line %d\n", cudaGetErrorString(err),__LINE__);
    exit(EXIT_FAILURE);
  }

  return d_frame;
}

unsigned char* allocate_frame_device_pinned(int width, int height) 
{
  // std::cout << "Allocating frame in GPU device memory" << std::endl;
  int n_pixels = width * height;

  size_t frame_bytes = n_pixels * PIXEL_SIZE;
  //Allocate space in the in the device
  unsigned char* h_frame = NULL;

  cudaError_t err = cudaMallocHost((void**)&h_frame, frame_bytes);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "ERROR: Failed to allocate device memory frame, %s at line %d\n", cudaGetErrorString(err), __LINE__);
    exit(EXIT_FAILURE);
  }

  return h_frame;
}


/***
* @Brief stores the dimensions width and height in the constant device memory
*/
void save_frame_dimensions(int h_width, int h_height) 
{
  int verbose = 0;

  cudaMemcpyToSymbol(d_width, &h_width, sizeof(int), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_height, &h_height, sizeof(int), 0, cudaMemcpyHostToDevice);

  if (verbose == 1)       // Validate by copying back to host and printing
  {
    int retrieved_width, retrieved_height;
    cudaMemcpyFromSymbol(&retrieved_width, d_width, sizeof(int), 0, cudaMemcpyDeviceToHost);
    cudaMemcpyFromSymbol(&retrieved_height, d_height, sizeof(int), 0, cudaMemcpyDeviceToHost);

    printf("d_width (copied back): %d\n", retrieved_width);
    printf("d_height (copied back): %d\n", retrieved_height);
  }
  
}

/***
* @Brief releases constant device resources
*/
void free_frame_dimensions() 
{
  cudaFree(&d_width);
  cudaFree(&d_height);
}

/**
* @Brief retrieves the output frame from device into host
*/
void copy_device2host(unsigned char* d_output_frame, unsigned char* h_output_frame, int width, int height) {

  size_t bytes = width * height * PIXEL_SIZE;

  //Retrieving the ouput frame from device to host
  cudaError_t err = cudaMemcpy((void**)h_output_frame, d_output_frame, bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
    fprintf(stderr, "ERROR: Failed to retrieve device data into host data, %s at line %d\n", cudaGetErrorString(err), __LINE__);
    exit(EXIT_FAILURE);
  }
}