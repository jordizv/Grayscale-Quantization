#ifndef FRAME_PROCESSING_H
#define FRAME_PROCESSING_H

#include <iostream>
#include <cuda_runtime.h>
#include <tuple>


#define PIXEL_SIZE sizeof(unsigned char)
#define MAX_INTERVAL 256
#define OFFSET 1


/* Constant memory */
__constant__ int d_width;
__constant__ int d_height;

/*Global memory*/

void save_frame_dimensions(int width, int height);
int * allocate_cuantization_table(int* table, int M);
void free_frame_dimensions();

unsigned char * allocate_frame_device_pageable(int width, int height);
unsigned char * allocate_frame_device_pinned(int width, int height);
void copy_host2device(unsigned char* h_frame, unsigned char* d_frame, int width, int height);

void execute_kernel_1(unsigned char* input_frame, unsigned char* output_frame, int * cuant_table, int width, int height, int M);
void execute_kernel_2(unsigned char *input_frame, unsigned char* output_frame, int * cuant_table, int width, int height, int M);

__global__ void frame_cuantization(unsigned char* input, unsigned char* output, int * cuant_table, int M);
__global__ void frame_reduction_cuantization(unsigned char* input, unsigned char* output, int* cuant_table, int M);

void copy_device2host(unsigned char* d_output_frame, unsigned char* h_output_frame, int width, int height);

__device__ unsigned char cuantization_value(unsigned char actual_value, int * cuant_table);


#endif // FRAME_PROCESSING_H
