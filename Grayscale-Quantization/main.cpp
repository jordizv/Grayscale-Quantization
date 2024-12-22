#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>

#include "frame_processing.h"

#define VALID_M_VALUES 5


int * create_cuantization_table(int M) {

	int* table = (int*) malloc(sizeof(int) * (M + OFFSET));

	int interval_amplitude = MAX_INTERVAL / M;

	// In position 0, store the size of array
	table[0] = M;

	for (int i = 0; i < M; i++)
	{
		table[i+1] = interval_amplitude * i;
	}

	return table;
}

/*Debug porpuses*/
void print_table(int* table, int m) {

	for (int i = 0; i < m+1; i++)
	{
		printf("Value at pos.%d is %d,  ", i, table[i]);
	}
	printf("\n");
}

void gpu_1(cv::VideoCapture& cap, int M) {

	int* cuantization_table = create_cuantization_table(M);
//	print_table(cuantization_table, M);

	cv::Mat frame, gray_frame;

	cap >> frame;


	//Get core components with first frame
	int fr_w = frame.cols;
	int fr_h = frame.rows;

	save_frame_dimensions(fr_w, fr_h);
	int * d_cuant_table = allocate_cuantization_table(cuantization_table, M);

	uchar* d_input_frame = allocate_frame_device_pageable(fr_w, fr_h);
	uchar* d_output_frame = allocate_frame_device_pageable(fr_w, fr_h);

	uchar* pinned_input_frame = allocate_frame_device_pinned(fr_w, fr_h);
	uchar* pinned_output_frame = allocate_frame_device_pinned(fr_w, fr_h);

	while (true) {

		if (frame.empty())
			break;

		cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

		auto start = std::chrono::high_resolution_clock::now();

		memcpy(pinned_input_frame, gray_frame.data, fr_w * fr_h * sizeof(uchar));

		
		copy_host2device(pinned_input_frame, d_input_frame, fr_w, fr_h);
		execute_kernel_1(d_input_frame, d_output_frame, d_cuant_table, fr_w, fr_h, M);
		
		copy_device2host(d_output_frame, pinned_output_frame, fr_w, fr_h);
		
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;

		// Convert uchar* to cv::Mat
		cv::Mat gray_frame1(fr_h, fr_w, CV_8UC1, pinned_output_frame);


		cv::imshow("live-gpu", gray_frame1);

		if (cv::waitKey(20) > 0) break;

		cap >> frame;
	}

	//Device memory
	cudaFree(d_input_frame);
	cudaFree(d_output_frame);
	cudaFree(d_cuant_table);

	//Pinned memory
	cudaFreeHost(pinned_input_frame);
	cudaFreeHost(pinned_output_frame);

	//Free device constants
	free_frame_dimensions();
	cap.release();
}

void gpu_2(cv::VideoCapture &cap, int M) {

	cv::Mat frame, gray_frame;

	int* cuantization_table = create_cuantization_table(M);

	cap >> frame;

	//Get core data from frame 1
	int fr_w = frame.cols;
	int fr_h = frame.rows;

	int output_fr_w = ceil(fr_w / 2);
	int output_fr_h = ceil(fr_h / 2);

	save_frame_dimensions(fr_w, fr_h);
	int* d_cuant_table = allocate_cuantization_table(cuantization_table, M);

	uchar* d_input_frame = allocate_frame_device_pageable(fr_w, fr_h);
	uchar* d_output_frame = allocate_frame_device_pageable(output_fr_w, output_fr_h);

	uchar* pinned_input_frame = allocate_frame_device_pinned(fr_w, fr_h);
	uchar* pinned_output_frame = allocate_frame_device_pinned(output_fr_w, output_fr_h);
	
	while (true) {

		if (frame.empty())
			break;

		cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

		auto start = std::chrono::high_resolution_clock::now();

		memcpy(pinned_input_frame, gray_frame.data, fr_w * fr_h * sizeof(uchar));


		copy_host2device(pinned_input_frame, d_input_frame, fr_w, fr_h);
		execute_kernel_2(d_input_frame, d_output_frame, d_cuant_table, output_fr_w, output_fr_h, M);

		copy_device2host(d_output_frame, pinned_output_frame, output_fr_w, output_fr_h);

		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;

		// Convert uchar* to cv::Mat
		cv::Mat gray_frame1(output_fr_h, output_fr_w, CV_8UC1, pinned_output_frame);

		cv::imshow("live-gpu", gray_frame1);

		if (cv::waitKey(20) > 0) break;

		cap >> frame;
	}

	//Device memory
	cudaFree(d_input_frame);
	cudaFree(d_output_frame);
	cudaFree(d_cuant_table);

	//Pinned memory
	cudaFreeHost(pinned_input_frame);
	cudaFreeHost(pinned_output_frame);

	//Free device constants
	free_frame_dimensions();
	cap.release();

}

int main(int argc, char * argv[]) {


	char* path = argv[1];
	int M = std::atoi(argv[2]);
	int type = std::atoi(argv[3]);

	/* -- debug --
	int M = 6; 
	int type = 1;
	char * path = "":
	*/

	std::cout << "Path: " << path << std::endl;
	std::cout << "M value: " << M << std::endl;
	std::cout << "Type value: " << type << std::endl;

	cv::VideoCapture cap("videos/sample.mp4");

	if (!cap.isOpened()) {
		fprintf(stderr, "Error Video: Could not access\n");
		exit(EXIT_FAILURE);
	}


	int valid_M[VALID_M_VALUES] = {2,4,6,8,10};	
	
	bool flag = false;
	for (int i = 0; i < VALID_M_VALUES; i++)
	{
		if (valid_M[i] == M)
		{
			flag = true;
			break;
		}
	}

	if (!flag)
	{
		fprintf(stderr, "Error: Invalid M value\n");
		exit(EXIT_FAILURE);
	}

	if (type == 1)
		gpu_1(cap, M);
	else if (type == 2)
		gpu_2(cap, M);
	else
	{
		fprintf(stderr, "Error: invalid type (1 or 2 only)\n");
		exit(EXIT_FAILURE);
	}

	return 0;
}