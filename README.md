# Grayscale-Quantization
Quantization (in image processing) is one of the steps commonly used in image compression such as JPEG.
It's a lossy compression technique achieved by compressing a range of values to a single quantum (discrete) value.
The purpose and scope of the project is to simulate quantization in videos (image sequence, frames) by exploiting the power of GPU parallelism using CUDA.

From a number of partitions M, create the quantization table and, in grayscale, assign the recovery pixel value at each pixel. In short, we are restricting the input pixels in
intervals and in each interval assign the same value for output.  

## Table of Contents
	
- [Features](#features)
- [Environment and Dependencies](#Environment-and-Dependencies)
- [How to Run Application](#How-to-Run-Application)
- [Example](#example)

## Features

- **CUDA GPU Acceleration**
- **Real Time**
- **Pinned memory**: Faster transmision of data between host and device and better bandwidt, necessary for *real-time efficiency*. As we use the same input and ouput direction in all frames, its not limited and does not affect system memory.
- **Shared memory**: when can avoid accesing global variables more than once. Shared between blocks, ~100 faster than a global access.

## Environment and Dependencies

Developed in Visual Studio 2022.

Dependencies:

- **Opencv - 4.10**
- **CUDA - 12.6**

## How to Run Application

Takes 3 arguments, in order:

- 1: Path video to test
- 2: M, number of partitions in the quantization table, by convention restricted to any of these values: [2,4,6,8,10]
- 3: Type of quantization. Can be two types:
	- 1: Same size output as input.
	- 2: Dimensions of the output image are half of the input.

Place the video you want to test in the "videos" folder. In the same directory as Grayscale-Quantization.exe, in power-shell (u other), run the next command:
In this case, will be with M = 4 and type = 1.
```bash
\Grayscale-Quantization> .\Grayscale-Quantization.exe ".\videos\sample.mp4" 4 1
```


## Example 
- Tested with
	- CPU: AMD Ryzen 7 6800HS 
	- GPU: GeForce RTX 3050 Laptop GPU

- Video Input Sample:

https://github.com/user-attachments/assets/e8dc9129-20ab-40f2-83ce-8351a8f09118

**Samples are recorded in real-time with OBS Studio (overload)**

- Sample with M=2 and Type=1:

https://github.com/user-attachments/assets/8ff5109b-3f17-429f-b19f-2fa3b25f8d4f

- Sample with M=8 and Type=2:

https://github.com/user-attachments/assets/01c1e699-0ed3-48e5-8eb1-7873f3da1118



