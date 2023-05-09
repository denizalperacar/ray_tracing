/*
@file:common.h
@author: Deniz A. ACAR
@brief: Common definitions and functions used in the project
*/

#ifndef CUDA_BASED_RAY_TRACING_COMMON_COMMON_H_
#define CUDA_BASED_RAY_TRACING_COMMON_COMMON_H_


#include "namespaces.h"


#include <cstdint>
#include <array>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <limits>
#include <chrono>
#include <atomic>
#include <unordered_map>
#include <atomic>
#include <stdexcept>
#include <string>
#include <functional>


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#include "helper.h"
#include "vector.h"
#include "ray.h"

// common methods and data structures


// Image properties
constexpr float DEFAULT_IMAGE_ASPECT_RATIO = 1.0f;
constexpr uint32_t DEFAULT_IMAGE_WIDTH = 1024;
constexpr int DEFAULT_IMAGE_HEIGHT = static_cast<int>(DEFAULT_IMAGE_WIDTH / DEFAULT_IMAGE_ASPECT_RATIO);
constexpr uint32_t DEFAULT_NUMBER_OF_PIXELS = DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT;

// constants

constexpr uint32_t  NUM_THREADS_MIN = 32;
constexpr uint32_t  NUM_THREADS_64  = 64;
constexpr uint32_t  NUM_THREADS_128 = 128;
constexpr uint32_t  NUM_THREADS_256 = 256;
constexpr uint32_t  NUM_THREADS_MAX = 1024;

constexpr float pi = 3.1415926535897932385f;


typedef void (*writeOneByte)(unsigned char);

// output file
std::ofstream myFile("../result.jpg", std::ios_base::out | std::ios_base::binary);

// write a single byte compressed by tooJpeg
void image_output(unsigned char byte)
{
	myFile << byte;
}


class render_color {
public:

	CBRT_HOST_DEVICE render_color() : r(0), g(0), b(0) {}

	CBRT_HOST_DEVICE render_color(uint8_t red, uint8_t green, uint8_t blue)
		: r(red), g(green), b(blue) {}

	CBRT_HOST std::ofstream& draw(std::ofstream& os) {
		os << r << " " << g << " " << b << "\n";
		return os;
	}

	CBRT_HOST_DEVICE void print() {
		printf("%d %d %d\n", r, g, b);
	}

public:
	uint8_t r;
	uint8_t g;
	uint8_t b;

};

std::ostream& operator<<(std::ostream& os, render_color obj) {
	os << +obj.r << " " << +obj.g << " " << +obj.b << "\n";
	return os;
}

std::fstream& operator<<(std::fstream& os, render_color obj) {
	os << +obj.r << " " << +obj.g << " " << +obj.b << "\n";
	return os;
}



#endif
