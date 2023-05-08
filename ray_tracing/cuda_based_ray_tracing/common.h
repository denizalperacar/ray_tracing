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


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>

#include "helper.h"

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




#endif
