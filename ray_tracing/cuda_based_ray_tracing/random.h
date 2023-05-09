#ifndef CUDA_BASED_RAY_TRACING_COMMON_RANDOM_H_
#define CUDA_BASED_RAY_TRACING_COMMON_RANDOM_H_

#include "namespaces.h"
#include "helper.h"
#include <cstdlib>
#include <random>

// [TODO] update the random number generator functions for cuda

CBRT_BEGIN
/*
template <typename T>
inline T random_function() {
  static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
  static std::mt19937 generator;
  return distribution(generator);
}

template <typename T>
 CBRT_HOST_DEVICE T random_function(T min, T max) {
  return min + (max - min) * random_function<T>();
}

 CBRT_HOST_DEVICE float random_float() {
  static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
  static std::mt19937 generator;
  return distribution(generator);
}

 CBRT_HOST_DEVICE float random_float(float min, float max) {
  // Returns a random real in [min,max).
  return min + (max - min) * random_float();
}

 CBRT_HOST_DEVICE float clamp(float x, float min, float max) {
  if (x < min) return min;
  if (x > max) return max;
  return x;
}

 CBRT_HOST_DEVICE int random_int(int min, int max) {
  return static_cast<int>(random_float(static_cast<float>(min), static_cast<float>(max + 1)));
}
*/

CBRT_END

#endif
