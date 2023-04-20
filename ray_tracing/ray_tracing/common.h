#ifndef RAY_TRACER_COMMON_COMMON_H_
#define RAY_TRACER_COMMON_COMMON_H_


#include <memory>
#include <cmath>
#include <limits>
#include <random>


using std::shared_ptr;
using std::make_shared;
using std::sqrtf;
using std::sqrt;


constexpr auto IMAGE_ASPECT_RATIO = 16.0 / 9.0;
constexpr int IMAGE_WIDTH = 400;
constexpr int IMAGE_HEIGHT = static_cast<int>(IMAGE_WIDTH / IMAGE_ASPECT_RATIO);
constexpr int SAMPLES_PER_PIXEL = 100;


// constants
const float infinityf = std::numeric_limits<float>::infinity();
const double infinityd = std::numeric_limits<double>::infinity();

const double pi = 3.1415926535897932385;
const float pi_f = 3.1415926535897932385f;

// utility functions
inline float degrees_to_radians(float degrees) {
	return degrees * pi_f / 180.0f;
}

inline double degrees_to_radians(double degrees) {
	return degrees * pi / 180.0;
}

inline double random_float() {
  static std::uniform_real_distribution<float> distribution(0.0, 1.0);
  static std::mt19937 generator;
  return distribution(generator);
}

inline double random_float(float min, float max) {
  // Returns a random real in [min,max).
  return min + (max - min) * random_float();
}

inline double clamp(double x, double min, double max) {
  if (x < min) return min;
  if (x > max) return max;
  return x;
}

// common headers
#include "ray.h"
#include "vec3.h"


#endif
