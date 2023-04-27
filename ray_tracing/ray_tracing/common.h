#ifndef RAY_TRACER_COMMON_COMMON_H_
#define RAY_TRACER_COMMON_COMMON_H_


#include <memory>
#include <cmath>
#include <limits>
#include <cstdint>


using std::shared_ptr;
using std::make_shared;
using std::sqrtf;
using std::sqrt;


constexpr float IMAGE_ASPECT_RATIO = 1.0f; // 16.0f / 9.0f;
constexpr int IMAGE_WIDTH = 400;
constexpr int IMAGE_HEIGHT = static_cast<int>(IMAGE_WIDTH / IMAGE_ASPECT_RATIO);
constexpr int SAMPLES_PER_PIXEL = 200;
constexpr int MAX_DEPTH = 50;
constexpr int NUMTHREADS = 4;

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

// common headers
#include "ray.h"
#include "vec3.h"


#endif
