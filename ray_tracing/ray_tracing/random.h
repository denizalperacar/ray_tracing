#ifndef RAY_TRACING_COMMON_RANDOM_H_
#define RAY_TRACING_COMMON_RANDOM_H_

#include <cstdlib>
#include <random>

template <typename T>
inline T random_function() {
  static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
  static std::mt19937 generator;
  return distribution(generator);
}

template <typename T>
inline T random_function(T min, T max) {
  return min + (max - min) * random_function<T>();
}

inline float random_float() {
  static std::uniform_real_distribution<float> distribution(0.0f, 1.0f);
  static std::mt19937 generator;
  return distribution(generator);
}

inline float random_float(float min, float max) {
  // Returns a random real in [min,max).
  return min + (max - min) * random_float();
}

inline float clamp(float x, float min, float max) {
  if (x < min) return min;
  if (x > max) return max;
  return x;
}

inline int random_int(int min, int max) {
  return static_cast<int>(random_float(static_cast<float>(min), static_cast<float>(max+1)));
}

#endif
