#ifndef CUDA_BASED_RAY_TRACING_RAY_RAY_COLOR_H_
#define CUDA_BASED_RAY_TRACING_RAY_RAY_COLOR_H_

#include "common.h"

CBRT_BEGIN

CBRT_HOST_DEVICE color3f ray_color(const ray& r) {
  vec3f unit_direction = unit_vector(r.direction());
  auto t = 0.5f * (unit_direction.y() + 1.0f);
  return (1.0f - t) * color3f(1.0f, 1.0f, 1.0f) + t * color3f(0.5f, 0.7f, 1.0f);
}

CBRT_END

#endif // !1

