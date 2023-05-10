#ifndef CUDA_BASED_RAY_TRACING_RAY_RAY_COLOR_H_
#define CUDA_BASED_RAY_TRACING_RAY_RAY_COLOR_H_

#include "common.h"
#include "sphere.h"

CBRT_BEGIN

CBRT_HOST_DEVICE color3f ray_color(const ray& r, hittable* world) {
  hit_record rec;

  if (world->hit(r, 0.f, infinity, rec)) {
    return 0.5f * (rec.normal + color3f(1.0f, 1.0f, 1.0f));
  }

  vec3f unit_direction = unit_vector(r.direction());
  float t = 0.5f * (unit_direction.y() + 1.0f);
  return (1.0f - t) * color3f(1.0f, 1.0f, 1.0f) + t * color3f(0.5f, 0.7f, 1.0f);
}

CBRT_END

#endif // !1

