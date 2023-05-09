#ifndef CUDA_BASED_RAY_TRACING_RAY_RAY_COLOR_H_
#define CUDA_BASED_RAY_TRACING_RAY_RAY_COLOR_H_

#include "common.h"
#include "sphere.h"

CBRT_BEGIN

CBRT_HOST_DEVICE color3f ray_color(const ray& r) {
  float t = hit_sphere(point3f(0.f, 0.f, -1.f), 0.5f, r);

  if (t > 0.f) {
		vec3f N = unit_vector(r.at(t) - vec3f(0.f, 0.f, -1.f));
		return 0.5f * color3f(N.x() + 1.f, N.y() + 1.f, N.z() + 1.f);
	}

  vec3f unit_direction = unit_vector(r.direction());
  t = 0.5f * (unit_direction.y() + 1.0f);
  return (1.0f - t) * color3f(1.0f, 1.0f, 1.0f) + t * color3f(0.5f, 0.7f, 1.0f);
}

CBRT_END

#endif // !1

