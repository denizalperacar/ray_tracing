#ifndef CUDA_BASED_RAY_TRACING_SHAPES_SPHERE_SPHERE_H_
#define CUDA_BASED_RAY_TRACING_SHAPES_SPHERE_SPHERE_H_

#include "common.h"

CBRT_BEGIN

CBRT_HOST_DEVICE float hit_sphere(const point3f& center, double radius, const ray& r) {
  vec3f oc = r.origin() - center;
  auto a = r.direction().length_squared();
  auto half_b = dot(oc, r.direction());
  auto c = oc.length_squared() - radius * radius;
  auto discriminant = half_b * half_b - a * c;

  return (discriminant < 0) * -1.f + (discriminant >= 0) * (-half_b - sqrt(discriminant)) / a;
}

/*
CBRT_HOST_DEVICE float hit_sphere_normal_coloring(const point3f& center, double radius, const ray& r) {
  vec3f oc = r.origin() - center;
  auto a = dot(r.direction(), r.direction());
  auto b = 2.0 * dot(oc, r.direction());
  auto c = dot(oc, oc) - radius * radius;
  auto discriminant = b * b - 4 * a * c;
  return (discriminant < 0) * -1.f + (discriminant >= 0) * (-b - sqrt(discriminant)) / (2.0f * a);
}
*/


CBRT_END



#endif // !CUDA_BASED_RAY_TRACING_SHAPES_SPHERE_SPHERE_H_

