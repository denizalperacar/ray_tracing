#ifndef CUDA_BASED_RAY_TRACING_SHAPES_SPHERE_SPHERE_H_
#define CUDA_BASED_RAY_TRACING_SHAPES_SPHERE_SPHERE_H_

#include "common.h"
#include "hittable.h"

CBRT_BEGIN


class sphere : public hittable {

public:
	CBRT_HOST_DEVICE sphere() {}
	
	CBRT_HOST_DEVICE sphere(point3f cen, float r) : center(cen), radius(r) {}
	CBRT_HOST_DEVICE virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

public:
  point3f center;
	float radius;

};

CBRT_HOST_DEVICE bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	vec3f oc = r.origin() - center;
	auto a = r.direction().length_squared();
	auto half_b = dot(oc, r.direction());
	auto c = oc.length_squared() - radius * radius;

	auto discriminant = half_b * half_b - a * c;
	if (discriminant < 0) return false;
	auto sqrt_f = sqrtf(discriminant);
	// Find the nearest root that lies in the acceptable range.
	auto root = (-half_b - sqrt_f) / a;
	if (root < t_min || t_max < root) {
		root = (-half_b + sqrt_f) / a;
		if (root < t_min || t_max < root)
			return false;
	}

	rec.t = root;
	rec.p = r.at(rec.t);
	vec3f outward_normal = (rec.p - center) / radius;
	rec.set_face_normal(r, outward_normal);

	return true;
}




CBRT_HOST_DEVICE float hit_sphere(const point3f& center, float radius, const ray& r) {
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

