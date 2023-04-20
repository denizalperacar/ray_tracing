#include "simple_shapes.h"

float hit_sphere(const point3f& center, float radius, const rayf& r) {
	vec3f oc = r.origin() - center;
	auto a = r.direction().length_squared();
	// auto b = 2.0f * dot(oc, r.direction());
	auto half_b = dot(oc, r.direction());
	auto c = oc.length_squared() - radius * radius;
	auto discriminant = half_b * half_b -  a * c;
	
	if (discriminant < 0.f) {
		return - 1.0f;
	}
	else {
		return (-half_b - sqrtf(discriminant)) / a;
	}
}
