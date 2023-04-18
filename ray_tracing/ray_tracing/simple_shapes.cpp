#include "simple_shapes.h"

bool hit_sphere(const point3f& center, float radius, const rayf& r) {
	vec3f oc = r.origin() - center;
	auto a = dot(r.direction(), r.direction());
	auto b = 2.0f * dot(oc, r.direction());
	auto c = dot(oc, oc) - radius * radius;
	auto discriminant = b * b - 4 * a * c;
	return (discriminant > 0.f);
}
