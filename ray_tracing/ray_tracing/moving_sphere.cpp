#include "moving_sphere.h"

bool moving_sphere::hit(
	const rayf& r, float t_min, float t_max, hit_record& rec
) const {
	
	vec3f oc = r.origin() - center(r.time());
	auto a = r.direction().length_squared();
	auto half_b = dot(oc, r.direction());
	auto c = oc.length_squared() - radius * radius;
	auto discriminant = half_b * half_b - a * c;

	if (discriminant < 0.f) return false;
	float sqrtfd = sqrtf(discriminant);

	// find the nearest root that is within t_min and t_max
	auto root = (-half_b - sqrtfd) / a;
	if (root < t_min || root > t_max) {
		root = (-half_b + sqrtfd) / a;
		if (root < t_min || root > t_max) {
			return false;
		}
	}

	rec.t = root;
	rec.p = r.at(root);
	vec3f outward_normal = (rec.p - center(r.time())) / radius;
	rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mat_ptr;

	return true;

}

point3f moving_sphere::center(float time) const {
	return center0 + ((time - time0) /  (time1 - time0)) * (center1 - center0);
}

bool moving_sphere::bounding_box(
	float _time0, float _time1, aabb& output_box) const {
	aabb box0(
		center(_time0) - vec3f(radius, radius, radius),
		center(_time0) + vec3f(radius, radius, radius));
	aabb box1(
		center(_time1) - vec3f(radius, radius, radius),
		center(_time1) + vec3f(radius, radius, radius));
	output_box = surrounding_box(box0, box1);
	return true;
}