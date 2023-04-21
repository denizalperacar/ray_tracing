#include "sphere.h"
#include "vec3.h"

bool sphere::hit(const rayf& r, float t_min, float t_max, hit_record& rec) const
{

	vec3f oc = r.origin() - center;
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
	vec3f outward_normal = (rec.p - center) / radius;
	rec.set_face_normal(r, outward_normal);
	rec.mat_ptr = mat_ptr;

	return true;
}
