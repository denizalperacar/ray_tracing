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
	get_sphere_uv(outward_normal, rec.u, rec.v);
	rec.mat_ptr = mat_ptr;

	return true;
}


bool sphere::bounding_box(float time0, float time1, aabb& output_box) const {
	output_box = aabb(
		center - vec3f(radius, radius, radius),
		center + vec3f(radius, radius, radius)
	);
	return true;
}

void sphere::get_sphere_uv(const point3f& p, float& u, float& v) {

	float theta = acosf(-p.y());
	float phi = atan2f(-p.z(), p.x()) + pi_f;

	u = phi / (2 * pi_f);
	v = theta / (pi_f);
}