#include "rectangle.h"

bool xy_rect::hit(const rayf& r, float t_min, float t_max, hit_record& rec) const {

	float t = (k - r.origin().z()) / r.direction().z();

	if (t < t_min || t > t_max) {
		return false;
	}

	float x = r.origin().x() + t * r.direction().x();
	float y = r.origin().y() + t * r.direction().y();

	if (x < x0 || x > x1 || y < y0 || y > y1) {
		return false;
	}

	rec.u = (x - x0) / (x1-x0);
	rec.v = (y - y0) / (y1 - y0);
	rec.t = t;
	vec3f out_normal = vec3f(0.f, 0.f, 1.f);
	rec.set_face_normal(r, out_normal);
	rec.mat_ptr = mp;
	rec.p = r.at(t);
	return true;

}