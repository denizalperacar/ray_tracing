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

bool xz_rect::hit(const rayf& r, float t_min, float t_max, hit_record& rec) const {

	float t = (k - r.origin().y()) / r.direction().y();

	if (t < t_min || t > t_max) {
		return false;
	}

	float x = r.origin().x() + t * r.direction().x();
	float z = r.origin().z() + t * r.direction().z();

	if (x < x0 || x > x1 || z < z0 || z > z1) {
		return false;
	}

	rec.u = (x - x0) / (x1 - x0);
	rec.v = (z - z0) / (z1 - z0);
	rec.t = t;
	vec3f out_normal = vec3f(0.f, 1.f, 0.f);
	rec.set_face_normal(r, out_normal);
	rec.mat_ptr = mp;
	rec.p = r.at(t);
	return true;

}


bool yz_rect::hit(const rayf& r, float t_min, float t_max, hit_record& rec) const {

	float t = (k - r.origin().x()) / r.direction().x();

	if (t < t_min || t > t_max) {
		return false;
	}

	float y = r.origin().y() + t * r.direction().y();
	float z = r.origin().z() + t * r.direction().z();

	if (y < y0 || y > y1 || z < z0 || z > z1) {
		return false;
	}

	rec.u = (y - y0) / (y1 - y0);
	rec.v = (z - z0) / (z1 - z0);
	rec.t = t;
	vec3f out_normal = vec3f(1.f, 0.f, 0.f);
	rec.set_face_normal(r, out_normal);
	rec.mat_ptr = mp;
	rec.p = r.at(t);
	return true;

}