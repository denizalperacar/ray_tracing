#include "constant_medium.h"

bool constant_medium::hit(
	const rayf& r, float t_min, float t_max, hit_record& rec
) const {
	const bool enableDebug = false;
	const bool debugging = enableDebug && random_float() < 0.00001f;

	hit_record rec1, rec2;

	if (!boundary->hit(r, -infinityf, infinityf, rec1)) {
		return false;
	}

	if (!boundary->hit(r, rec1.t+0.0001f, infinityf, rec2)) {
		return false;
	}

	if (debugging) {
		std::cerr << "\nt_min=" << rec.t << ", t_max=" << rec2.t << "\n";
	}
	if (rec1.t < t_min) { rec1.t = t_min; }
	if (rec2.t > t_max) { rec2.t = t_max; }
	if (rec1.t >= rec2.t) { return false; }

	if (rec1.t < 0.f) { rec.t = 0.0f; }

	const float ray_length = r.direction().length();
	const float distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
	const float hit_distance = neg_inv_density * logf(random_float());

	if (hit_distance > distance_inside_boundary) { return false; }

	rec.t = rec1.t + hit_distance / ray_length;
	rec.p = r.at(rec.t);

	if (debugging) {
		std::cerr << "hit_distance = " << hit_distance << "\n"
			<< "rec.t = " << rec.t << "\n"
			<< "rec.p = " << rec.p << "\n";
	}

	rec.normal = vec3f(1.f, 0.f, 0.f);
	rec.front_face = true;
	rec.mat_ptr = phase_function;

	return true;
}