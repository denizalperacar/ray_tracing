#include "ray_color.h"

color3f ray_color(const rayf& r, const hittable& world, int depth) {
	hit_record rec;

	//
	if (depth <= 0) {
		return color3f(0.f, 0.f, 0.f);
	}

	if (world.hit(r, 0.001f, infinityf, rec)) {

		rayf scattered;
		color3f attenuation;

		if (rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
			return attenuation * ray_color(scattered, world, depth - 1);
		}
		return color3f(0.f, 0.f, 0.f);
	}

	vec3f unit_direction = unit_vector(r.direction());
	auto t = static_cast<float> (0.5f * (unit_direction.y() + 1.0f));
	return (1.0f - t) * color3f(1.0f, 1.0f, 1.0f) + t * color3f(0.5f, 0.7f, 1.0f);
}

color3d ray_color(const rayd& r) {
	vec3d unit_direction = unit_vector(r.direction());
	double t = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - t) * color3d(1.0, 1.0, 1.0) + t * color3d(0.5, 0.7, 1.0);
}