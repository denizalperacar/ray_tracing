#include "ray_color.h"

color3f ray_color(const rayf& r, const color3f& background, const hittable& world, int depth) {
	hit_record rec;

	//
	if (depth <= 0) {
		return color3f(0.f, 0.f, 0.f);
	}

	if (!world.hit(r, 0.001f, infinityf, rec)) {
		return background;
	}

	rayf scattered;
	color3f attenuation;
	color3f emitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p);

	if (!rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
		return emitted;
	}

	return emitted + attenuation * ray_color(scattered, background, world, depth - 1);
}

color3d ray_color(const rayd& r) {
	vec3d unit_direction = unit_vector(r.direction());
	double t = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - t) * color3d(1.0, 1.0, 1.0) + t * color3d(0.5, 0.7, 1.0);
}