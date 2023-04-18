#ifndef RAY_TRACER_RAY_RAY_H_
#define RAY_TRACER_RAY_RAY_H_

#include "vec3.h"

template<typename T>
class ray {
public:
	ray() = default;
	ray(const point<T>& origin, const vec3<T>& direction)
		: orig(origin), dir(direction)
	{}

	point<T> origin() const { return orig; }
	vec3<T> direction() const { return dir; }
	point<T> at(T t) const {
		return orig + t * dir;
	}

public:
	point<T> orig;
	vec3<T> dir;
};

using rayf = ray<float>;
using rayd = ray<double>;

color3f ray_color(const rayf& r) {
	vec3f unit_direction = unit_vector(r.direction());
	float t = static_cast<float> (0.5f * (unit_direction.y() + 1.0f));
	return (1.0f - t) * color3f(1.0f, 1.0f, 1.0f) + t * color3f(0.5f, 0.7f, 1.0f);
}

color3d ray_color(const rayd& r) {
	vec3d unit_direction = unit_vector(r.direction());
	double t = 0.5 * (unit_direction.y() + 1.0);
	return (1.0 - t) * color3d(1.0, 1.0, 1.0) + t * color3d(0.5, 0.7, 1.0);
}

#endif
