#ifndef RAY_TRACER_RAY_RAY_H_
#define RAY_TRACER_RAY_RAY_H_

#include "vec3.h"

template<typename T>
class ray {
public:
	ray() = default;
	ray(const point<T>& origin, const vec3<T>& direction, T time = static_cast<T>(0.))
		: orig(origin), dir(direction)
	{}

	point<T> origin() const { return orig; }
	vec3<T> direction() const { return dir; }
	T time() const { return tm; }
	point<T> at(T t) const {
		return orig + t * dir;
	}

public:
	point<T> orig;
	vec3<T> dir;
	T tm;
};

using rayf = ray<float>;
using rayd = ray<double>;

#endif
