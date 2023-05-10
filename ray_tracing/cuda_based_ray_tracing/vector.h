#ifndef CUDA_BASED_RAY_TRACER_VECTOR_VECOR_H_
#define CUDA_BASED_RAY_TRACER_VECTOR_VECOR_H_

#include "namespaces.h"
#include "helper.h"

#include "random.h"
#include <iostream>
#include <array>
#include <type_traits>


using std::sqrt;
using std::fabs;


template <typename T>
class vec3 {

public:
	CBRT_HOST_DEVICE vec3() : e{ (T)0., (T)0., (T)0. } { }

	CBRT_HOST_DEVICE explicit vec3(T e1, T e2, T e3) : e{ e1, e2, e3 } {}

	CBRT_HOST_DEVICE T x() const { return e[0]; }
	CBRT_HOST_DEVICE T y() const { return e[1]; }
	CBRT_HOST_DEVICE T z() const { return e[2]; }

	CBRT_HOST_DEVICE vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
	CBRT_HOST_DEVICE T operator[](int i) const { return e[i]; }
	CBRT_HOST_DEVICE T& operator[](int i) { return e[i]; }

	CBRT_HOST_DEVICE vec3& operator+=(const vec3& v) {
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];
		return *this;
	}

	CBRT_HOST_DEVICE vec3& operator*=(const vec3& v) {
		e[0] *= v.e[0];
		e[1] *= v.e[1];
		e[2] *= v.e[2];
		return *this;
	}

	CBRT_HOST_DEVICE vec3& operator/=(const T t) {
		return *this *= 1 / t;
	}

	CBRT_HOST_DEVICE T length() const {
		return (T) sqrtf(length_squared());
	}

	CBRT_HOST_DEVICE T length_squared() const {
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}

	CBRT_HOST_DEVICE bool near_zero() const {
		// Return true if the vector is close to zero in all dimensions.
		const auto s = 1e-8;
		return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
	}


public:
	T e[3];
};

using vec3f = vec3<float>;
using vec3d = vec3<double>;
using vec3c = vec3<uint8_t>;

using point3d = vec3<double>;
using point3f = vec3<float>;
using point3c = vec3<uint8_t>;

using color3d = vec3<double>;
using color3f = vec3<float>;
using color3c = vec3<uint8_t>;

template <typename T> using color = vec3<T>;
template <typename T> using point = vec3<T>;


// vec3 utility functions for float

CBRT_HOST CBRT_INLINE std::ostream& operator<<(std::ostream& out, const vec3f& v) {
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

CBRT_HOST_DEVICE CBRT_INLINE vec3f operator+(const vec3f& u, const vec3f& v) {
	return vec3f(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

CBRT_HOST_DEVICE CBRT_INLINE vec3f operator-(const vec3f& u, const vec3f& v) {
	return vec3f(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

CBRT_HOST_DEVICE CBRT_INLINE vec3f operator*(const vec3f& u, const vec3f& v) {
	return vec3f(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

CBRT_HOST_DEVICE CBRT_INLINE vec3f operator*(float t, const vec3f& v) {
	return vec3f(t * v.e[0], t * v.e[1], t * v.e[2]);
}

CBRT_HOST_DEVICE CBRT_INLINE vec3f operator*(const vec3f& v, float t) {
	return t * v;
}

CBRT_HOST_DEVICE CBRT_INLINE vec3f operator/(vec3f v, float t) {
	return (1 / t) * v;
}

CBRT_HOST_DEVICE CBRT_INLINE float dot(const vec3f& u, const vec3f& v) {
	return u.e[0] * v.e[0]
		+ u.e[1] * v.e[1]
		+ u.e[2] * v.e[2];
}

CBRT_HOST_DEVICE CBRT_INLINE vec3f cross(const vec3f& u, const vec3f& v) {
	return vec3f(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

CBRT_HOST_DEVICE CBRT_INLINE vec3f unit_vector(vec3f v) {
	return v / v.length();
}


/*
// random methods
CBRT_INLINE vec3f random_in_unit_sphere_f() {
	while (true) {
		auto p = vec3f::random(-1.f, 1.f);
		if (p.length_squared() >= 1.f) continue;
		return p;
	}
}

// random methods
CBRT_INLINE vec3d random_in_unit_sphere_d() {
	while (true) {
		auto p = vec3d::random(-1., 1.);
		if (p.length_squared() >= 1.) continue;
		return p;
	}
}

CBRT_INLINE vec3f random_unit_vector_on_sphere_f() {
	return unit_vector(random_in_unit_sphere_f());
}

CBRT_INLINE vec3d random_unit_vector_on_sphere_d() {
	return unit_vector(random_in_unit_sphere_d());
}


CBRT_INLINE vec3f random_in_hemisphere(const vec3f& normal) {
	vec3f in_unit_sphere = random_in_unit_sphere_f();
	if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
		return in_unit_sphere;
	else
		return -in_unit_sphere;
}


CBRT_INLINE vec3f reflect(const vec3f& v, const vec3f& n) {
	return v - 2.0f * dot(v, n) * n;
}

CBRT_INLINE vec3d reflect(const vec3d& v, const vec3d& n) {
	return v - 2.0 * dot(v, n) * n;
}


CBRT_INLINE vec3f refract(const vec3f& uv, const vec3f& n, float etai_over_etae) {
	auto cos_theta = fminf(dot(-uv, n), 1.0f);
	vec3f r_out_prependicular = etai_over_etae * (uv + cos_theta * n);
	vec3f r_out_parallel = -sqrtf(fabsf(1.0f - r_out_prependicular.length_squared())) * n;
	return r_out_parallel + r_out_prependicular;

}

CBRT_INLINE vec3f random_in_unit_disk() {
	while (true) {
		auto p = vec3f(random_float(-1.f, 1.f), random_float(-1.f, 1.f), 0);
		if (p.length_squared() >= 1) continue;
		return p;
	}
}

	CBRT_HOST_DEVICE CBRT_INLINE static vec3 random() {
		return vec3(random_function<T>(), random_function<T>(), random_function<T>());
	}

	CBRT_HOST_DEVICE CBRT_INLINE static vec3 random(T min, T max) {
		return vec3(
			random_function<T>(min, max),
			random_function<T>(min, max),
			random_function<T>(min, max)
		);
	}

*/

#endif
