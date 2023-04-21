#ifndef RAY_TRACER_VECTOR3_VEC3_H_
#define RAY_TRACER_VECTOR3_VEC3_H_

#include "random.h"
#include <iostream>
#include <array>
#include <type_traits>


using std::sqrt;
using std::fabs;


template <typename T>
class vec3 {

	static_assert (
		std::is_floating_point<T>::value, 
		"class A can only be instantiated with floating point types"
	);

public:
	vec3() {
		if (std::is_same<T, float>::value) {
			e = { 0.f,0.f,0.f };
		}
		else {
			e = { 0.,0.,0. };
		}
	}

	explicit vec3(T e1, T e2, T e3) : e{ e1, e2, e3 } {}

	T x() const { return e[0]; }
	T y() const { return e[1]; }
	T z() const { return e[2]; }

	vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
	T operator[](int i) const { return e[i]; }
	T& operator[](int i) { return e[i]; }
	
	vec3& operator+=(const vec3& v) {
		e[0] += v.e[0];
		e[1] += v.e[1];
		e[2] += v.e[2];
		return *this;
	}

	vec3& operator*=(const vec3& v) {
		e[0] *= v.e[0];
		e[1] *= v.e[1];
		e[2] *= v.e[2];
		return *this;
	}

	vec3& operator/=(const T t) {
		return *this *= 1 / t;
	}

	T length() const {
		return sqrt(length_squared());
	}

	T length_squared() const {
		return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
	}

	inline static vec3 random() {
		return vec3(random_function<T>(), random_function<T>(), random_function<T>());
	}

	inline static vec3 random(T min, T max) {
		return vec3(
			random_function<T>(min, max), 
			random_function<T>(min, max), 
			random_function<T>(min, max)
		);
	}

	bool near_zero() const {
		// Return true if the vector is close to zero in all dimensions.
		const auto s = 1e-8;
		return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
	}


public:
	std::array<T, 3> e;
};

using vec3f = vec3<float>;
using vec3d = vec3<double>;
using point3d = vec3<double>;
using point3f = vec3<float>;
using color3d = vec3<double>;
using color3f = vec3<float>;
template <typename T> using color = vec3<T>;
template <typename T> using point = vec3<T>;


// vec3 utility functions for float

inline std::ostream& operator<<(std::ostream& out, const vec3f& v) {
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline vec3f operator+(const vec3f& u, const vec3f& v) {
	return vec3f(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline vec3f operator-(const vec3f& u, const vec3f& v) {
	return vec3f(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline vec3f operator*(const vec3f& u, const vec3f& v) {
	return vec3f(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline vec3f operator*(float t, const vec3f& v) {
	return vec3f(t * v.e[0], t * v.e[1], t * v.e[2]);
}

inline vec3f operator*(const vec3f& v, float t) {
	return t * v;
}

inline vec3f operator/(vec3f v, float t) {
	return (1 / t) * v;
}

inline float dot(const vec3f& u, const vec3f& v) {
	return u.e[0] * v.e[0]
		+ u.e[1] * v.e[1]
		+ u.e[2] * v.e[2];
}

inline vec3f cross(const vec3f& u, const vec3f& v) {
	return vec3f(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline vec3f unit_vector(vec3f v) {
	return v / v.length();
}

// vec3 utility functions for double

inline std::ostream& operator<<(std::ostream& out, const vec3d& v) {
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

inline vec3d operator+(const vec3d& u, const vec3d& v) {
	return vec3d(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

inline vec3d operator-(const vec3d& u, const vec3d& v) {
	return vec3d(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

inline vec3d operator*(const vec3d& u, const vec3d& v) {
	return vec3d(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

inline vec3d operator*(double t, const vec3d& v) {
	return vec3d(t * v.e[0], t * v.e[1], t * v.e[2]);
}

inline vec3d operator*(const vec3d& v, double t) {
	return t * v;
}

inline vec3d operator/(vec3d v, double t) {
	return (1 / t) * v;
}

inline double dot(const vec3d& u, const vec3d& v) {
	return u.e[0] * v.e[0]
		+ u.e[1] * v.e[1]
		+ u.e[2] * v.e[2];
}

inline vec3d cross(const vec3d& u, const vec3d& v) {
	return vec3d(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

inline vec3d unit_vector(vec3d v) {
	return v / v.length();
}


// random methods
inline vec3f random_in_unit_sphere_f() {
	while (true) {
		auto p = vec3f::random(-1.f, 1.f);
		if (p.length_squared() >= 1.f) continue;
		return p;
	}
}

// random methods
inline vec3d random_in_unit_sphere_d() {
	while (true) {
		auto p = vec3d::random(-1., 1.);
		if (p.length_squared() >= 1.) continue;
		return p;
	}
}

inline vec3f random_unit_vector_on_sphere_f() {
	return unit_vector(random_in_unit_sphere_f());
}

inline vec3d random_unit_vector_on_sphere_d() {
	return unit_vector(random_in_unit_sphere_d());
}


inline vec3f random_in_hemisphere(const vec3f& normal) {
	vec3f in_unit_sphere = random_in_unit_sphere_f();
	if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
		return in_unit_sphere;
	else
		return -in_unit_sphere;
}


inline vec3f reflect(const vec3f& v, const vec3f& n) {
	return v - 2.0f * dot(v, n) * n;
}

inline vec3d reflect(const vec3d& v, const vec3d& n) {
	return v - 2.0 * dot(v, n) * n;
}


inline vec3f refract(const vec3f& uv, const vec3f& n, float etai_over_etae) {
	auto cos_theta = fminf(dot(-uv, n), 1.0f);
	vec3f r_out_prependicular = etai_over_etae * (uv + cos_theta * n);
	vec3f r_out_parallel = -sqrtf(fabsf(1.0f - r_out_prependicular.length_squared())) * n;
	return r_out_parallel + r_out_prependicular;

}

#endif