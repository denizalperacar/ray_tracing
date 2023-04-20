#ifndef RAY_TRACER_VECTOR3_VEC3_H_
#define RAY_TRACER_VECTOR3_VEC3_H_

#include "random.h"
#include <iostream>
#include <array>
#include <type_traits>


using std::sqrt;


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


#endif