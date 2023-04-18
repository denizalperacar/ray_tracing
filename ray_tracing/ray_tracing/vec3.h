#ifndef RAY_TRACER_VECTOR3_VEC3_H_
#define RAY_TRACER_VECTOR3_VEC3_H_

#include "common.h"
#include <cmath>
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
			e{ 0.f,0.f,0.f };
		}
		else {
			e{ 0.,0.,0. };
		}
	}

	vec3(T e1, T e2, T e3) : e{ e1, e2, e3 } {}

	double x() const { return e[0]; }
	double y() const { return e[1]; }
	double z() const { return e[2]; }

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

public:
	array<T, 3> e;
};

using point3d = vec3<double>;
using point3f = vec3<float>;
using color3d = vec3<double>;
using color3f = vec3<float>;
template <typename T> using color = vec3<T>;


// vec3 utility functions

template <typename T>
inline std::ostream& operator<<(std::ostream& out, const vec3<T>& v) {
	return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

template <typename T>
inline vec3<T> operator+(const vec3<T>& u, const vec3<T>& v) {
	return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

template <typename T>
inline vec3<T> operator-(const vec3<T>& u, const vec3<T>& v) {
	return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

template <typename T>
inline vec3<T> operator*(const vec3<T>& u, const vec3<T>& v) {
	return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

template <typename T>
inline vec3<T> operator*(double t, const vec3<T>& v) {
	return vec3<T>(t * v.e[0], t * v.e[1], t * v.e[2]);
}

template <typename T>
inline vec3<T> operator*(const vec3<T>& v, double t) {
	return t * v;
}

template <typename T>
inline vec3<T> operator/(vec3<T> v, double t) {
	return (1 / t) * v;
}

template <typename T>
inline double dot(const vec3<T>& u, const vec3<T>& v) {
	return u.e[0] * v.e[0]
		+ u.e[1] * v.e[1]
		+ u.e[2] * v.e[2];
}

template <typename T>
inline vec3<T> cross(const vec3<T>& u, const vec3<T>& v) {
	return vec3<T>(u.e[1] * v.e[2] - u.e[2] * v.e[1],
		u.e[2] * v.e[0] - u.e[0] * v.e[2],
		u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

template <typename T>
inline vec3<T> unit_vector(vec3<T> v) {
	return v / v.length();
}


#endif