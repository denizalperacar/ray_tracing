#ifndef CUDA_BASED_RAY_TRACING_RAY_RAY_H_
#define CUDA_BASED_RAY_TRACING_RAY_RAY_H_

#include "namespaces.h"
#include "vector.h"
#include "helper.h"

CBRT_BEGIN


class ray {

public:
	CBRT_HOST_DEVICE ray() { }
	
	CBRT_HOST_DEVICE ray(const point3f& origin, const vec3f& direction) : orig(origin), dir(direction) {}
	
	CBRT_HOST_DEVICE point3f origin() const { return orig; }
	
	CBRT_HOST_DEVICE vec3f direction() const { return dir; }
	
	CBRT_HOST_DEVICE point3f at(float t) const {
		return orig + t * dir;
	}

public:
	point3f orig;
	vec3f dir;
};

CBRT_END

#endif // !CUDA_BASED_RAY_TRACING_RAY_RAY_H_

