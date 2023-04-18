#ifndef RAY_TRACER_SCENE_CAMERA_H_
#define RAY_TRACER_SCENE_CAMERA_H_

#include "common.h"
#include "vec3.h"

struct camera {

	float viewport_height{ 2.0f };
	float viewport_width{ static_cast<float> (IMAGE_ASPECT_RATIO) * viewport_height };
	float focal_length{ 1.0f };
	vec3<float> origin = point3f(0.f, 0.f, 0.f);
	vec3<float> horizontal = vec3<float>(viewport_width, 0.f, 0.f);
	vec3<float> vertical = vec3<float>(0.f, viewport_height, 0.f);
	vec3<float> lower_left_corner = (origin
		- horizontal / 2.0f
		- vertical / 2.f
		- vec3<float>(0.f, 0.f, focal_length)
	);

};



#endif