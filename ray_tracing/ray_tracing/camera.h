#ifndef RAY_TRACER_SCENE_CAMERA_H_
#define RAY_TRACER_SCENE_CAMERA_H_

#include "common.h"
#include "vec3.h"

class camera {

public:

	camera() {
		float viewport_height{ 2.0f };
		float viewport_width{ static_cast<float> (IMAGE_ASPECT_RATIO) * viewport_height };
		float focal_length{ 1.0f };

		origin_ = point3f(0.f, 0.f, 0.f);
		horizontal_ = vec3<float>(viewport_width, 0.f, 0.f);
		vertical_ = vec3<float>(0.f, viewport_height, 0.f);
		lower_left_corner_ = (origin_
			- horizontal_ / 2.0f
			- vertical_ / 2.f
			- vec3<float>(0.f, 0.f, focal_length)
			);
	}

	rayf get_ray(float u, float v) const {
		return rayf(
			origin_, 
			lower_left_corner_ + u * horizontal_ + v * vertical_ - origin_
		);
	}

private:
	point3f origin_;
	vec3f horizontal_;
	vec3f vertical_;
	point3f lower_left_corner_;
};



#endif