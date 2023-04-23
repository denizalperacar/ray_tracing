#ifndef RAY_TRACER_SCENE_CAMERA_H_
#define RAY_TRACER_SCENE_CAMERA_H_

#include "common.h"
#include "vec3.h"

class camera {

public:
	camera() = delete;
	camera(
		point3f lookfrom,
		point3f lookat,
		vec3f up_vector,
		float vertical_field_of_view,
		float aspect_ratio,
		float aperature,
		float focus_distance,
		float _time0 = 0.f,
		float _time1 = 0.f
		) {

		float theta = degrees_to_radians(vertical_field_of_view);
		float h = tanf(theta / 2.f);
		float viewport_height{2.0f * h};
		float viewport_width{ static_cast<float> (aspect_ratio) * viewport_height };
		
		w_ = unit_vector(lookfrom - lookat);
		u_ = unit_vector(cross(up_vector, w_));
		v_ = unit_vector(cross(w_, u_));

		origin_ = lookfrom;
		horizontal_ = focus_distance * viewport_width * u_;
		vertical_ = focus_distance * viewport_height * v_;
		lower_left_corner_ = (
			origin_ - horizontal_ / 2.0f 
			- vertical_ / 2.f - focus_distance * w_
		);
		lens_radius_ = aperature / 2.f;
		time0 = _time0;
		time1 = _time1;
	}

	rayf get_ray(float s, float t) const {

		vec3f rd = lens_radius_ * random_in_unit_disk();
		vec3f offset = u_ * rd.x() + v_ * rd.y();

		return rayf(
			origin_ + offset,
			lower_left_corner_ + s * horizontal_ + t * vertical_ - origin_ - offset,
			random_float(time0, time1)
		);
	}

private:
	point3f origin_;
	vec3f horizontal_;
	vec3f vertical_;
	point3f lower_left_corner_;
	vec3f u_, v_, w_;
	float lens_radius_;
	float time0, time1;
};



#endif