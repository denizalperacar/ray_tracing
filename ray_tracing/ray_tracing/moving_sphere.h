#ifndef RAY_TRACING_SHAPES_SPHERE_MOVING_SHPERE_H_
#define RAY_TRACING_SHAPES_SPHERE_MOVING_SHPERE_H_

#include "common.h"
#include "hittable.h"

class moving_sphere : public hittable {
public:
	moving_sphere() = default;

	moving_sphere(
		point3f center_0, point3f center_1,
		float _time0, float _time1,
		float r, shared_ptr<material> m
	) : center0(center_0), center1(center_1),
		time0(_time0), time1(_time1), radius(r),
		mat_ptr(m) {};

	virtual bool hit(
		const rayf& r, float t_min, float t_max, hit_record& rec
	) const override;

	point3f center(float time) const;

public:
	point3f center0, center1;
	float time0, time1;
	float radius;
	shared_ptr<material> mat_ptr;
};


#endif;
