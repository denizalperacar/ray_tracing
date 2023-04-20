#ifndef RAY_TRACING_SHAPES_SPHERE_SPHERE_H_
#define RAY_TRACING_SHAPES_SPHERE_SPHERE_H_

#include "hittable.h"


class sphere : public hittable {
public:
	sphere() = default;
	sphere(point3f cen, float r) : center(cen), radius(r) { }

	virtual bool hit(const rayf& r, float t_min, float t_max, hit_record& rec) const override;

public: 
	point3f center;
	float radius;

};


#endif

