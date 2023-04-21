#ifndef RAY_TRACING_SHAPES_HITTABLE_H_
#define RAY_TRACING_SHAPES_HITTABLE_H_

#include "common.h"

class material;

struct hit_record {
	point3f p;
	vec3f normal;
	float t;
	bool front_face;
	shared_ptr < material > mat_ptr;

	inline void set_face_normal(const rayf& r, const vec3f& outward_normal) {
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};


class hittable {
public:
	virtual bool hit(const rayf& r, float t_min, float t_max, hit_record& rec) const = 0;
};


#endif
