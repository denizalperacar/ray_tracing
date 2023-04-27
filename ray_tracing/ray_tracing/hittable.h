#ifndef RAY_TRACING_SHAPES_HITTABLE_H_
#define RAY_TRACING_SHAPES_HITTABLE_H_

#include "common.h"
#include "AABB.h"

class material;

struct hit_record {
	point3f p;
	vec3f normal;
	float t;
	float u;
	float v;
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
	virtual bool bounding_box(float time0, float time1, aabb& output_box) const = 0;
};


class translate : public hittable {
public:

	translate(shared_ptr<hittable> p, const vec3f& dispalcement) 
		: ptr(p), offset(dispalcement) {}

	virtual bool hit(
		const rayf& r, float t_min, float t_max, hit_record& rec) const override;

	virtual bool bounding_box(float time0, float time1, aabb& output_box) const override;

public:
	shared_ptr<hittable> ptr;
	vec3f offset;
};

class rotate_y : public hittable {
public:
	rotate_y(shared_ptr<hittable> p, float angle);
	virtual bool hit(
		const rayf& r, float time0, float time1, hit_record& rec) const override;

	virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
		output_box = bbox;
		return has_box;
	}

public:
	shared_ptr<hittable> ptr;
	float sin_theta;
	float cos_theta;
	bool has_box;
	aabb bbox;
};



#endif
