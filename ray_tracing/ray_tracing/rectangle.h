#ifndef RAY_TRACER_SHAPES_RECTANGLE_REACTANGLE_H_
#define RAY_TRACER_SHAPES_RECTANGLE_REACTANGLE_H_

#include "common.h"
#include "hittable.h"

class xy_rect : public hittable {

public:
	xy_rect() = default;

	xy_rect(
		float x0_, float x1_, 
		float y0_, float y1_, float k_, 
		shared_ptr<material> mat) 
		: x0(x0_), x1(x1_), y0(y0_), y1(y1_), k(k_), mp(mat) {}

	virtual bool hit(const rayf& r, float t_min, float t_max, hit_record& rec) const override;
	virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
		// The bounding box must have non-zero width in each dimension, so pad the Z
		// dimension a small amount.
		output_box = aabb(point3f(x0, y0, k - 0.0001f), point3f(x1, y1, k + 0.0001f));
		return true;
	}

public:
	shared_ptr<material> mp;
	float x0, x1, y0, y1, k;

};


class xz_rect : public hittable {

public:
	xz_rect() = default;

	xz_rect(
		float x0_, float x1_,
		float z0_, float z1_, float k_,
		shared_ptr<material> mat)
		: x0(x0_), x1(x1_), z0(z0_), z1(z1_), k(k_), mp(mat) {}

	virtual bool hit(const rayf& r, float t_min, float t_max, hit_record& rec) const override;
	virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
		// The bounding box must have non-zero width in each dimension, so pad the Z
		// dimension a small amount.
		output_box = aabb(point3f(x0, k - 0.0001f, z0), point3f(x1, k + 0.0001f, z1));
		return true;
	}

public:
	shared_ptr<material> mp;
	float x0, x1, z0, z1, k;

};

class yz_rect : public hittable {

public:
	yz_rect() = default;

	yz_rect(
		float y0_, float y1_,
		float z0_, float z1_, float k_,
		shared_ptr<material> mat)
		: y0(y0_), y1(y1_), z0(z0_), z1(z1_), k(k_), mp(mat) {}

	virtual bool hit(const rayf& r, float t_min, float t_max, hit_record& rec) const override;
	virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
		// The bounding box must have non-zero width in each dimension, so pad the Z
		// dimension a small amount.
		output_box = aabb(point3f(k - 0.0001f, y0, z0), point3f(k + 0.0001f, y1, z1));
		return true;
	}

public:
	shared_ptr<material> mp;
	float y0, y1, z0, z1, k;

};

#endif
