#ifndef RAY_TRACING_SHAPES_MEDIUM_CONSTANT_MEDIUM_H_
#define RAY_TRACING_SHAPES_MEDIUM_CONSTANT_MEDIUM_H_


#include "common.h"
#include "hittable.h"
#include "texture.h"
#include "materials.h"

class constant_medium : public hittable {

public:
	constant_medium(shared_ptr<hittable> b, float density, shared_ptr<texture> a)
		: boundary(b), 
		  neg_inv_density(-1/density), 
		  phase_function(make_shared<isotropic>(a)) {}

	constant_medium(shared_ptr<hittable> b, float d, color3f c)
		: boundary(b),
		neg_inv_density(-1 / d),
		phase_function(make_shared<isotropic>(c))
	{}

	virtual bool hit(
		const rayf& r, float t_min, float t_max, hit_record& rec
	) const override;

	virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
		return boundary->bounding_box(time0, time1, output_box);
	}

public:
	shared_ptr<hittable> boundary;
	shared_ptr<material> phase_function;
	float neg_inv_density;
};


#endif
