#ifndef RAY_TRACING_SHAPES_BVH_BVH_H_
#define RAY_TRACING_SHAPES_BVH_BVH_H_

#include "common.h"

#include "hittable.h"
#include "hittable_list.h"

class bvh_node : public hittable {

public:
	bvh_node() = default;
	bvh_node(const hittable_list& list, float time0, float time1) 
		: bvh_node(bvh_node(list.objects, 0, list.objects.size(), time0, time1)) {}

	bvh_node(
		const std::vector<shared_ptr<hittable>>& src_objects,
		size_t start, size_t end, float time0, float time1
	);

	virtual bool hit(const rayf& r, float t_min, float t_max, hit_record& rec) const override;
	virtual bool bounding_box(float time0, float time1, aabb& output_box) const override;

public:
	shared_ptr<hittable> left;
	shared_ptr<hittable> right;
	aabb box;

};


#endif
