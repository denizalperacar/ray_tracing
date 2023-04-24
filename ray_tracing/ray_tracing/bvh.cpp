#include "bvh.h"

bool bvh_node::bounding_box(float time0, float time1, aabb& output_box) const {
	output_box = box;
	return true;
}

bool bvh_node::hit(const rayf& r, float t_min, float t_max, hit_record& rec) const {
	if (!box.hit(r, t_min, t_max)) {
		return false;
	}

	bool hit_left = left->hit(r, t_min, t_max, rec);
	bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

	return hit_left || hit_right;
}


