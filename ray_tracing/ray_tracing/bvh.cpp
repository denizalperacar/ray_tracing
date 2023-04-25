#include "bvh.h"

#include <algorithm>

inline bool box_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b, int axis) {
	aabb box_a, box_b;
	if (!a->bounding_box(0.f, 0.f, box_a) || !b->bounding_box(0.f, 0.f, box_b)) {
		std::cerr << "No bounding box in bvh constructor.\n";
	}
	return box_a.min().e[axis] < box_b.min().e[axis];
}

bool box_x_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
	return box_compare(a, b, 0);
}

bool box_y_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
	return box_compare(a, b, 1);
}

bool box_z_compare(const shared_ptr<hittable> a, const shared_ptr<hittable> b) {
	return box_compare(a, b, 2);
}

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

bvh_node::bvh_node(
	const std::vector<shared_ptr<hittable>>& src_objects,
	size_t start, size_t end, float time0, float time1
) {

	std::vector<shared_ptr<hittable>> objects = src_objects;
	int axis = random_int(0, 2);
	auto compare_operator = 
		(axis == 0) ? box_x_compare
		: (axis == 1) ? box_y_compare 
		: box_z_compare;

	size_t object_span = end - start;

	if (object_span == 1) { // base case 1
		left = right = objects[start];
	}
	else if (object_span == 2) { // base case 2
		if (compare_operator(objects[start], objects[start + 1 /*or end*/])) {
			left = objects[start];
			right = objects[start + 1];
		}
		else {
			left = objects[start+1];
			right = objects[start];
		}
	}
	else { // recursion 
		std::sort(objects.begin() + start, objects.begin() + end, compare_operator);

		auto mid = start + object_span / 2;
		left = make_shared<bvh_node>(objects, start, mid, time0, time1);
		right = make_shared<bvh_node>(objects, mid, end, time0, time1);
	}

	aabb box_left, box_right;
	if (!left->bounding_box(time0, time1, box_left)
		|| !right->bounding_box(time0, time1, box_right)) {
		std::cerr << "No bounding box in bvh_node constructor.\n";
	}

	box = surrounding_box(box_left, box_right);
}

