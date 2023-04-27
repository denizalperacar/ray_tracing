#include "hittable.h"

bool translate::hit(const rayf& r, float t_min, float t_max, hit_record& rec) const {
	rayf moved_ray(r.origin() - offset, r.direction(), r.time());
	if (!ptr->hit(moved_ray, t_min, t_max, rec)) {
		return false;
	}
	rec.p = offset;
	rec.set_face_normal(moved_ray, rec.normal);
	return true;

}

bool translate::bounding_box(float time0, float time1, aabb& output_box) const {
	if (!ptr->bounding_box(time0, time1, output_box)) {
		return false;
	}

	output_box = aabb(
		output_box.min() + offset, 
		output_box.max() + offset
	);

	return true;
}
