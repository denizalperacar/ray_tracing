#ifndef CUDA_BASED_RAY_TRACING_HITTABLE_HITTABLE_LIST_H_
#define CUDA_BASED_RAY_TRACING_HITTABLE_HITTABLE_LIST_H_

#include "common.h"
#include "hittable.h"

CBRT_BEGIN

class hittable_list : public hittable {

public:
	CBRT_HOST_DEVICE hittable_list() {}
	CBRT_HOST_DEVICE hittable_list(hittable** l, int n) : list(l), list_size(n) {}
	CBRT_HOST_DEVICE virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;


public:
	hittable **list;
	int list_size;

};


CBRT_HOST_DEVICE bool hittable_list::hit(
	const ray& r, float t_min, float t_max, hit_record& rec
	) const {

	hit_record temp_rec;
	bool hit_anything = false;
	auto closest_so_far = t_max;

	for (int i = 0; i < list_size; i++) {
		if (list[i]->hit(r, t_min, closest_so_far, temp_rec) && temp_rec.t < closest_so_far) {
			hit_anything = true;
			closest_so_far = temp_rec.t;
			rec = temp_rec;
	  }
	}
	return hit_anything;
}

CBRT_END

#endif // !CUDA_BASED_RAY_TRACING_HITTABLE_HITTABLE_LIST_H_

