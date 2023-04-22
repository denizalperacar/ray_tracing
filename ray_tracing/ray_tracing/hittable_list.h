#ifndef RAY_TRACING_SHAPES_HITTABLE_LIST_H_
#define RAY_TRACING_SHAPES_HITTABLE_LIST_H_

#include "common.h"
#include "hittable.h"

#include <vector>

class hittable_list : public hittable {

public:
	hittable_list() {}
	hittable_list(shared_ptr<hittable> object) { add(object); }

	void clear() { objects.clear(); }
	void add(shared_ptr<hittable> object) { objects.push_back(object); }

	virtual bool hit(
		const rayf& r, float t_min, float t_max, hit_record& rec 
	) const override;

public:
	std::vector<shared_ptr<hittable>> objects;
};

#endif