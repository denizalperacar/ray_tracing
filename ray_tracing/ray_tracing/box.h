#ifndef RAY_TRACING_SHAPES_RECTANGLE_BOX_H_
#define RAY_TRACING_SHAPES_RECTANGLE_BOX_H_

#include "common.h"
#include "rectangle.h"
#include "hittable_list.h"

class box : public hittable {
public:
	box() {}
	box(const point3f& p0, const point3f& p1, shared_ptr<material> ptr);

  virtual bool hit(const rayf& r, float t_min, float t_max, hit_record& rec) const override;

  virtual bool bounding_box(float time0, float time1, aabb& output_box) const override {
    output_box = aabb(box_min, box_max);
    return true;
  }

public:
  point3f box_min;
  point3f box_max;
  hittable_list sides;

};

#endif

