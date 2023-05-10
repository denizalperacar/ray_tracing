#ifndef CUDA_BASED_RAY_TRACING_HITTABLE_H_
#define CUDA_BASED_RAY_TRACING_HITTABLE_H_

#include "common.h"

CBRT_BEGIN

struct hit_record {
  point3f p;
  vec3f normal;
  float t;

  bool front_face;

  CBRT_HOST_DEVICE inline void set_face_normal(const ray& r, const vec3f& outward_normal) {
    front_face = dot(r.direction(), outward_normal) < 0;
    normal = front_face ? outward_normal : -outward_normal;
  }

};

class hittable {
public:
  CBRT_HOST_DEVICE virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

CBRT_END

#endif
