#ifndef RAY_TRACER_RAY_RAY_COLOR_H_
#define RAY_TRACER_RAY_RAY_COLOR_H_

#include "ray.h"
#include "vec3.h"
#include "hittable_list.h"

color3f ray_color(const rayf& r, const hittable &world, int depth);
color3d ray_color(const rayd& r);


#endif
