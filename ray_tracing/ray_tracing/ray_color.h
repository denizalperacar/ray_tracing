#ifndef RAY_TRACER_RAY_RAY_COLOR_H_
#define RAY_TRACER_RAY_RAY_COLOR_H_

#include "ray.h"
#include "vec3.h"
#include "simple_shapes.h"

color3f ray_color(const rayf& r);
color3d ray_color(const rayd& r);


#endif
