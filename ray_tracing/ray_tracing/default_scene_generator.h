#ifndef RAY_TRACING_SCENE_DEFAULT_SCENE_GENERATOR_H_
#define RAY_TRACING_SCENE_DEFAULT_SCENE_GENERATOR_H_

#include "common.h"
#include "hittable_list.h"


hittable_list random_scene(int n);
hittable_list two_spheres();
hittable_list two_perlin_spheres();

#endif
