#pragma once

#include "common.h"
#include "sphere.h"
#include "hittable_list.h"

CBRT_BEGIN

CBRT_KERNEL void two_spheres(hittable** list, hittable** world) {
	
	uint32_t i = 0;
	list[i++] = new sphere(point3f(0.f, 0.f, -1.0f), 0.5f);
	list[i++] = new sphere(point3f(0.f, -100.5f, -1.f), 100.f);
	*world = new hittable_list(list, i);
}


CBRT_END

