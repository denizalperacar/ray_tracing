#ifndef RAY_TRACER_RANDERER_RENDERING_H_
#define RAY_TRACER_RANDERER_RENDERING_H_

#include "common.h"
#include "camera.h"
#include "hittable.h"
#include "hittable_list.h"
#include <string>

void generate_default_image(std::string file_name);
void generate_image(std::string file_name, camera const &camera, const hittable& world);



#endif
