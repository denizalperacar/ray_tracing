#include "common.h"
#include "rendering.h"
#include "hittable_list.h"
#include "default_scene_generator.h"
#include <string>


int main() {

	point3f lookfrom(13.f, 2.f, 3.f);
	point3f lookat(0.f,0.f,0.f);
	vec3f up(0.f, 1.f, 0.f);
	float dist_to_focus = 10.f;
	float aperature = 0.1f;

	// camera cam(point3f(-2.0f, 2.f, 1.f), point3f(0.f, 0.f, -1.f), vec3f(0.f, 1.f, 0.f), 90.f, IMAGE_ASPECT_RATIO);
	camera cam(lookfrom, lookat, up, 20.f, IMAGE_ASPECT_RATIO, aperature, dist_to_focus, 0.f, 1.0f);
	
	hittable_list world = random_scene(2);

	generate_image("../image.ppm", cam, world);



	return 0;
}