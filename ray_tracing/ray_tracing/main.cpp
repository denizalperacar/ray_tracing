#include "common.h"
#include "rendering.h"
#include "hittable_list.h"
#include "sphere.h"
#include "materials.h"
#include <string>


int main() {

	camera cam;
	
	hitable_list world;
	
	auto material_ground = make_shared<lambertian>(color3f(0.8f, 0.8f, 0.0f));
	auto material_center = make_shared<lambertian>(color3f(0.1f, 0.2f, 0.5f));
	// auto material_left = make_shared<metal>(color3f(0.8f, 0.8f, 0.8f), 0.3f);
	// auto material_center = make_shared<dielectric>(1.5f);
	auto material_left = make_shared<dielectric>(1.5f);
	auto material_right = make_shared<metal>(color3f(0.8f, 0.6f, 0.2f), 1.0f);
	
	world.add(make_shared<sphere>(point3f(0.0f, -100.5f, -1.0f), 100.0f, material_ground));
	world.add(make_shared<sphere>(point3f(0.0f, 0.0f, -1.0f), 0.5f, material_center));
	world.add(make_shared<sphere>(point3f(-1.0f, 0.0f, -1.0f), 0.5f, material_left));
	world.add(make_shared<sphere>(point3f(-1.0f, 0.0f, -1.0f), -0.4f, material_left));
	world.add(make_shared<sphere>(point3f(1.0f, 0.0f, -1.0f), 0.5f, material_right));


	generate_image("../image.ppm", cam, world);



	return 0;
}