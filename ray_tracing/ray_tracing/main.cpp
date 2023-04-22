#include "common.h"
#include "rendering.h"
#include "hittable_list.h"
#include "sphere.h"
#include "materials.h"
#include <string>


int main() {

	point3f lookfrom(3.f, 3.f, 2.f);
	point3f lookat(0.f,0.f,-1.f);
	vec3f up(0.f, 1.f, 0.f);
	float dist_to_focus = (lookfrom - lookat).length();
	float aperature = 2.f;

	// camera cam(point3f(-2.0f, 2.f, 1.f), point3f(0.f, 0.f, -1.f), vec3f(0.f, 1.f, 0.f), 90.f, IMAGE_ASPECT_RATIO);
	camera cam(lookfrom, lookat, up, 20.f, IMAGE_ASPECT_RATIO, aperature, dist_to_focus);
	
	hitable_list world;

	auto material_ground = make_shared<lambertian>(color3f(0.8f, 0.8f, 0.0f));
	auto material_center = make_shared<lambertian>(color3f(0.1f, 0.2f, 0.5f));
	auto material_left = make_shared<dielectric>(1.5f);
	auto material_right = make_shared<metal>(color3f(0.1f, 0.2f, 0.5f), 0.0f);

	world.add(make_shared<sphere>(point3f(0.0f, -100.5f, -1.0f), 100.0f, material_ground));
	world.add(make_shared<sphere>(point3f(0.0f, 0.0f, -1.0f), 0.5f, material_right));
	world.add(make_shared<sphere>(point3f(-1.0f, 0.0f, -1.0f), 0.5f, material_left));
	world.add(make_shared<sphere>(point3f(-1.0f, 0.0f, -1.0f), -0.45f, material_left));
	world.add(make_shared<sphere>(point3f(1.0f, 0.0f, -1.0f), 0.5f, material_right));


	generate_image("../image.ppm", cam, world);



	return 0;
}