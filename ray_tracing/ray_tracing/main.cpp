#include "common.h"
#include "rendering.h"
#include "hittable_list.h"
#include "sphere.h"
#include <string>


int main() {

	camera cam;
	
	hitable_list world;
	world.add(make_shared<sphere>(point3f(0.f,0.f,-1.f), 0.5f));
	world.add(make_shared<sphere>(point3f(0.f, -100.5f, -1.f), 100.f));


	generate_image("../image.ppm", cam, world);



	return 0;
}