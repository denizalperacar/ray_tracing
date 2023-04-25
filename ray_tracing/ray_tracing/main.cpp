#include "common.h"
#include "rendering.h"
#include "hittable_list.h"
#include "default_scene_generator.h"
#include <string>


int main() {

  hittable_list world;

  point3f lookfrom;
  point3f lookat;
  auto vfov = 40.0f;
  auto aperture = 0.0f;

  switch (0) {
  case 1:
    world = random_scene(2);
    lookfrom = point3f(13.f, 2.f, 3.f);
    lookat = point3f(0.f, 0.f, 0.f);
    vfov = 20.0f;
    aperture = 0.1f;
    break;

  default:
  case 2:
    world = two_spheres();
    lookfrom = point3f(13.f, 2.f, 3.f);
    lookat = point3f(0.f, 0.f, 0.f);
    vfov = 20.0f;
    break;
  }

  // Camera

  vec3f up(0.f, 1.f, 0.f);
  auto dist_to_focus = 10.0f;
  int image_height = static_cast<int>(IMAGE_WIDTH / IMAGE_ASPECT_RATIO);

	camera cam(lookfrom, lookat, up, 20.f, IMAGE_ASPECT_RATIO, aperture, dist_to_focus, 0.f, 1.0f);
	generate_image("../image.ppm", cam, world);

	return 0;
}