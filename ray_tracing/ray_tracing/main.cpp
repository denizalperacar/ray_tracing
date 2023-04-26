#include "common.h"
#include "rendering.h"
#include "hittable_list.h"
#include "default_scene_generator.h"
#include "rectangle.h"
#include <string>


int main() {

  hittable_list world;
  color3f background(0.f, 0.f, 0.f);
  point3f lookfrom;
  point3f lookat;
  auto vfov = 40.0f;
  auto aperture = 0.0f;

  switch (0) {
  case 1:
    world = random_scene(2);
    background = color3f(0.70f, 0.80f, 1.00f);
    lookfrom = point3f(13.f, 2.f, 3.f);
    lookat = point3f(0.f, 0.f, 0.f);
    vfov = 20.0f;
    aperture = 0.1f;
    break;

  
  case 2:
    world = two_spheres();
    background = color3f(0.70f, 0.80f, 1.00f);
    lookfrom = point3f(13.f, 2.f, 3.f);
    lookat = point3f(0.f, 0.f, 0.f);
    vfov = 20.0f;
    break;

  
  case 3:
    world = two_perlin_spheres();
    background = color3f(0.70f, 0.80f, 1.00f);
    lookfrom = point3f(13.f, 2.f, 3.f);
    lookat = point3f(0.f, 0.f, 0.f);
    vfov = 20.0f;
    break;

  case 4:
    world = earth();
    background = color3f(0.70f, 0.80f, 1.00f);
    lookfrom = point3f(13.f, 2.f, 3.f);
    lookat = point3f(0.f, 0.f, 0.f);
    vfov = 20.0f;
    break;

  default:
  case 5:
    world = simple_light();
    background = color3f(0.f, 0.f, 0.f);
    lookfrom = point3f(26.f, 3.f, 6.f);
    lookat = point3f(0.f, 2.f, 0.f);
    vfov = 20.0f;
    break;
  }

  // Camera

  vec3f up(0.f, 1.f, 0.f);
  auto dist_to_focus = 10.0f;
  int image_height = static_cast<int>(IMAGE_WIDTH / IMAGE_ASPECT_RATIO);

	camera cam(lookfrom, lookat, up, vfov, IMAGE_ASPECT_RATIO, aperture, dist_to_focus, 0.f, 1.0f);
	generate_image("../image.ppm", cam, world, background);

	return 0;
}