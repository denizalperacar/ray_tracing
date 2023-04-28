#include "common.h"
#include "rendering.h"
#include "hittable_list.h"
#include "default_scene_generator.h"
#include "rectangle.h"
#include <string>

#include <chrono>


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


  case 5:
    world = simple_light();
    background = color3f(0.f, 0.f, 0.f);
    lookfrom = point3f(26.f, 3.f, 6.f);
    lookat = point3f(0.f, 2.f, 0.f);
    vfov = 20.0f;
    break;

  case 6:
    world = cornell_box();
    background = color3f(0.f, 0.f, 0.f);
    lookfrom = point3f(278.f, 278.f, -800.f);
    lookat = point3f(278.f, 278.f, 0.f);
    vfov = 40.0f;
    break;

  default:
  case 7:
    world = book_2_final_scene();
    background = color3f(0.f, 0.f, 0.f);
    lookfrom = point3f(478.f, 278.f, -600.f);
    lookat = point3f(278.f, 278.f, 0.f);
    vfov = 40.0f;
    break;

  }

  // Camera

  vec3f up(0.f, 1.f, 0.f);
  auto dist_to_focus = 10.0f;
  int image_height = static_cast<int>(IMAGE_WIDTH / IMAGE_ASPECT_RATIO);

	camera cam(lookfrom, lookat, up, vfov, IMAGE_ASPECT_RATIO, aperture, dist_to_focus, 0.f, 1.0f);
  auto start = std::chrono::high_resolution_clock::now();
  //generate_image("../image.ppm", cam, world, background);
  //auto stop1 = std::chrono::high_resolution_clock::now();
  generate_image_parallel_for("../image.ppm", cam, world, background);
  auto stop2 = std::chrono::high_resolution_clock::now();
  //std::cerr << std::chrono::duration_cast<std::chrono::microseconds>(stop1 - start).count() << "\n";
  std::cerr << static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(stop2 - start).count()) / 1000000.f << "\n";
	return 0;
}