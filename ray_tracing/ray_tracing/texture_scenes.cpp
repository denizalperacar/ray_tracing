#include "default_scene_generator.h"
#include "materials.h"
#include "sphere.h"
#include "image_texture.h"

hittable_list two_spheres() {
  hittable_list objects;

  auto checker = make_shared<checker_texture>(color3f(0.2f, 0.3f, 0.1f), color3f(0.9f, 0.9f, 0.9f));

  objects.add(make_shared<sphere>(point3f(0.f, -10.f, 0.f), 10.f, make_shared<lambertian>(checker)));
  objects.add(make_shared<sphere>(point3f(0.f, 10.f, 0.f), 10.f, make_shared<lambertian>(checker)));

  return objects;
}

hittable_list two_perlin_spheres() {
  hittable_list objects;

  auto checker = make_shared<noise_texture>(4.0f);

  objects.add(make_shared<sphere>(point3f(0.f, -1000.f, 0.f), 1000.f, make_shared<lambertian>(checker)));
  objects.add(make_shared<sphere>(point3f(0.f, 2.f, 0.f), 2.f, make_shared<lambertian>(checker)));

  return objects;
}

hittable_list earth() {
  auto earth_texture = make_shared<image_texture>("../earth.jpg");
  auto earth_surface = make_shared<lambertian>(earth_texture);
  auto globe = make_shared<sphere>(point3f(0.f, 0.f, 0.f), 2.f, earth_surface);

  return hittable_list(globe);
}