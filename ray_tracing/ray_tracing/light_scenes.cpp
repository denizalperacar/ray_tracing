#include "default_scene_generator.h"
#include "materials.h"
#include "sphere.h"
#include "rectangle.h"


hittable_list simple_light() {
  hittable_list objects;

  auto pertext = make_shared<noise_texture>(4.f);
  auto difflight = make_shared<diffuse_light>(color3f(4.f, 4.f, 4.f));
  objects.add(make_shared<sphere>(point3f(0.f, -1000.f, 0.f), 1000.f, make_shared<lambertian>(pertext)));
  objects.add(make_shared<sphere>(point3f(0.f, 2.f, 0.f), 2.f, make_shared<lambertian>(pertext)));


  objects.add(make_shared<xy_rect>(3.f, 5.f, 1.f, 3.f, -2.f, difflight));

  return objects;
}
