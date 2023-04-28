#include "default_scene_generator.h"
#include "materials.h"
#include "box.h"
#include "constant_medium.h"


hittable_list cornell_box() {
  hittable_list objects;

  auto red = make_shared<lambertian>(color3f(.65f, .05f, .05f));
  auto white = make_shared<lambertian>(color3f(.73f, .73f, .73f));
  auto green = make_shared<lambertian>(color3f(.12f, .45f, .15f));
  auto light = make_shared<diffuse_light>(color3f(15.f, 15.f, 15.f));

  objects.add(make_shared<yz_rect>(0.f, 555.f, 0.f, 555.f, 555.f, green));
  objects.add(make_shared<yz_rect>(0.f, 555.f, 0.f, 555.f, 0.f, red));
  objects.add(make_shared<xz_rect>(213.f, 343.f, 227.f, 332.f, 554.f, light));
  objects.add(make_shared<xz_rect>(0.f, 555.f, 0.f, 555.f, 0.f, white));
  objects.add(make_shared<xz_rect>(0.f, 555.f, 0.f, 555.f, 555.f, white));
  objects.add(make_shared<xy_rect>(0.f, 555.f, 0.f, 555.f, 555.f, white));

  shared_ptr<hittable> box1 = make_shared<box>(point3f(0.f, 0.f, 0.f), point3f(165.f, 330.f, 165.f), white);
  box1 = make_shared<rotate_y>(box1, 15.f);
  box1 = make_shared<translate>(box1, vec3f(265.f, 0.f, 295.f));

  shared_ptr<hittable> box2 = make_shared<box>(point3f(0.f, 0.f, 0.f), point3f(165.f, 165.f, 165.f), white);
  box2 = make_shared<rotate_y>(box2, -18.f);
  box2 = make_shared<translate>(box2, vec3f(130.f, 0.f, 65.f));

  objects.add(make_shared<constant_medium>(box1, 0.01f, color3f(0.f, 0.f, 0.f)));
  objects.add(make_shared<constant_medium>(box2, 0.01f, color3f(1.f, 1.f, 1.f)));

  return objects;
}