#include "default_scene_generator.h"
#include "bvh.h"
#include "box.h"
#include "moving_sphere.h"
#include "sphere.h"
#include "constant_medium.h"
#include "image_texture.h"

hittable_list book_2_final_scene() {
  hittable_list boxes1;
  auto ground = make_shared<lambertian>(color3f(0.48f, 0.83f, 0.53f));

  const int boxes_per_side = 20;
  for (int i = 0; i < boxes_per_side; i++) {
    for (int j = 0; j < boxes_per_side; j++) {
      auto w = 100.0f;
      auto x0 = -1000.0f + i * w;
      auto z0 = -1000.0f + j * w;
      auto y0 = 0.0f;
      auto x1 = x0 + w;
      auto y1 = random_float(1.f, 101.f);
      auto z1 = z0 + w;

      boxes1.add(make_shared<box>(point3f(x0, y0, z0), point3f(x1, y1, z1), ground));
    }
  }

  hittable_list objects;

  objects.add(make_shared<bvh_node>(boxes1, 0.f, 1.f));

  auto light = make_shared<diffuse_light>(color3f(7.f, 7.f, 7.f));
  objects.add(make_shared<xz_rect>(123.f, 423.f, 147.f, 412.f, 554.f, light));

  auto center1 = point3f(400.f, 400.f, 200.f);
  auto center2 = center1 + vec3f(30.f, 0.f, 0.f);
  auto moving_sphere_material = make_shared<lambertian>(color3f(0.7f, 0.3f, 0.1f));
  objects.add(make_shared<moving_sphere>(center1, center2, 0.f, 1.f, 50.f, moving_sphere_material));

  objects.add(make_shared<sphere>(point3f(260.f, 150.f, 45.f), 50.f, make_shared<dielectric>(1.5f)));
  objects.add(make_shared<sphere>(
    point3f(0.f, 150.f, 145.f), 50.f, make_shared<metal>(color3f(0.8f, 0.8f, 0.9f), 1.0f)
  ));

  auto boundary = make_shared<sphere>(point3f(360.f, 150.f, 145.f), 70.f, make_shared<dielectric>(1.5f));
  objects.add(boundary);
  objects.add(make_shared<constant_medium>(boundary, 0.2f, color3f(0.2f, 0.4f, 0.9f)));
  boundary = make_shared<sphere>(point3f(0.f, 0.f, 0.f), 5000.f, make_shared<dielectric>(1.5f));
  objects.add(make_shared<constant_medium>(boundary, .0001f, color3f(1.f, 1.f, 1.f)));

  auto emat = make_shared<lambertian>(make_shared<image_texture>("../earth.jpg"));
  objects.add(make_shared<sphere>(point3f(400.f, 200.f, 400.f), 100.f, emat));
  auto pertext = make_shared<noise_texture>(0.1f);
  objects.add(make_shared<sphere>(point3f(220.f, 280.f, 300.f), 80.f, make_shared<lambertian>(pertext)));

  hittable_list boxes2;
  auto white = make_shared<lambertian>(color3f(.73f, .73f, .73f));
  int ns = 1000;
  for (int j = 0; j < ns; j++) {
    boxes2.add(make_shared<sphere>(point3f::random(0.f, 165.f), 10.f, white));
  }

  objects.add(make_shared<translate>(
    make_shared<rotate_y>(
      make_shared<bvh_node>(boxes2, 0.0f, 1.0f), 15.f),
    vec3f(-100.f, 270.f, 395.f)
  )
  );

  return objects;
}