#include "default_scene_generator.h"
#include "materials.h"
#include "sphere.h"
#include "moving_sphere.h"
#include "image_texture.h"
#include "rectangle.h"
#include "box.h"


hittable_list random_scene(int n = 11) {

	hittable_list world;

  auto checker = make_shared<checker_texture>(color3f(0.2f, 0.3f, 0.1f), color3f(0.9f,0.9f,0.9f));
	auto ground_material = make_shared<lambertian>(checker);
	world.add(make_shared<sphere>(point3f(0.f, -1000.f, 0.f), 1000.f, ground_material));

  for (int a = -n; a < n; a++) {
    for (int b = -n; b < n; b++) {
      auto choose_mat = random_float();
      point3f center(a + 0.9f * random_float(), 0.2f, b + 0.9f * random_float());

      if ((center - point3f(4.f, 0.2f, 0.f)).length() > 0.9f) {
        std::cout << "new shape added\n";
        shared_ptr<material> sphere_material;

        if (choose_mat < 0.8f) {
          // diffuse
          auto albedo = color3f::random() * color3f::random();
          sphere_material = make_shared<lambertian>(albedo);
          auto center2 = center + vec3f(0.f, random_float(0.f, 0.5f), 0.f);
          if (choose_mat < 0.1f) {
            world.add(make_shared<moving_sphere>(center, center2, 0.f, 1.0f, 0.2f, sphere_material));
          }
          world.add(make_shared<sphere>(center, 0.2f, sphere_material));
        }
        else if (choose_mat < 0.95f) {
          // metal
          auto albedo = color3f::random(0.5f, 1.f);
          auto fuzz = random_float(0.f, 0.5f);
          sphere_material = make_shared<metal>(albedo, fuzz);
          world.add(make_shared<sphere>(center, 0.2f, sphere_material));
        }
        else {
          // glass
          sphere_material = make_shared<dielectric>(1.5f);
          world.add(make_shared<sphere>(center, 0.2f, sphere_material));
        }
      }
    }
  }

  
  auto material1 = make_shared<dielectric>(1.5f);
  world.add(make_shared<sphere>(point3f(0.f, 1.f, 0.f), 1.0f, material1));

  auto material2 = make_shared<lambertian>(color3f(0.4f, 0.2f, 0.1f));
  world.add(make_shared<sphere>(point3f(-4.f, 1.f, 0.f), 1.0f, material2));

  auto material3 = make_shared<metal>(color3f(0.7f, 0.6f, 0.5f), 0.0f);
  world.add(make_shared<sphere>(point3f(4.f, 1.f, 0.f), 1.0f, material3));
  

	return world;
}


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


hittable_list simple_light() {
  hittable_list objects;

  auto pertext = make_shared<noise_texture>(4.f);
  auto difflight = make_shared<diffuse_light>(color3f(4.f, 4.f, 4.f));
  objects.add(make_shared<sphere>(point3f(0.f, -1000.f, 0.f), 1000.f, make_shared<lambertian>(pertext)));
  objects.add(make_shared<sphere>(point3f(0.f, 2.f, 0.f), 2.f, make_shared<lambertian>(pertext)));

  
  objects.add(make_shared<xy_rect>(3.f, 5.f, 1.f, 3.f, -2.f, difflight));

  return objects;
}


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

  objects.add(make_shared<box>(point3f(130.f, 0.f, 65.f), point3f(295.f, 165.f, 230.f), white));
  objects.add(make_shared<box>(point3f(265.f, 0.f, 295.f), point3f(430.f, 330.f, 460.f), white));
  return objects;
}
