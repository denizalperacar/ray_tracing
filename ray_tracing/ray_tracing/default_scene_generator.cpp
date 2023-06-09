#include "default_scene_generator.h"
#include "materials.h"
#include "sphere.h"
#include "moving_sphere.h"


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

