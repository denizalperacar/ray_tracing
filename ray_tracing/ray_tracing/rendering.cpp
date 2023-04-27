#include "rendering.h"
#include "vec3.h"
#include "color.h"
#include "ray.h"
#include "ray_color.h"

#include <fstream>
#include <exception>
#include <iostream>
#include <thread>



/*
@brief: The pixels are written out in rows with pixels left to right.
The rows are written out from top to bottom.
By convention, each of the red/green/blue components range from 0.0 to 1.0. 
We will relax that later when we internally use high dynamic range, but before 
output we will tone map to the zero to one range, so this code won’t change.
Red goes from fully off (black) to fully on (bright red) from left to right, 
and green goes from black at the bottom to fully on at the top. 
Red and green together make yellow so we should expect the upper right corner to be yellow.
*/
void generate_default_image(std::string file_name)
{
	std::ofstream image;
	try {
		image.open(file_name);
		image << "P3\n" << IMAGE_WIDTH << " " << IMAGE_HEIGHT << "\n255\n";
		for (int j = IMAGE_HEIGHT - 1; j >= 0; --j) {
			std::cerr << "\rScanlines remaining: " << j << " " << std::flush;
			for (int i = 0; i < IMAGE_WIDTH; i++) {
				color3f pixel_color(
					static_cast<float>(i) / (IMAGE_WIDTH - 1), 
					static_cast<float>(j) / (IMAGE_HEIGHT - 1), 
					0.25
				);
				write_color(image, pixel_color, SAMPLES_PER_PIXEL);
			}
		}
	} 
	catch (std::exception &e) {
		std::cout << e.what() << std::endl;
	}
	std::cerr << "\nDone.\n";
}

void generate_image(std::string file_name, camera const& camera, const hittable& world, const color3f& background)
{
	std::ofstream image;
	try {
		image.open(file_name);
		image << "P3\n" << IMAGE_WIDTH << " " << IMAGE_HEIGHT << "\n255\n";
		for (int j = IMAGE_HEIGHT - 1; j >= 0; --j) {
			std::cerr << "\rScanlines remaining: " << j << " " << std::flush;
			for (int i = 0; i < IMAGE_WIDTH; i++) {
				color3f pixel_color{0.f, 0.f, 0.f};
				for (int s = 0; s < SAMPLES_PER_PIXEL; ++s) {
					auto u = static_cast<float>(i + random_float()) / (IMAGE_WIDTH - 1);
					auto v = static_cast<float>(j + random_float()) / (IMAGE_HEIGHT - 1);
					rayf r = camera.get_ray(u, v);
					pixel_color += ray_color(r, background, world, MAX_DEPTH) ;
				}
				write_color(image, pixel_color, SAMPLES_PER_PIXEL);
			}
		}
	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
	}
	std::cerr << "\nDone.\n";
}


color3f trace_ray(int i, int j, int start, int end, camera const& camera, const hittable& world, const color3f& background) {
	color3f captured_color{ 0.f,0.f,0.f };
	for (int k = start; k < end; k++) {
		auto u = static_cast<float>(i + random_float()) / (IMAGE_WIDTH - 1);
		auto v = static_cast<float>(j + random_float()) / (IMAGE_HEIGHT - 1);
		rayf r = camera.get_ray(u, v);
		captured_color += ray_color(r, background, world, MAX_DEPTH);
	}
	return captured_color;
}

void generate_image_parallel_for(std::string file_name, camera const& camera, const hittable& world, const color3f& background)
{
	std::ofstream image;
	try {
		image.open(file_name);
		image << "P3\n" << IMAGE_WIDTH << " " << IMAGE_HEIGHT << "\n255\n";
		for (int j = IMAGE_HEIGHT - 1; j >= 0; --j) {
			std::cerr << "\rScanlines remaining: " << j << " " << std::flush;
			for (int i = 0; i < IMAGE_WIDTH; i++) {
				std::vector<color3f> results(NUMTHREADS);
				std::vector<std::thread> threads;

				int chunk_size = SAMPLES_PER_PIXEL / NUMTHREADS;
				for (int t = 0; t < NUMTHREADS; t++) {
					int start = t * chunk_size;
					int end = (t + 1) * chunk_size;
					results[t] = color3f(0.f, 0.f, 0.f);
					threads.emplace_back([t, i, j, start, end, &results, &camera, &world, &background]()
						{ results[t] = trace_ray(i, j, start, end, camera, world, background);  }
					);
				}
				
				// wait for all threads to finish
				for (int t = 0; t < NUMTHREADS; t++) {
					threads[t].join();
				}

				color3f pixel_color{0.f,0.f,0.f};
				for (int t = 0; t < NUMTHREADS; t++) {
					pixel_color += results[t];
				}


				write_color(image, pixel_color, SAMPLES_PER_PIXEL);
			}
		}
	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
	}
	std::cerr << "\nDone.\n";
}
