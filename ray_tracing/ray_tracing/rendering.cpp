#include "rendering.h"
#include "vec3.h"
#include "color.h"
#include "ray.h"

#include <fstream>
#include <exception>
#include <iostream>



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
				write_color(image, pixel_color);
			}
		}
	} 
	catch (std::exception &e) {
		std::cout << e.what() << std::endl;
	}
	std::cerr << "\nDone.\n";
}

void generate_image(std::string file_name, camera const& camera)
{
	std::ofstream image;
	try {
		image.open(file_name);
		image << "P3\n" << IMAGE_WIDTH << " " << IMAGE_HEIGHT << "\n255\n";
		for (int j = IMAGE_HEIGHT - 1; j >= 0; --j) {
			std::cerr << "\rScanlines remaining: " << j << " " << std::flush;
			for (int i = 0; i < IMAGE_WIDTH; i++) {
				auto u = static_cast<float>(i) / (IMAGE_WIDTH - 1);
				auto v = static_cast<float>(j) / (IMAGE_HEIGHT - 1);
				rayf r(
					camera.origin,
					camera.lower_left_corner 
					+ u * camera.horizontal 
					+ v * camera.vertical 
					- camera.origin
				);
				color3f pixel_color{ ray_color(r) };
				write_color(image, pixel_color);
			}
		}
	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
	}
	std::cerr << "\nDone.\n";
}
