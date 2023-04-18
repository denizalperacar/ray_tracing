#include "rendering.h"

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
				auto r = static_cast<double>(i) / static_cast<double>(IMAGE_WIDTH - 1);
				auto g = static_cast<double>(j) / static_cast<double>(IMAGE_HEIGHT - 1);
				auto b = 0.25;

				int ir = static_cast<int>(255.0 * r);
				int ig = static_cast<int>(255.0 * g);
				int ib = static_cast<int>(255.0 * b);

				image << ir << " " << ig << " " << ib << "\n";
			}
		}
	} 
	catch (std::exception &e) {
		std::cout << e.what() << std::endl;
	}
}
