#ifndef RAY_TRACER_VECTOR3_COLOR_H_
#define RAY_TRACER_VECTOR3_COLOR_H_

#include "vec3.h"
#include <fstream>

void write_color(std::ostream& out, color3f pixel_color, int samples_per_pixel) {
	// writes the mapped [0, 255] value of each colow componenet
	auto r = pixel_color.x();
	auto g = pixel_color.y();
	auto b = pixel_color.z();

	// scale the color by the number of samples sent to the sensor
	auto scale = 1.0f / samples_per_pixel;
	r *= scale;
	g *= scale;
	b *= scale;


	out << static_cast<int> (255.999 * clamp(r, 0.f, 0.999f)) << " "
		<< static_cast<int> (255.999 * clamp(g, 0.f, 0.999f)) << " "
		<< static_cast<int> (255.999 * clamp(b, 0.f, 0.999f)) << " "
		<< "\n";
}

void write_color(std::ofstream& out, color3f pixel_color, int samples_per_pixel) {

	// writes the mapped [0, 255] value of each colow componenet
	auto r = pixel_color.x();
	auto g = pixel_color.y();
	auto b = pixel_color.z();

	// scale the color by the number of samples sent to the sensor
	// and gamma corrected for gamma = 2.0
	auto scale = 1.0f / samples_per_pixel;
	r = sqrtf(scale * r);
	g = sqrtf(scale * g);
	b = sqrtf(scale * b);


	out << static_cast<int> (255.999 * clamp(r, 0.f, 0.999f)) << " "
		<< static_cast<int> (255.999 * clamp(g, 0.f, 0.999f)) << " "
		<< static_cast<int> (255.999 * clamp(b, 0.f, 0.999f)) << " "
		<< "\n";
}


#endif
