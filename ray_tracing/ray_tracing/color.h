#ifndef RAY_TRACER_VECTOR3_COLOR_H_
#define RAY_TRACER_VECTOR3_COLOR_H_

#include "vec3.h"
#include <fstream>

template<typename T>
void write_color(std::ostream& out, color<T> pixel_color) {
	// writes the mapped [0, 255] value of each colow componenet

	out << static_cast<int> (255.999 * pixel_color.x()) << " "
		<< static_cast<int> (255.999 * pixel_color.y()) << " "
		<< static_cast<int> (255.999 * pixel_color.z()) << " "
		<< "\n";
}

template<typename T>
void write_color(std::ofstream& out, color<T> pixel_color) {
	// writes the mapped [0, 255] value of each colow componenet

	out << static_cast<int> (255.999 * pixel_color.x()) << " "
		<< static_cast<int> (255.999 * pixel_color.y()) << " "
		<< static_cast<int> (255.999 * pixel_color.z()) << " "
		<< "\n";
}


#endif
