#ifndef CUDA_BASED_RAY_TRACING_VECTOR_GET_COLOR_H_
#define CUDA_BASED_RAY_TRACING_VECTOR_GET_COLOR_H_

#include "common.h"

CBRT_BEGIN

CBRT_HOST_DEVICE render_color get_color(color3f& pixel_color, int samples_per_pixel) {
	// writes the mapped [0, 255] value of each colow componenet
	float r = pixel_color.x();
	float g = pixel_color.y();
	float b = pixel_color.z();

	// scale the color by the number of samples sent to the sensor
	float scale = 1.0f / samples_per_pixel;
	r *= scale;
	g *= scale;
	b *= scale;

	render_color c;

	c.r = (uint8_t)(255.999 * r);
	c.g = (uint8_t)(255.999 * g);
	c.b = (uint8_t)(255.999 * b);

	return c;
	
}


CBRT_END

#endif
