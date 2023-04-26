#ifndef RAY_TRACING_TEXTURE_IMAGE_TEXTURE_H_
#define RAY_TRACING_TEXTURE_IMAGE_TEXTURE_H_

#include "common.h"
#include "load_images.h"
#include "texture.h"
#include "perlin_noise.h"
#include <string>

class image_texture : public texture {

public:

	const static int bytes_per_pixel = 3;

	image_texture() 
		: data(nullptr), width(0), height(0), bytes_per_scanline(0) {}

	image_texture(const char* filename) {
	
		auto component_per_pixel = bytes_per_pixel;

		data = stbi_load(
			filename, 
			&width, &height,
			&component_per_pixel, 
			component_per_pixel
		);
	
		if (!data) {
			std::cerr << "Error: could not read the image file \'" << filename << "\'.\n";
			width = height = 0;
		}

		bytes_per_scanline = bytes_per_pixel * width;

	}

	~image_texture() {
		delete data;
	}

	virtual color3f value(float u, float v, const vec3f& p) const override {
		if (data == nullptr) {
			return color3f(0.f, 1.f, 1.f);
		}

		u = clamp(u, 0.0f, 1.0f);
		v = 1.0f - clamp(v, 0.0f, 1.0f);

		int i = static_cast<int> (u * width);
		int j = static_cast<int> (v * height);

		if (i >= width) {
			i = width - 1;
			j = height - 1;
		}

		const float color_scale = 1.0f / 255.f;
		auto pixel = data + j * bytes_per_scanline + i * bytes_per_pixel;

		return color3f(
			color_scale * pixel[0],
			color_scale * pixel[1],
			color_scale * pixel[2]
		);

	}


private:

	unsigned char* data;
	int width, height;
	int bytes_per_scanline;

};



#endif
