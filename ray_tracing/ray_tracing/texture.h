#ifndef RAY_TRACING_TEXTURE_TEXTURE_H_
#define RAY_TRACING_TEXTURE_TEXTURE_H_

#include "common.h"
#include "perlin_noise.h"

/*
@brief: An abstract calls to query texture from the derived texture 
classes
float u, v: U, V coordinates of the surface
*/
class texture {
public:
	virtual color3f value(float u, float v, const point3f& p) const = 0;
};


class solid_color : public texture {
public:
	solid_color() = default;
	solid_color(color3f c) : color_value(c) {}
	solid_color(float red, float green, float blue) : solid_color(color3f(red, green, blue)) {}

	virtual color3f value(float u, float v, const point3f& p) const override {
		return color_value;
	}

private:
	color3f color_value;

};

class checker_texture : public texture {

public:
	checker_texture() = default;
	checker_texture(
		shared_ptr<texture> even_tile_texture, 
		shared_ptr<texture> odd_tile_texture) 
		: odd(odd_tile_texture), even(even_tile_texture) {}
	checker_texture(color3f c_even, color3f c_odd) 
		: even(make_shared<solid_color>(c_even)), 
		odd(make_shared<solid_color>(c_odd)) {}

	virtual color3f value(float u, float v, const point3f& p) const override {
		float sines = sinf(frequency * p.x()) * sinf(frequency * p.y()) * sinf(frequency * p.z());
		if (sines < 0) {
			return odd->value(u, v, p);
		}
		else {
			return even->value(u, v, p);
		}
	}

public:
	shared_ptr<texture> odd;
	shared_ptr<texture> even;
	float frequency = 10.f;
};


class noise_texture : public texture {
public:
	noise_texture() = default;
	noise_texture(float noise_scale) : scale(noise_scale) {}


	virtual color3f value(float u, float v, const point3f& p) const override {
		return color3f(1.f, 1.f, 1.f) * 0.5f * (1 + noise.noise(scale * p));
	}
private:
	perlin noise;
	float scale;
};

#endif