#ifndef RAY_TRACING_MATERIAL_MATERIAL_H_
#define RAY_TRACING_MATERIAL_MATERIAL_H_

#include "common.h"
#include "texture.h"

struct hit_record;


class material {
public:
	virtual bool scatter(
		const rayf& r_in, const hit_record& rec, color3f& attenuation, rayf& scattered
	) const = 0;
};

class lambertian : public material {
public:
	lambertian(const color3f& a) : albedo(make_shared<solid_color>(a)) {}
	lambertian(shared_ptr<texture> a) : albedo(a) {}

	virtual bool scatter(
		const rayf& r_in, const hit_record& rec, color3f& attenuation, rayf& scattered
	) const override {
		auto scatter_direction = rec.normal + random_unit_vector_on_sphere_f();

		if (scatter_direction.near_zero()) {
			scatter_direction = rec.normal;
		}

		scattered = rayf(rec.p, scatter_direction, r_in.time());
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		return true;
	}

public:
	shared_ptr<texture> albedo;
};


class metal : public material {

public:
	metal(const color3f& a, float f) : albedo(a), fuzz(f < 1.0f ? f : 1.0f) {}

	virtual bool scatter(
		const rayf& r_in, const hit_record& rec, color3f& attenuation, rayf& scattered
	) const override {
		vec3f reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		scattered = rayf(rec.p, reflected + fuzz * random_in_unit_sphere_f(), r_in.time());
		attenuation = albedo;
		return (dot(scattered.direction(), rec.normal) > 0.0f);
	}



public:
	color3f albedo;
	float fuzz;
};



class dielectric : public material {
public:
	dielectric(float index_of_refraction) : ir(index_of_refraction) {}
	virtual bool scatter(
		const rayf& r_in, const hit_record& rec, color3f& attenuation, rayf& scattered
	) const override {

		attenuation = color3f(1.f, 1.f, 1.f);
		float refraction_ratio = rec.front_face ? (1.0f / ir) : ir;

		vec3f unit_direction = unit_vector(r_in.direction());
		
		float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0f);
		float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);

		bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
		vec3f direction;

		if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float()) {
			direction = reflect(unit_direction, rec.normal);
		}
		else {
			direction = refract(unit_direction, rec.normal, refraction_ratio);
		}

		scattered = rayf(rec.p, direction, r_in.time());
		return true;
	}

public:
	float ir; // index of refraction

private:
	static float reflectance(float cosine, float ref_idx) {
		auto r0 = (1 - ref_idx) / (1 + ref_idx);
		r0 = r0 * r0;
		return r0 + (1 - r0) * powf((1.0f - cosine), 5);
	}
};


#endif