#include "hittable.h"


rotate_y::rotate_y(shared_ptr<hittable> p, float angle) : ptr(p) {

	float radians = degrees_to_radians(angle);
	sin_theta = sinf(radians);
	cos_theta = cosf(radians);
	has_box = ptr->bounding_box(0.f, 1.f, bbox);

	point3f min(infinityf, infinityf, infinityf);
	point3f max(-infinityf, -infinityf, -infinityf);

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			for (int k = 0; k < 2; k++) {
				// test all the permutations of all the points in the ractangular box
				float x = i * bbox.max().x() + (1 - i) * bbox.min().x();
				float y = j * bbox.max().y() + (1 - j) * bbox.min().y();
				float z = k * bbox.max().z() + (1 - k) * bbox.min().z();

				float newx = cos_theta * x + sin_theta * z;
				float newz = -sin_theta * x + cos_theta * z;

				vec3f tester(newx, y, newz);

				// find the min and max bound of the smallest bounding box that covers 
				// the rotated rectangular box
				for (int c = 0; c < 3; c++) {
					min[c] = fminf(min[c], tester[c]);
					max[c] = fmaxf(max[c], tester[c]);
				}
			}
		}
	}
	bbox = aabb(min, max);

}

// [TODO] implement the rotation matrix calculation for these
bool rotate_y::hit(const rayf& r, float time0, float time1, hit_record& rec) const {

	point3f origin = r.origin();
	vec3f direction = r.direction();

	// update the x and z poitions of the ray poition as its rotated about global axis
	origin[0] = cos_theta * r.origin()[0] - sin_theta * r.origin()[2];
	origin[2] = sin_theta * r.origin()[0] + cos_theta * r.origin()[2];

	direction[0] = cos_theta * r.direction()[0] - sin_theta * r.direction()[2];
	direction[2] = sin_theta * r.direction()[0] + cos_theta * r.direction()[2];

	rayf rotated_ray(origin, direction, r.time());
	
	if (!ptr->hit(rotated_ray, time0, time1, rec)) {
		return false;
	}

	point3f p = rec.p;
	vec3f normal = rec.normal;

	p[0] = cos_theta * rec.p[0] + sin_theta * rec.p[2];
	p[2] = -sin_theta * rec.p[0] + cos_theta * rec.p[2];

	normal[0] = cos_theta * rec.normal[0] + sin_theta * rec.normal[2];
	normal[2] = -sin_theta * rec.normal[0] + cos_theta * rec.normal[2];

	rec.p = p;
	rec.set_face_normal(rotated_ray, normal);

	return true;
}