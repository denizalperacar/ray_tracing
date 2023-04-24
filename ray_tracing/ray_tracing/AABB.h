#ifndef RAY_TRACING_SHAPES_AABB_H_
#define RAY_TRACING_SHAPES_AABB_H_

#include "common.h"

class aabb {
public:
	aabb() = default;
	aabb(const point3f& a, const point3f& b) { 
		minimum = a;
		maximum = b;
	}

	point3f max() { return maximum; }
	point3f min() { return minimum; }

	bool hit(const rayf& r, float t_min, float t_max) const {
		/* general method
		for (int i = 0; i < 3; i++) {
			float t0 = fminf(
				(minimum[i] - r.origin()[i]) / r.direction()[i],
				(maximum[i] - r.origin()[i]) / r.direction()[i]
			);
			float t1 = fmaxf(
				(minimum[i] - r.origin()[i]) / r.direction()[i],
				(maximum[i] - r.origin()[i]) / r.direction()[i]
			);
			t_min = fmaxf(t0, t_min);
			t_max = fmaxf(t1, t_max);

			if (t_min >= t_max) {
				return false;
			}
		}
		return true;
		*/

		// optimized method
		for (int i = 0; i < 3; i++) {
			float invD = 1.0f / r.direction()[i];
			float t0 = (minimum[i] - r.origin()[i]) * invD;
			float t1 = (maximum[i] - r.origin()[i]) * invD;
			if (invD < 0.0f) {
				std::swap(t0, t1);
			}
			t_min = t0 > t_min ? t0 : t_min;
			t_max = t1 < t_max ? t1 : t_max;
			if (t_max <= t_min) {
				return false;
			}
		}

		return true;
	}

	point3f minimum;
	point3f maximum;

};


aabb surrounding_box(aabb box0, aabb box1) {
	point3f small(fmin(box0.min().x(), box1.min().x()),
		fmin(box0.min().y(), box1.min().y()),
		fmin(box0.min().z(), box1.min().z()));

	point3f big(fmax(box0.max().x(), box1.max().x()),
		fmax(box0.max().y(), box1.max().y()),
		fmax(box0.max().z(), box1.max().z()));

	return aabb(small, big);
}

#endif
