#ifndef RAY_TRACING_TEXTURE_PERLIN_NOISE_H_
#define RAY_TRACING_TEXTURE_PERLIN_NOISE_H_

#include "common.h"

class perlin {
public:
	perlin() {
		ranvec = new vec3f[point_count];
		for (int32_t i = 0; i < point_count; i++) {
			ranvec[i] = unit_vector(vec3f::random(-1, 1));
		}

		perm_x = perlin_generate_perm();
		perm_y = perlin_generate_perm();
		perm_z = perlin_generate_perm();

	}

	~perlin() {
		delete[] ranvec;
		delete[] perm_x;
		delete[] perm_y;
		delete[] perm_z;
	}

	float noise(const point3f& p) const {
		
		float u = p.x() - floorf(p.x());
		float v = p.y() - floorf(p.y());
		float w = p.z() - floorf(p.z());

		int i = static_cast<int> (floorf(p.x()));
		int j = static_cast<int> (floorf(p.y()));
		int k = static_cast<int> (floorf(p.z()));

		vec3f c[2][2][2];

		for (int di = 0; di < 2; di++) {
			for (int dj = 0; dj < 2; dj++) {
				for (int dk = 0; dk < 2; dk++) {
					c[di][dj][dk] = ranvec[
						perm_x[(i + di) & 255] ^
						perm_y[(j + dj) & 255] ^
						perm_z[(k + dk) & 255]
					];
				}
			}
		}

		// return ranfloat[perm_x[i] ^ perm_y[j] ^ perm_z[k]]; // cool version just give them i 
		return trilinear_interpolation(c, u, v, w);
	}

	float turbulance(const point3f& p, int depth = 7) const {
		float accum = 0.0f;
		point3f temp_p = p;
		float weight = 1.0f;

		for (int i = 0; i < depth; i++) {
			accum += weight * noise(temp_p);
			weight *= 0.5;
			temp_p = 2.0f * temp_p;
		}

		return fabsf(accum);
	}


private:
	static const uint32_t point_count{ 256 };
	vec3f* ranvec;
	int32_t* perm_x;
	int32_t* perm_y;
	int32_t* perm_z;

	static int* perlin_generate_perm() {
		auto p = new int[point_count];

		for (int i = 0; i < perlin::point_count; i++ ) {
			p[i] = i;
		}

		permute(p, point_count);
		return p;
	}

	static void permute(int* p, int n) {
		for (int i = n - 1; i > 0; i--) {
			int target = random_int(0, i);
			int tmp = p[i];
			p[i] = p[target];
			p[target] = tmp;
		}
	}

	static float trilinear_interpolation(vec3f c[2][2][2], float u, float v, float w) {
		float accum = 0.f;

		float uu = u * u * (3.0f - 2.0f * u);
		float vv = v * v * (3.0f - 2.0f * v);
		float ww = w * w * (3.0f - 2.0f * w);

		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				for (int k = 0; k < 2; k++) {
					vec3f weight(u - (float)i, v - (float)j, w - (float)k);
					accum += 
						(i * uu + (1 - i) * (1 - uu))
						* (j * vv + (1 - j) * (1 - vv))
						* (k * ww + (1 - k) * (1 - ww)) 
						* dot(c[i][j][k], weight);
				}
			}
		}
		return accum;
	}

};


#endif