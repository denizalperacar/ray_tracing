#include "common.h"

class color {
public:

	CBRT_HOST_DEVICE color() : r(0), g(0), b(0) {}

	CBRT_HOST_DEVICE color(uint8_t red, uint8_t green, uint8_t blue) 
		: r(red), g(green), b(blue) {}

	CBRT_HOST std::ofstream& draw(std::ofstream& os) {
		os << r << " " << g << " " << b << "\n";
		return os;
	}

	CBRT_HOST_DEVICE void print() {
		printf("%d %d %d\n", r, g, b);
	}

public:
	uint8_t r;
	uint8_t g;
	uint8_t b;

};

std::ostream& operator<<(std::ostream& os, color obj) {
	os << obj.r << " " << obj.g << " " << obj.b << "\n";
	return os;
}

// solve this 
CBRT_KERNEL void render(color* result) {
	uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t j = threadIdx.y + blockDim.y * blockIdx.y;
	uint32_t idx = j * gridDim.x * blockDim.x + i;

	color c;

	if (i < DEFAULT_IMAGE_WIDTH && j < DEFAULT_IMAGE_HEIGHT) {
		float r = (float)i / (DEFAULT_IMAGE_WIDTH - 1);
		float g = (float)j / (DEFAULT_IMAGE_HEIGHT - 1);
		float b = 0.25;

		c.r = (uint8_t)(255.999 * r);
		c.g = (uint8_t)(255.999 * g);
		c.b = (uint8_t)(255.999 * b);

		result[idx] = c;
	}
}

int main() {



	color* device_ptr;
	color host_data [DEFAULT_NUMBER_OF_PIXELS];


	size_t size = DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT * sizeof(color);
	cudaMalloc(&device_ptr, size);

	dim3 grid(DEFAULT_IMAGE_WIDTH / NUM_THREADS_MIN, DEFAULT_IMAGE_HEIGHT / NUM_THREADS_MIN);
	dim3 block(NUM_THREADS_MIN, NUM_THREADS_MIN);
	render << < grid, block  >> > (device_ptr);

	cudaMemcpy(host_data, device_ptr, size, CBRT_DTH);

	for (size_t i = 0; i < DEFAULT_NUMBER_OF_PIXELS; i++) {
		std::cout << host_data[i] << "\n";
	}

	cudaFree(device_ptr);
	return 0;
}

