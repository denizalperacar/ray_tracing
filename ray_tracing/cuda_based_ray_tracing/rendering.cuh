#ifndef CUDA_BASED_RAY_TRACING_RENDERING_RENDERING_CUH_
#define CUDA_BASED_RAY_TRACING_RENDERING_RENDERING_CUH_

#include "common.h"
#include "device_memory.h"
#include "get_color.h"
#include "toojpeg.h"

CBRT_BEGIN

// solve this 
CBRT_KERNEL void render(render_color* result) {
	uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t j = threadIdx.y + blockDim.y * blockIdx.y;
	uint32_t idx = j * gridDim.x * blockDim.x + i;

	render_color c;

	if (i < DEFAULT_IMAGE_WIDTH && j < DEFAULT_IMAGE_HEIGHT) {
		color3f pixel_color((float)i / (DEFAULT_IMAGE_WIDTH - 1), (float)j / (DEFAULT_IMAGE_HEIGHT - 1), 0.25);
		result[idx] = get_color(pixel_color, 1);
		//c.print();
	}
}

void render() {
	size_t size = DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT * sizeof(render_color);
	CBRT::DeviceMemory<render_color> device_ptr(DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT);
	device_ptr.allocate_memory(size);
	dim3 grid(DEFAULT_IMAGE_WIDTH / NUM_THREADS_MIN, DEFAULT_IMAGE_HEIGHT / NUM_THREADS_MIN);
	dim3 block(NUM_THREADS_MIN, NUM_THREADS_MIN);
	render << < grid, block >> > (device_ptr.data());
	std::vector<render_color> host_ptr(DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT);
	device_ptr.copy_to_host(host_ptr);

	auto ok = TooJpeg::writeJpeg(image_output, host_ptr.data(), DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, true, 90, false);
}

CBRT_END

#endif