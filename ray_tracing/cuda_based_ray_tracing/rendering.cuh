#ifndef CUDA_BASED_RAY_TRACING_RENDERING_RENDERING_CUH_
#define CUDA_BASED_RAY_TRACING_RENDERING_RENDERING_CUH_

#include "common.h"
#include "device_memory.h"
#include "get_color.h"
#include "ray_color.h"
#include "toojpeg.h"

CBRT_BEGIN

// solve this 
CBRT_KERNEL void render(render_color* result) {
	uint32_t i = threadIdx.x + blockDim.x * blockIdx.x;
	uint32_t j = threadIdx.y + blockDim.y * blockIdx.y;
	uint32_t idx = j * gridDim.x * blockDim.x + i;

	if (i < DEFAULT_IMAGE_WIDTH && j < DEFAULT_IMAGE_HEIGHT) {
		// Camera

		auto viewport_height = 2.0f;
		auto viewport_width = DEFAULT_IMAGE_ASPECT_RATIO * viewport_height;
		auto focal_length = 1.0f;

		auto origin = point3f(0.f, 0.f, 0.f);
		auto horizontal = vec3f(viewport_width, 0.f, 0.f);
		auto vertical = vec3f(0, viewport_height, 0.f);
		auto lower_left_corner = origin - horizontal / 2.f - vertical / 2.f - vec3f(0.f, 0.f, focal_length);


		render_color c;
		auto u = float(i) / (DEFAULT_IMAGE_WIDTH - 1);
		auto v = float(j) / (DEFAULT_IMAGE_HEIGHT - 1);
		ray r(origin, lower_left_corner + u * horizontal + v * vertical - origin);
		color3f pixel_color = ray_color(r);
		result[idx] = get_color(pixel_color, 1);
		//c.print();
	}
}

void render() {

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	size_t size = DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT * sizeof(render_color);
	CBRT::DeviceMemory<render_color> device_ptr(DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT);
	device_ptr.allocate_memory(size);
	dim3 grid(DEFAULT_IMAGE_WIDTH / NUM_THREADS_MIN, DEFAULT_IMAGE_HEIGHT / NUM_THREADS_MIN);
	dim3 block(NUM_THREADS_MIN, NUM_THREADS_MIN);
	render << < grid, block >> > (device_ptr.data());
	std::vector<render_color> host_ptr(DEFAULT_IMAGE_WIDTH * DEFAULT_IMAGE_HEIGHT);
	device_ptr.copy_to_host(host_ptr);

	auto ok = TooJpeg::writeJpeg(image_output, host_ptr.data(), DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT, true, 90, false);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

	if (ok) {
		std::cout << "Image saved successfully" << std::endl;
	}
	else {
		std::cout << "Error in saving image" << std::endl;
	}

	std::cout << "Time taken to render the image: " 
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() 
		<< " ms" << std::endl;
}

CBRT_END

#endif