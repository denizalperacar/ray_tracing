#ifndef CUDA_BASED_RAY_TRACING_MEMORY_DEVICE_MEMORY_H_
#define CUDA_BASED_RAY_TRACING_MEMORY_DEVICE_MEMORY_H_

#include "common.h"

CBRT_BEGIN

template <typename U>
class DeviceMemory {
private:
	U* device_ptr = nullptr;
	size_t size=0;
public:
	DeviceMemory() {}
	
	DeviceMemory(size_t size) {
		resize(size);
	}
	
	DeviceMemory<U>& operator=(DeviceMemory<U> obj) {
		std::swap(device_ptr, obj.device_ptr);
		std::swap(size, obj.size);
		return *this;
	}

	DeviceMemory(DeviceMemory<U>&& obj) {
		*this = std::move(obj);
	}




private:
	void resize(size_t s) {
		size = s;
	}
};



/*
template <typename T, typename U>
CBRT_INLINE TransferedData<U> transfer_to_device(
	const T& host_data, uint32_t num_elements, bool allocate
) {
	TransferedData<U> data(num_elements);
	if (allocate) {
		cudaMalloc(&data.device_ptr, data.size);
	}
	cudaMemCpy(data.device_ptr, host_data, data.size, CBRT_HTD);
	return data;
}

template <typename T, typename U>
CBRT_INLINE void transfer_to_host(
	T& host_data, TransferedData<U>& device_data, uint32_t num_elements, bool remove
) {

	if (remove) {
		cudaFree(device_data.device_ptr);
	}
}
*/

CBRT_END

#endif
