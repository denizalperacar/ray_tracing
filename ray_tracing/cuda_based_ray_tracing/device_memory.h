#ifndef CUDA_BASED_RAY_TRACING_MEMORY_DEVICE_MEMORY_H_
#define CUDA_BASED_RAY_TRACING_MEMORY_DEVICE_MEMORY_H_

#include "common.h"

CBRT_BEGIN

// tracks the bytes allcoated by all the DeviceMemory allocations
CBRT_INLINE std::atomic<size_t>& total_n_bytes_allocated() {
	static std::atomic<size_t> s_total_n_bytes_allocated{ 0 };
	return s_total_n_bytes_allocated;
}


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

	DeviceMemory<U>& operator=(DeviceMemory<U>& obj) = delete;
	
	explicit DeviceMemory(const DeviceMemory<U>& obj) {
		copy_from_device(obj);
	}

	void allocate_memory(size_t n_bytes) {
		
		if (n_bytes == 0) {
			return;
		}
		uint8_t* raw_ptr;
		cudaMalloc(&raw_ptr, n_bytes);
		m_data = (T*)raw_ptr;
		total_n_bytes_allocated() += n_bytes; 
	}

	void free_memory() {
		if (!m_data) {
			return;
		}

		uint8_t* raw_ptr = (uint8_t*)m_data;
		cudaFree(raw_ptr);
		total_n_bytes_allocated() -= get_bytes();

		m_data = nullptr;
		m_size = 0;
	}

	CBRT_HOST_DEVICE ~DeviceMemory() {
#ifndef __CUDA_ARCH__
		try {
			if (m_data) {
				free_memory();
				m_size = 0;
			}
		}
		catch (std::runtime_error e) {
			if (std::string{ e.what() }.find("driver shutting down") == std::string::npos) {
				std::cerr << "Could not free memory: " << e.what() << std::endl;
			}
		}
#endif
	}

	void resize(size_t s) {
		if (m_size != size) {
			if (m_size) {
				try {
					free_memory();
				}
				catch (std::runtime_error e) {
					throw std::runtime_error{ "Could not allocate memory: " + e.what()) };
				}
			}
			m_size = size;
		}
	}

	void enlarge(const size_t size) {
		if (size > m_size) {
			resize(size);
		}
	}

	void memset(const int value, const size_t num_elements, const size_t offset = 0) {
		if (num_elements + offset > m_size) {
			throw std::runtime_error{ "Could not set memory : Number of elements is larger than allocated memory"};
		}

		cudaMemset(m_data + offset. value, num_elements * sizeof(T));
	}

	void memset(const int value) {
		memset(value, m_size);
	}

	void copy_from_host(const T* host_data, const size_t num_elements) {
		cudaMemcpy(data(), host_data, num_elements * sizeof(T), CBRT_HTD);
	}

	void copy_from_host(const std::vector<T>& data, const size_t num_elements) {
		if (data.size() < num_elements) {
			throw std::runtime_error{"Trying to copy " + std::to_string(num_elements) + "but vector size is " + std::to_string(data.size())};
		}
		copy_from_host(data.data(), num_elements);
	}

	void copy_from_host(const T* data) {
		copy_from_host(data, m_size);
	}

	void enlarge_and_copy_from_host(const T* data, const size_t num_elements) {
		enlarge(num_elements);
		copy_from_host(data, num_elements);
	}

	void enlarge_and_copy_from_host(const std::vector<T>& data, const size_t num_elements) {
		enlarge_and_copy_from_host(data.data(), num_elements);
	}

	void enlarge_and_copy_from_host(const std::vector<T>& data) {
		enlarge_and_copy_from_host(data.data(), data.size());
	}



	void copy_from_device(DeviceMemory<U>& obj) {

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
