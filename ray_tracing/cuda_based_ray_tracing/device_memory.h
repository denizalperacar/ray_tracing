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
	size_t m_size=0;
public:
	DeviceMemory() {}
	
	DeviceMemory(size_t s) {
		resize(s);
	}
	
	DeviceMemory<U>& operator=(DeviceMemory<U> obj) {
		std::swap(device_ptr, obj.device_ptr);
		std::swap(m_size, obj.m_size);
		return *this;
	}

	DeviceMemory(DeviceMemory<U>&& obj) {
		*this = std::move(obj);
	}

	DeviceMemory<U>& operator=(DeviceMemory<U>& obj) = delete;
	
	explicit DeviceMemory(const DeviceMemory<U>& obj) {
		copy_from_device(obj);
	}

	U* data() {
		return device_ptr;
	}

	size_t get_num_elements() const {
		return m_size;
	}

	size_t size() const {
		return get_num_elements();
	}

	size_t get_bytes() const {
		return m_size * sizeof(U);
	}

	size_t bytes() const {
		return get_bytes();
	}

	void allocate_memory(size_t n_bytes) {
		
		if (n_bytes == 0) {
			return;
		}
		uint8_t* raw_ptr;
		cudaMalloc(&raw_ptr, n_bytes);
		device_ptr = (U*)raw_ptr;
		total_n_bytes_allocated() += n_bytes; 
	}

	void free_memory() {
		if (!device_ptr) {
			return;
		}

		uint8_t* raw_ptr = (uint8_t*)device_ptr;
		cudaFree(raw_ptr);
		total_n_bytes_allocated() -= get_bytes();

		device_ptr = nullptr;
		m_size = 0;
	}

	CBRT_HOST_DEVICE ~DeviceMemory() {
#ifndef __CUDA_ARCH__
		try {
			if (device_ptr) {
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
		if (m_size != s) {
			if (m_size) {
				try {
					free_memory();
				}
				catch (std::runtime_error e) {
					std::string str(e.what());
					throw std::runtime_error{ "Could not allocate memory: " + str };
				}
			}
			m_size = s;
		}
	}

	void enlarge(const size_t s) {
		if (s > m_size) {
			resize(s);
		}
	}

	void memset(const int value, const size_t num_elements, const size_t offset = 0) {
		if (num_elements + offset > m_size) {
			throw std::runtime_error{ "Could not set memory : Number of elements is larger than allocated memory"};
		}

		cudaMemset(device_ptr + offset, value, num_elements * sizeof(U));
	}

	void memset(const int value) {
		memset(value, m_size);
	}

	void copy_from_host(const U* host_data, const size_t num_elements) {
		cudaMemcpy(data(), host_data, num_elements * sizeof(U), CBRT_HTD);
	}

	void copy_from_host(const std::vector<U>& data, const size_t num_elements) {
		if (data.m_size() < num_elements) {
			throw std::runtime_error{"Trying to copy " + std::to_string(num_elements) + "but vector m_size is " + std::to_string(data.m_size())};
		}
		copy_from_host(data.data(), num_elements);
	}

	void copy_from_host(const U* data) {
		copy_from_host(data, m_size);
	}

	void copy_from_host(const std::vector<U>& data) {
		if (data.m_size() < m_size) {
			throw std::runtime_error{ "Trying to copy " + std::to_string(m_size) + ", but vector m_size is only " + std::to_string(data.m_size()) + "."};
		}
		copy_from_host(data.data(), m_size);
	}

	void enlarge_and_copy_from_host(const U* data, const size_t num_elements) {
		enlarge(num_elements);
		copy_from_host(data, num_elements);
	}

	void enlarge_and_copy_from_host(const std::vector<U>& data, const size_t num_elements) {
		enlarge_and_copy_from_host(data.data(), num_elements);
	}

	void enlarge_and_copy_from_host(const std::vector<U>& data) {
		enlarge_and_copy_from_host(data.data(), data.m_size());
	}

	void resize_and_copy_from_host(const U* data, const size_t num_elements) {
		resize(num_elements);
		copy_from_host(data, num_elements);
	}

	void resize_and_copy_from_host(const std::vector<U>& data, const size_t num_elements) {
		resize_and_copy_from_host(data.data(), num_elements);
	}

	void resize_and_copy_from_host(const std::vector<U>& data) {
		resize_and_copy_from_host(data.data(), data.m_size());
	}

	void copy_to_host(U* host_data, const size_t num_elements) const {
		if (num_elements > m_size) {
			throw std::runtime_error{ "Trying to copy " + std::to_string(num_elements) + ", but vector m_size is only " + std::to_string(m_size) + "." };
		}
		cudaMemcpy(host_data, data(), num_elements * sizeof(U), CBRT_DTH);
	}

	void copy_to_host(std::vector<U>& data, const size_t num_elements) const {
		if (data.m_size() < num_elements) {
			throw std::runtime_error{"Trying to copy " + std::to_string(num_elements) + " elements, but vector m_size is only " + std::to_string(data.m_size())};
		}
		copy_to_host(data.data(), num_elements);
	}

	void copy_to_host(U* data) const {
		copy_to_host(data, m_size);
	}

	void copy_to_host(std::vector<U>& data) const {
		if (data.m_size() < m_size) {
			throw std::runtime_error{ "Trying to copy " + std::to_string(m_size) + " elements, but vector m_size is only " + std::to_string(data.m_size()) };
		}

		copy_to_host(data.data(), m_size);
	}

	void copy_from_device(DeviceMemory<U> obj, const size_t size) {

		if (size == 0) {
			return;
		}

		if (m_size < size) {
			resize(size);
		}

		cudaMemcpy(device_ptr, obj.device_ptr, size * sizeof(U), CBRT_DTD);
	}

	void copy_from_device(DeviceMemory<U>& obj) {
		copy_from_device(obj, obj.m_size);
	}

	DeviceMemory<U> copy(size_t size) const {
		DeviceMemory<U> result{ size };
		result.copy_from_device(*this);
		return result;
	}

	DeviceMemory<U> copy() {
		return copy(m_size);
	}

	CBRT_HOST_DEVICE T& operator[](size_t idx) const {
		if (idx > m_size) {
			printf("WARNING: buffer overrun of %p at idx %zu\n", idx);
		}

		return device_ptr[idx];
	}

	CBRT_HOST_DEVICE T& operator[](uint32_t idx) const {
		if (idx > m_size) {
			printf("WARNING: buffer overrun of %p at idx %u\n", idx);
		}

		return device_ptr[idx];
	}

};



/*
template <typename T, typename U>
CBRT_INLINE TransferedData<U> transfer_to_device(
	const T& host_data, uint32_t num_elements, bool allocate
) {
	TransferedData<U> data(num_elements);
	if (allocate) {
		cudaMalloc(&data.device_ptr, data.m_size);
	}
	cudaMemCpy(data.device_ptr, host_data, data.m_size, CBRT_HTD);
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
