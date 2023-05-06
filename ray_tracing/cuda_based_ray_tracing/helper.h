#ifndef CUDA_BASED_RAY_TRACING_COMMON_HELPER_H_
#define CUDA_BASED_RAY_TRACING_COMMON_HELPER_H_

#include "namespaces.h"

CBRT_BEGIN

#if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__))

#define CBRT_HOST_DEVICE __host__ __device__
#define CBRT_DEVICE __device__
#define CBRT_HOST __host__
#define CBRT_SHARED __shared__
#define CBRT_CONST __constant__
#define CBRT_KERNEL __global__

#else 
#define CBRT_HOST_DEVICE
#define CBRT_DEVICE 
#define CBRT_HOST 
#define CBRT_SHARED 
#define CBRT_CONST 
#define CBRT_KERNEL 

#endif

// define helpers for unrolling the loops for better performance
#if defined(__CUDA_ARCH__)
#if defined(__CUDACC_RTC__) || (defined(__clang__) && defined(__CUDA__))
#define CBRT_UNROLL _Pragma("unroll")
#define CBRT_NO_UNROLL _Pragma("unroll 1")
#else
#define CBRT_UNROLL #pragma unroll
#define CBRT_NO_UNROLL #pragma unroll 1
#endif
#else
#define CBRT_UNROLL 
#define CBRT_NO_UNROLL 
#endif

#define CBRT_INLINE __inline__
#define CBRT_VECTOR


// aliases
#define CBRT_HTD cudaMemcpyHostToDevice
#define CBRT_DTH cudaMemcpyDeviceToHost


CBRT_END

#endif
