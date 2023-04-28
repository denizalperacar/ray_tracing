#ifndef RAY_TRACING_COMMON_LOAD_IMAGE_H_
#define RAY_TRACING_COMMON_LOAD_IMAGE_H_

#ifdef _MSC_VER
    // Microsoft Visual C++ Compiler
#pragma warning (push, 0)
#endif

#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// Restore warning levels.
#ifdef _MSC_VER
    // Microsoft Visual C++ Compiler
#pragma warning (pop)
#endif


#endif
