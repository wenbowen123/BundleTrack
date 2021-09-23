#pragma once

#ifndef _CUDA_UTIL_
#define _CUDA_UTIL_

#undef max
#undef min

#include "cutil_inline.h"
#include "cutil_math.h"

#define cudaAssert(condition) if (!(condition)) { printf("ASSERT: %s %s\n", #condition, __FILE__); }

#if defined(__CUDA_ARCH__)
#define __CONDITIONAL_UNROLL__ #pragma unroll
#else
#define __CONDITIONAL_UNROLL__
#endif


#ifdef __CUDACC__

#endif

#endif
