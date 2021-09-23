#pragma once

#ifndef _SOLVER_UTIL_
#define _SOLVER_UTIL_

#include "cutil_inline.h"
#include <cutil_math.h>
#include "cuda_SimpleMatrixUtil.h"

#define FLOAT_EPSILON 0.000001f

#ifndef BYTE
using BYTE = unsigned char;
#endif

#define MINF __int_as_float(0xff800000)

__inline__ __device__ void get2DIdx(int idx, unsigned int width, unsigned int height, int& i, int& j)
{
	i = idx / width;
	j = idx % width;
}

__inline__ __device__ unsigned int get1DIdx(int i, int j, unsigned int width, unsigned int height)
{
	return i*width+j;
}

__inline__ __device__ bool isInsideImage(int i, int j, unsigned int width, unsigned int height)
{
	return (i >= 0 && i < height && j >= 0 && j < width);
}

void printPosesMatricesCU(const float4x4* d_transforms, const int numTransforms);


#endif
