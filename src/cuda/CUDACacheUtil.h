#pragma once
#ifndef CUDA_CACHE_UTIL
#define CUDA_CACHE_UTIL

#include "mLibCuda.h"

#define CUDACACHE_UCHAR_NORMALS
#define CUDACACHE_FLOAT_NORMALS

struct CUDACachedFrame {
	void alloc(unsigned int width, unsigned int height) {
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_depthDownsampled, sizeof(float) * width * height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_cameraposDownsampled, sizeof(float4) * width * height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_num_valid_points, sizeof(int)));
		MLIB_CUDA_SAFE_CALL(cudaMemset(d_num_valid_points, 0, sizeof(int)));

		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_intensityDownsampled, sizeof(float) * width * height));
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_intensityDerivsDownsampled, sizeof(float2) * width * height));
#ifdef CUDACACHE_UCHAR_NORMALS
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_normalsDownsampledUCHAR4, sizeof(uchar4) * width * height));
#endif
#ifdef CUDACACHE_FLOAT_NORMALS
		MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_normalsDownsampled, sizeof(float4) * width * height));
#endif
	}
	void free() {
		cutilSafeCall(cudaFree(d_depthDownsampled));
		cutilSafeCall(cudaFree(d_num_valid_points));
		cutilSafeCall(cudaFree(d_cameraposDownsampled));

		cutilSafeCall(cudaFree(d_intensityDownsampled));
		cutilSafeCall(cudaFree(d_intensityDerivsDownsampled));
#ifdef CUDACACHE_UCHAR_NORMALS
		cutilSafeCall(cudaFree(d_normalsDownsampledUCHAR4));
#endif
#ifdef CUDACACHE_FLOAT_NORMALS
		cutilSafeCall(cudaFree(d_normalsDownsampled));
#endif
	}

	int* d_num_valid_points;
	float* d_depthDownsampled;
	float4* d_cameraposDownsampled;

	float* d_intensityDownsampled;
	float2* d_intensityDerivsDownsampled;
#ifdef CUDACACHE_UCHAR_NORMALS
	uchar4* d_normalsDownsampledUCHAR4;
#endif
#ifdef CUDACACHE_FLOAT_NORMALS
	float4* d_normalsDownsampled;
#endif
};

#endif //CUDA_CACHE_UTIL