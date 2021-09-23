#ifndef CUDA_IMAGE_UTIL_H
#define CUDA_IMAGE_UTIL_H

#include <cuda_runtime.h>
#include "cuda_SimpleMatrixUtil.h"
#include "mLibCuda.h"

#define T_PER_BLOCK 16
#define MINF __int_as_float(0xff800000)

namespace CUDAImageUtil
{
	template<class T> void copy(T* d_output, T* d_input, unsigned int width, unsigned int height);
	void resampleToIntensity(float* d_output, unsigned int outputWidth, unsigned int outputHeight, const uchar4* d_input, unsigned int inputWidth, unsigned int inputHeight);

	void resampleFloat4(float4* d_output, unsigned int outputWidth, unsigned int outputHeight, const float4* d_input, unsigned int inputWidth, unsigned int inputHeight);
	void resampleFloat(float* d_output, unsigned int outputWidth, unsigned int outputHeight, const float* d_input, unsigned int inputWidth, unsigned int inputHeight);
	void resampleUCHAR4(uchar4* d_output, unsigned int outputWidth, unsigned int outputHeight, const uchar4* d_input, unsigned int inputWidth, unsigned int inputHeight);

	void convertDepthFloatToCameraSpaceFloat4(float4* d_output, const float* d_input, const float4x4& intrinsicsInv, unsigned int width, unsigned int height);
	void computeNormals(float4* d_output, const float4* d_input, unsigned int width, unsigned int height);

	void jointBilateralFilterColorUCHAR4(uchar4* d_output, uchar4* d_input, float* d_depth, float sigmaD, float sigmaR, unsigned int width, unsigned int height);

	void erodeDepthMap(float* d_output, float* d_input, int structureSize, unsigned int width, unsigned int height, float dThresh, float fracReq);

	void gaussFilterDepthMap(float* d_output, const float* d_input, int radius, float sigmaD, float sigmaR, unsigned int width, unsigned int height);
	void gaussFilterIntensity(float* d_output, const float* d_input, float sigmaD, unsigned int width, unsigned int height);

	void convertUCHAR4ToIntensityFloat(float* d_output, const uchar4* d_input, unsigned int width, unsigned int height);

	void computeIntensityDerivatives(float2* d_output, const float* d_input, unsigned int width, unsigned int height);
	void computeIntensityGradientMagnitude(float* d_output, const float* d_input, unsigned int width, unsigned int height);

	void convertNormalsFloat4ToUCHAR4(uchar4* d_output, const float4* d_input, unsigned int width, unsigned int height);
	void computeNormalsSobel(float4* d_output, const float4* d_input, unsigned int width, unsigned int height);
	void adaptiveGaussFilterDepthMap(float* d_output, const float* d_input, float sigmaD, float sigmaR, float adaptFactor, unsigned int width, unsigned int height);
	void adaptiveGaussFilterIntensity(float* d_output, const float* d_input, const float* d_depth, float sigmaD, float adaptFactor, unsigned int width, unsigned int height);

	void jointBilateralFilterFloat(float* d_output, float* d_input, float* d_depth, float sigmaD, float sigmaR, unsigned int width, unsigned int height);
	void adaptiveBilateralFilterIntensity(float* d_output, const float* d_input, const float* d_depth, float sigmaD, float sigmaR, float adaptFactor, unsigned int width, unsigned int height);

	void countNumValidDepth(int *d_cnt, const float *d_input, const int height, const int width);

};

#endif