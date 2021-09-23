#include <iostream>
#include <fstream>
#include "GlobalDefines.h"
#include "SolverBundlingParameters.h"
#include "SolverBundlingState.h"
#include "SolverBundlingUtil.h"
#include "SolverBundlingEquationsLie.h"
#include "SolverBundlingDenseUtil.h"
#include "../CUDATimer.h"


#define THREADS_PER_BLOCK_DENSE_DEPTH 128
#define THREADS_PER_BLOCK_DENSE_DEPTH_FLIP 64
#define THREADS_PER_BLOCK_DENSE_OVERLAP 512


template<bool usePairwise>
__global__ void FindImageImageCorr_Kernel(SolverInput input, SolverState state, SolverParameters parameters)
{
	unsigned int i, j;
	i = blockIdx.x;
	j = blockIdx.y;
	if (i == j)
		return;
	if (input.d_cacheFrames[i].d_num_valid_points == input.d_cacheFrames[j].d_num_valid_points)
	{
		if (i >= j)
			return;
	}
	else if (input.d_cacheFrames[i].d_num_valid_points < input.d_cacheFrames[j].d_num_valid_points)
	{
		return;
	}

	if (input.d_validImages[i] == 0 || input.d_validImages[j] == 0)
	{
		return;
	}

	const unsigned int tidx = threadIdx.x;

	if (tidx == 0)
	{
		int addr = atomicAdd(state.d_numDenseOverlappingImages, 1);
		state.d_denseOverlappingImages[addr] = make_uint2(i, j);
	}
}

__global__ void FlipJtJ_Kernel(unsigned int total, unsigned int dim, float* d_JtJ)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < total) {
		const unsigned int x = idx % dim;
		const unsigned int y = idx / dim;
		if (x > y) {
			d_JtJ[y * dim + x] = d_JtJ[x * dim + y];
		}
	}
}
__global__ void FindDenseCorrespondences_Kernel(SolverInput input, SolverState state, SolverParameters parameters, int *d_corres_coord)
{
	const int imPairIdx = blockIdx.x;
	uint2 imageIndices = state.d_denseOverlappingImages[imPairIdx];
	unsigned int i = imageIndices.x;
	unsigned int j = imageIndices.y;

	const unsigned int tidx = threadIdx.x;
	const unsigned int gidx = tidx * gridDim.y + blockIdx.y;
	const int n_pixels = input.denseDepthWidth * input.denseDepthHeight;

	if (gidx < n_pixels)
	{
#ifdef USE_LIE_SPACE
		float4x4 transform = state.d_xTransformInverses[i] * state.d_xTransforms[j];
#else
		float4x4 transform_i = evalRtMat(state.d_xRot[i], state.d_xTrans[i]);
		float4x4 transform_j = evalRtMat(state.d_xRot[j], state.d_xTrans[j]);
		float4x4 invTransform_i = transform_i.getInverse();
		float4x4 transform = invTransform_i * transform_j;
#endif
		const int numWarps = THREADS_PER_BLOCK_DENSE_DEPTH / WARP_SIZE;
		__shared__ int s_count[numWarps];
		s_count[0] = 0;
		int count = 0;

		if (findDenseCorr(gidx, input.denseDepthWidth, input.denseDepthHeight,
											parameters.denseDistThresh, parameters.denseNormalThresh, transform, input.intrinsics,
											input.d_cacheFrames[i].d_depthDownsampled, input.d_cacheFrames[i].d_normalsDownsampled,
											input.d_cacheFrames[j].d_depthDownsampled, input.d_cacheFrames[j].d_normalsDownsampled,
											parameters.denseDepthMin, parameters.denseDepthMax))
		{
			d_corres_coord[imPairIdx*n_pixels+gidx] = 1;
			count++;
		}
		count = warpReduce(count);
		__syncthreads();
		if (tidx % WARP_SIZE == 0)
		{
			s_count[tidx / WARP_SIZE] = count;
		}
		__syncthreads();
		for (unsigned int stride = numWarps / 2; stride > 0; stride /= 2)
		{
			if (tidx < stride)
				s_count[tidx] = s_count[tidx] + s_count[tidx + stride];
			__syncthreads();
		}
		if (tidx == 0)
		{
			atomicAdd(&state.d_denseCorrCounts[imPairIdx], s_count[0]);
		}
	}
}

__global__ void WeightDenseCorrespondences_Kernel(unsigned int N, SolverState state)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N) {
		float x = state.d_denseCorrCounts[idx];
		if (x > 0) {
			if (x < 800) state.d_denseCorrCounts[idx] = 0;
			else {
				state.d_denseCorrCounts[idx] = 1.0f / min(logf(x), 9.0f);
			}
		}
	}
}

template<bool useDepth, bool useColor>
__global__ void BuildDenseSystem_Kernel(SolverInput input, SolverState state, SolverParameters parameters)
{
	const int imPairIdx = blockIdx.x;
	uint2 imageIndices = state.d_denseOverlappingImages[imPairIdx];
	unsigned int i = imageIndices.x;	unsigned int j = imageIndices.y;

	float imPairWeight = 1;
	if (imPairWeight == 0.0f) return;

	const unsigned int idx = threadIdx.x;
	const unsigned int srcIdx = idx * gridDim.y + blockIdx.y;

	if (srcIdx < (input.denseDepthWidth * input.denseDepthHeight)) {
#ifdef USE_LIE_SPACE
		float4x4 transform_i = state.d_xTransforms[i];
		float4x4 transform_j = state.d_xTransforms[j];
		float4x4 invTransform_i = state.d_xTransformInverses[i];
		float4x4 invTransform_j = state.d_xTransformInverses[j];
		float4x4 transform = invTransform_i * transform_j;

#else
		float4x4 transform_i = evalRtMat(state.d_xRot[i], state.d_xTrans[i]);
		float4x4 transform_j = evalRtMat(state.d_xRot[j], state.d_xTrans[j]);
		float4x4 invTransform_i = transform_i.getInverse();
		float4x4 transform = invTransform_i * transform_j;
#endif
		matNxM<1, 6> depthJacBlockRow_i, depthJacBlockRow_j; depthJacBlockRow_i.setZero(); depthJacBlockRow_j.setZero();
		float depthRes = 0.0f; float depthWeight = 0.0f;
		matNxM<1, 6> colorJacBlockRow_i, colorJacBlockRow_j; colorJacBlockRow_i.setZero(); colorJacBlockRow_j.setZero();
		float colorRes = 0.0f; float colorWeight = 0.0f;

		float3 camPosSrc; float3 camPosSrcToTgt; float3 camPosTgt; float3 normalTgt; float2 tgtScreenPos;
#ifdef CUDACACHE_FLOAT_NORMALS
		bool foundCorr = findDenseCorr(srcIdx, input.denseDepthWidth, input.denseDepthHeight,
			parameters.denseDistThresh, parameters.denseNormalThresh, transform, input.intrinsics,
			input.d_cacheFrames[i].d_cameraposDownsampled, input.d_cacheFrames[i].d_normalsDownsampled,
			input.d_cacheFrames[j].d_cameraposDownsampled, input.d_cacheFrames[j].d_normalsDownsampled,
			parameters.denseDepthMin, parameters.denseDepthMax, camPosSrc, camPosSrcToTgt, tgtScreenPos, camPosTgt, normalTgt);
#elif defined(CUDACACHE_UCHAR_NORMALS)
		bool foundCorr = findDenseCorr(srcIdx, input.denseDepthWidth, input.denseDepthHeight,
			parameters.denseDistThresh, parameters.denseNormalThresh, transform, input.intrinsics,
			input.d_cacheFrames[i].d_cameraposDownsampled, input.d_cacheFrames[i].d_normalsDownsampledUCHAR4,
			input.d_cacheFrames[j].d_cameraposDownsampled, input.d_cacheFrames[j].d_normalsDownsampledUCHAR4,
			parameters.denseDepthMin, parameters.denseDepthMax, camPosSrc, camPosSrcToTgt, tgtScreenPos, camPosTgt, normalTgt);
#endif

		if (useDepth)
		{
			if (foundCorr)
			{
				float3 diff = camPosTgt - camPosSrcToTgt;
				depthRes = dot(diff, normalTgt);
				float3 rho;
				huberLoss(depthRes*depthRes, parameters.robust_delta, rho);
				depthWeight = parameters.weightDenseDepth*rho.y;
#ifdef USE_LIE_SPACE
				if (i > 0)
					computeJacobianBlockRow_i(depthJacBlockRow_i, transform_i, invTransform_j, camPosSrc, normalTgt);
				if (j > 0)
					computeJacobianBlockRow_j(depthJacBlockRow_j, invTransform_i, transform_j, camPosSrc, normalTgt);
#else
				if (i > 0)
					computeJacobianBlockRow_i(depthJacBlockRow_i, state.d_xRot[i], state.d_xTrans[i], transform_j, camPosSrc, normalTgt);
				if (j > 0)
					computeJacobianBlockRow_j(depthJacBlockRow_j, state.d_xRot[j], state.d_xTrans[j], invTransform_i, camPosSrc, normalTgt);
#endif
			}
			addToLocalSystem(foundCorr, state.d_denseJtJ, state.d_denseJtr, input.numberOfImages * 6, depthJacBlockRow_i, depthJacBlockRow_j, i, j, depthRes, depthWeight, idx, state.d_sumResidual, state.d_corrCount);
		}
		if (useColor)
		{
			bool foundCorrColor = false;
			if (foundCorr)
			{
				const float2 intensityDerivTgt = bilinearInterpolationFloat2(tgtScreenPos.x, tgtScreenPos.y, input.d_cacheFrames[i].d_intensityDerivsDownsampled, input.denseDepthWidth, input.denseDepthHeight);
				const float intensityTgt = bilinearInterpolationFloat(tgtScreenPos.x, tgtScreenPos.y, input.d_cacheFrames[i].d_intensityDownsampled, input.denseDepthWidth, input.denseDepthHeight);
				colorRes = intensityTgt - input.d_cacheFrames[j].d_intensityDownsampled[srcIdx];
				foundCorrColor = (intensityDerivTgt.x != MINF && abs(colorRes) < parameters.denseColorThresh && length(intensityDerivTgt) > parameters.denseColorGradientMin);
				if (foundCorrColor)
				{
					const float2 focalLength = make_float2(input.intrinsics.x, input.intrinsics.y);
#ifdef USE_LIE_SPACE
					if (i > 0)
						computeJacobianBlockIntensityRow_i(colorJacBlockRow_i, focalLength, transform_i, invTransform_j, camPosSrc, camPosSrcToTgt, intensityDerivTgt);
					if (j > 0)
						computeJacobianBlockIntensityRow_j(colorJacBlockRow_j, focalLength, invTransform_i, transform_j, camPosSrc, camPosSrcToTgt, intensityDerivTgt);
#else
					if (i > 0)
						computeJacobianBlockIntensityRow_i(colorJacBlockRow_i, focalLength, state.d_xRot[i], state.d_xTrans[i], transform_j, camPosSrc, camPosSrcToTgt, intensityDerivTgt);
					if (j > 0)
						computeJacobianBlockIntensityRow_j(colorJacBlockRow_j, focalLength, state.d_xRot[j], state.d_xTrans[j], invTransform_i, camPosSrc, camPosSrcToTgt, intensityDerivTgt);
#endif
					colorWeight = parameters.weightDenseColor * imPairWeight * max(0.0f, 1.0f - abs(colorRes) / (1.15f * parameters.denseColorThresh));

				}
			}
			addToLocalSystem(foundCorrColor, state.d_denseJtJ, state.d_denseJtr, input.numberOfImages * 6, colorJacBlockRow_i, colorJacBlockRow_j, i, j, colorRes, colorWeight, idx, state.d_sumResidualColor, state.d_corrCountColor);
		}
	}
}

bool BuildDenseSystem(const SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer)
{
	const unsigned int N = input.numberOfImages;
	const int sizeJtr = 6 * N;
	const int sizeJtJ = sizeJtr * sizeJtr;

#ifdef PRINT_RESIDUALS_DENSE
	cutilSafeCall(cudaMemset(state.d_corrCount, 0, sizeof(int)));
	cutilSafeCall(cudaMemset(state.d_sumResidual, 0, sizeof(float)));
	cutilSafeCall(cudaMemset(state.d_corrCountColor, 0, sizeof(int)));
	cutilSafeCall(cudaMemset(state.d_sumResidualColor, 0, sizeof(float)));
#endif

#ifdef PRINT_ITER_POSES
	printPosesMatricesCU(state.d_xTransforms,N);
#endif


	const unsigned int maxDenseImPairs = input.numberOfImages * (input.numberOfImages - 1) / 2;
	cutilSafeCall(cudaMemset(state.d_denseCorrCounts, 0, sizeof(float) * maxDenseImPairs));
	cutilSafeCall(cudaMemset(state.d_denseJtJ, 0, sizeof(float) * sizeJtJ));
	cutilSafeCall(cudaMemset(state.d_denseJtr, 0, sizeof(float) * sizeJtr));
	cutilSafeCall(cudaMemset(state.d_numDenseOverlappingImages, 0, sizeof(int)));

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	dim3 gridImImOverlap;
	if (parameters.useDenseDepthAllPairwise)
	{
		gridImImOverlap = dim3(N, N, 1);
		FindImageImageCorr_Kernel<true><<<gridImImOverlap, dim3(1,1,1)>>>(input, state, parameters);
	}
	else
	{
		gridImImOverlap = dim3(N - 1, 1, 1);
		FindImageImageCorr_Kernel<false><<<gridImImOverlap, dim3(1,1,1)>>>(input, state, parameters);
	}

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	if (timer) timer->endEvent();

	int numOverlapImagePairs;
	cutilSafeCall(cudaMemcpy(&numOverlapImagePairs, state.d_numDenseOverlappingImages, sizeof(int), cudaMemcpyDeviceToHost));
	if (numOverlapImagePairs == 0) {
		printf("warning: no overlapping images for dense solve\n");
		return false;
	}
	const int reductionGlobal = (input.denseDepthWidth*input.denseDepthHeight + THREADS_PER_BLOCK_DENSE_DEPTH - 1) / THREADS_PER_BLOCK_DENSE_DEPTH;
	dim3 grid(numOverlapImagePairs, reductionGlobal);

	if (timer) timer->startEvent("BuildDenseDepthSystem - compute im-im weights");

	int *d_corres_coord;
	const int n_pixels = input.denseDepthWidth*input.denseDepthHeight;
	cutilSafeCall(cudaMalloc(&d_corres_coord, sizeof(int)*n_pixels*numOverlapImagePairs));
	cutilSafeCall(cudaMemset(d_corres_coord, 0, sizeof(int)*n_pixels*numOverlapImagePairs));
	FindDenseCorrespondences_Kernel<<<grid, THREADS_PER_BLOCK_DENSE_DEPTH>>>(input, state, parameters, d_corres_coord);
	cutilSafeCall(cudaFree(d_corres_coord));

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	if (timer) timer->endEvent();
	if (timer) timer->startEvent("BuildDenseDepthSystem - build jtj/jtr");

	if (parameters.weightDenseDepth > 0.0f)
	{
		if (parameters.weightDenseColor > 0.0f)
			BuildDenseSystem_Kernel<true, true><<<grid, THREADS_PER_BLOCK_DENSE_DEPTH>>>(input, state, parameters);
		else
			BuildDenseSystem_Kernel<true, false><<<grid, THREADS_PER_BLOCK_DENSE_DEPTH>>>(input, state, parameters);
	}
	else
	{
		BuildDenseSystem_Kernel<false, true><<<grid, THREADS_PER_BLOCK_DENSE_DEPTH>>>(input, state, parameters);
	}
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

#ifdef PRINT_RESIDUALS_DENSE
	if (parameters.weightDenseDepth > 0)
	{
		float sumResidual;
		int corrCount;
		cutilSafeCall(cudaMemcpy(&sumResidual, state.d_sumResidual, sizeof(float), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(&corrCount, state.d_corrCount, sizeof(int), cudaMemcpyDeviceToHost));
		printf("\tdense depth: weights * residual = %f * %f = %f\t[#corr = %d]\n", parameters.weightDenseDepth, sumResidual / parameters.weightDenseDepth, sumResidual, corrCount);
	}
	if (parameters.weightDenseColor > 0)
	{
		float sumResidual;
		int corrCount;
		cutilSafeCall(cudaMemcpy(&sumResidual, state.d_sumResidualColor, sizeof(float), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(&corrCount, state.d_corrCountColor, sizeof(int), cudaMemcpyDeviceToHost));
		printf("\tdense color: weights * residual = %f * %f = %f\t[#corr = %d]\n", parameters.weightDenseColor, sumResidual / parameters.weightDenseColor, sumResidual, corrCount);
	}
#endif

	const unsigned int flipgrid = (sizeJtJ + THREADS_PER_BLOCK_DENSE_DEPTH_FLIP - 1) / THREADS_PER_BLOCK_DENSE_DEPTH_FLIP;
	FlipJtJ_Kernel<<<flipgrid, THREADS_PER_BLOCK_DENSE_DEPTH_FLIP>>>(sizeJtJ, sizeJtr, state.d_denseJtJ);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	if (timer)
		timer->endEvent();
	return true;
}

__global__ void collectHighResidualsDevice(SolverInput input, SolverState state, SolverStateAnalysis analysis, SolverParameters parameters, unsigned int maxNumHighResiduals)
{
	const unsigned int N = input.numberOfCorrespondences;
	const unsigned int corrIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (corrIdx < N) {
		float residual = evalAbsMaxResidualDevice(corrIdx, input, state, parameters);
		if (residual > parameters.highResidualThresh) {
			int idx = atomicAdd(state.d_countHighResidual, 1);
			if (idx < maxNumHighResiduals) {
				analysis.d_maxResidual[idx] = residual;
				analysis.d_maxResidualIndex[idx] = corrIdx;
			}
		}
	}
}
extern "C" void collectHighResiduals(SolverInput& input, SolverState& state, SolverStateAnalysis& analysis, SolverParameters& parameters, CUDATimer* timer)
{
	if (timer) timer->startEvent(__FUNCTION__);
	cutilSafeCall(cudaMemset(state.d_countHighResidual, 0, sizeof(int)));

	const unsigned int N = input.numberOfCorrespondences;
	unsigned int maxNumHighResiduals = (input.maxCorrPerImage*input.maxNumberOfImages + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	collectHighResidualsDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, analysis, parameters, maxNumHighResiduals);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	if (timer) timer->endEvent();
}


__global__ void EvalMaxResidualDevice(SolverInput input, SolverState state, SolverStateAnalysis analysis, SolverParameters parameters)
{
	__shared__ int maxResIndex[THREADS_PER_BLOCK];
	__shared__ float maxRes[THREADS_PER_BLOCK];

	const unsigned int N = input.numberOfCorrespondences;
	const unsigned int corrIdx = blockIdx.x * blockDim.x + threadIdx.x;

	maxResIndex[threadIdx.x] = 0;
	maxRes[threadIdx.x] = 0.0f;

	if (corrIdx < N) {
		float residual = evalAbsMaxResidualDevice(corrIdx, input, state, parameters);

		maxRes[threadIdx.x] = residual;
		maxResIndex[threadIdx.x] = corrIdx;

		__syncthreads();

		for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride /= 2) {

			if (threadIdx.x < stride) {
				int first = threadIdx.x;
				int second = threadIdx.x + stride;
				if (maxRes[first] < maxRes[second]) {
					maxRes[first] = maxRes[second];
					maxResIndex[first] = maxResIndex[second];
				}
			}

			__syncthreads();
		}

		if (threadIdx.x == 0) {
			analysis.d_maxResidual[blockIdx.x] = maxRes[0];
			analysis.d_maxResidualIndex[blockIdx.x] = maxResIndex[0];
		}
	}
}

extern "C" void evalMaxResidual(SolverInput& input, SolverState& state, SolverStateAnalysis& analysis, SolverParameters& parameters, CUDATimer* timer)
{
	if (timer) timer->startEvent(__FUNCTION__);

	const unsigned int N = input.numberOfCorrespondences;
	EvalMaxResidualDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, analysis, parameters);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	if (timer) timer->endEvent();
}


__global__ void ResetResidualDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x == 0) state.d_sumResidual[0] = 0.0f;
}

__global__ void EvalResidualDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfCorrespondences;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	float residual = 0.0f;
	if (x < N) {
		residual = evalFDevice(x, input, state, parameters);
		atomicAdd(&state.d_sumResidual[0], residual);
	}
}

extern "C" float EvalResidual(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer)
{
	if (timer) timer->startEvent(__FUNCTION__);

	float residual = 0.0f;

	const unsigned int N = input.numberOfCorrespondences;
	ResetResidualDevice << < 1, 1, 1 >> >(input, state, parameters);
	EvalResidualDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);

	residual = state.getSumResidual();

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	if (timer) timer->endEvent();

	return residual;
}


__global__ void CountHighResidualsDevice(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfCorrespondences;
	const unsigned int corrIdx = blockIdx.x * blockDim.x + threadIdx.x;

	if (corrIdx < N) {
		float residual = evalAbsMaxResidualDevice(corrIdx, input, state, parameters);

		if (residual > parameters.verifyOptDistThresh)
			atomicAdd(state.d_countHighResidual, 1);
	}
}

extern "C" int countHighResiduals(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer)
{
	if (timer) timer->startEvent(__FUNCTION__);

	const unsigned int N = input.numberOfCorrespondences;
	cutilSafeCall(cudaMemset(state.d_countHighResidual, 0, sizeof(int)));
	CountHighResidualsDevice << <(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(input, state, parameters);

	int count;
	cutilSafeCall(cudaMemcpy(&count, state.d_countHighResidual, sizeof(int), cudaMemcpyDeviceToHost));
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	if (timer) timer->endEvent();
	return count;
}


__global__ void EvalGNConvergenceDevice(SolverInput input, SolverStateAnalysis analysis, SolverState state)
{
	__shared__ float maxVal[THREADS_PER_BLOCK];

	const unsigned int N = input.numberOfImages;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	maxVal[threadIdx.x] = 0.0f;

	if (x < N)
	{
		if (x == 0 || input.d_validImages[x] == 0)
			maxVal[threadIdx.x] = 0.0f;
		else {
			float3 r3 = fmaxf(fabs(state.d_deltaRot[x]), fabs(state.d_deltaTrans[x]));
			float r = fmaxf(r3.x, fmaxf(r3.y, r3.z));
			maxVal[threadIdx.x] = r;
		}
		__syncthreads();

		for (int stride = THREADS_PER_BLOCK / 2; stride > 0; stride /= 2) {
			if (threadIdx.x < stride) {
				int first = threadIdx.x;
				int second = threadIdx.x + stride;
				maxVal[first] = fmaxf(maxVal[first], maxVal[second]);
			}
			__syncthreads();
		}
		if (threadIdx.x == 0) {
			analysis.d_maxResidual[blockIdx.x] = maxVal[0];
		}
	}
}
float EvalGNConvergence(SolverInput& input, SolverState& state, SolverStateAnalysis& analysis, CUDATimer* timer)
{
	if (timer) timer->startEvent(__FUNCTION__);

	const unsigned int N = input.numberOfImages;
	const unsigned int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	EvalGNConvergenceDevice << < blocksPerGrid, THREADS_PER_BLOCK >> >(input, analysis, state);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	cutilSafeCall(cudaMemcpy(analysis.h_maxResidual, analysis.d_maxResidual, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost));
	cutilSafeCall(cudaMemcpy(analysis.h_maxResidualIndex, analysis.d_maxResidualIndex, sizeof(int) * blocksPerGrid, cudaMemcpyDeviceToHost));
	float maxVal = 0.0f;
	for (unsigned int i = 0; i < blocksPerGrid; i++) {
		if (maxVal < analysis.h_maxResidual[i]) maxVal = analysis.h_maxResidual[i];
	}
	if (timer) timer->endEvent();

	return maxVal;
}


template<bool useDense>
__global__ void PCGInit_Kernel1(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfImages;
	const int x = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0f;
	if (x > 0 && x < N)
	{
		float3 resRot, resTrans;
		evalMinusJTFDevice<useDense>(x, input, state, parameters, resRot, resTrans);

		state.d_rRot[x] = resRot;
		state.d_rTrans[x] = resTrans;

		const float3 pRot = state.d_precondionerRot[x] * resRot;
		state.d_pRot[x] = pRot;

		const float3 pTrans = state.d_precondionerTrans[x] * resTrans;
		state.d_pTrans[x] = pTrans;

		d = dot(resRot, pRot) + dot(resTrans, pTrans);
		state.d_Ap_XRot[x] = make_float3(0.0f, 0.0f, 0.0f);
		state.d_Ap_XTrans[x] = make_float3(0.0f, 0.0f, 0.0f);
	}

	d = warpReduce(d);
	if (threadIdx.x % WARP_SIZE == 0)
	{
		atomicAdd(state.d_scanAlpha, d);
	}
}

__global__ void PCGInit_Kernel2(unsigned int N, SolverState state)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x > 0 && x < N) state.d_rDotzOld[x] = state.d_scanAlpha[0];
}

void Initialization(SolverInput& input, SolverState& state, SolverParameters& parameters, CUDATimer* timer)
{
	const unsigned int N = input.numberOfImages;

	const int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	if (blocksPerGrid > THREADS_PER_BLOCK)
	{
		std::cout << "Too many variables for this block size. Maximum number of variables for two kernel scan: " << THREADS_PER_BLOCK*THREADS_PER_BLOCK << std::endl;
		while (1);
	}

	if (timer) timer->startEvent("Initialization");


	cutilSafeCall(cudaMemset(state.d_scanAlpha, 0, sizeof(float)));
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	if (parameters.useDense) PCGInit_Kernel1<true> << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state, parameters);
	else PCGInit_Kernel1<false> << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state, parameters);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	PCGInit_Kernel2 << <blocksPerGrid, THREADS_PER_BLOCK >> >(N, state);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	if (timer) timer->endEvent();

}


__global__ void PCGStep_Kernel_Dense_Brute(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfImages;
	const unsigned int x = blockIdx.x;

	if (x > 0 && x < N)
	{
		float3 rot, trans;
		applyJTJDenseBruteDevice(x, state, state.d_denseJtJ, input.numberOfImages, rot, trans);

		state.d_Ap_XRot[x] += rot;
		state.d_Ap_XTrans[x] += trans;
	}
}
__global__ void PCGStep_Kernel_Dense(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfImages;
	const unsigned int x = blockIdx.x;
	const unsigned int lane = threadIdx.x % WARP_SIZE;

	if (x > 0 && x < N)
	{
		float3 rot, trans;
		applyJTJDenseDevice(x, state, state.d_denseJtJ, input.numberOfImages, rot, trans, threadIdx.x);

		if (lane == 0)
		{
			atomicAdd(&state.d_Ap_XRot[x].x, rot.x);
			atomicAdd(&state.d_Ap_XRot[x].y, rot.y);
			atomicAdd(&state.d_Ap_XRot[x].z, rot.z);

			atomicAdd(&state.d_Ap_XTrans[x].x, trans.x);
			atomicAdd(&state.d_Ap_XTrans[x].y, trans.y);
			atomicAdd(&state.d_Ap_XTrans[x].z, trans.z);
		}
	}
}

__global__ void PCGStep_Kernel0(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfCorrespondences;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N)
	{
		const float3 tmp = applyJDevice(x, input, state, parameters);
		state.d_Jp[x] = tmp;
	}
}

__global__ void PCGStep_Kernel1a(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfImages;
	const unsigned int x = blockIdx.x;
	const unsigned int lane = threadIdx.x % WARP_SIZE;

	if (x > 0 && x < N)
	{
		float3 rot, trans;
		applyJTDevice(x, input, state, parameters, rot, trans, threadIdx.x, lane);

		if (lane == 0)
		{
			atomicAdd(&state.d_Ap_XRot[x].x, rot.x);
			atomicAdd(&state.d_Ap_XRot[x].y, rot.y);
			atomicAdd(&state.d_Ap_XRot[x].z, rot.z);

			atomicAdd(&state.d_Ap_XTrans[x].x, trans.x);
			atomicAdd(&state.d_Ap_XTrans[x].y, trans.y);
			atomicAdd(&state.d_Ap_XTrans[x].z, trans.z);
		}
	}
}

__global__ void PCGStep_Kernel1b(SolverInput input, SolverState state, SolverParameters parameters)
{
	const unsigned int N = input.numberOfImages;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	float d = 0.0f;
	if (x > 0 && x < N)
	{
		d = dot(state.d_pRot[x], state.d_Ap_XRot[x]) + dot(state.d_pTrans[x], state.d_Ap_XTrans[x]);
	}

	d = warpReduce(d);
	if (threadIdx.x % WARP_SIZE == 0)
	{
		atomicAdd(state.d_scanAlpha, d);
	}
}

__global__ void PCGStep_Kernel2(SolverInput input, SolverState state)
{
	const unsigned int N = input.numberOfImages;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	const float dotProduct = state.d_scanAlpha[0];

	float b = 0.0f;
	if (x > 0 && x < N)
	{
		float alpha = 0.0f;
		if (dotProduct > FLOAT_EPSILON) alpha = state.d_rDotzOld[x] / dotProduct;

		state.d_deltaRot[x] = state.d_deltaRot[x] + alpha*state.d_pRot[x];
		state.d_deltaTrans[x] = state.d_deltaTrans[x] + alpha*state.d_pTrans[x];

		float3 rRot = state.d_rRot[x] - alpha*state.d_Ap_XRot[x];
		state.d_rRot[x] = rRot;

		float3 rTrans = state.d_rTrans[x] - alpha*state.d_Ap_XTrans[x];
		state.d_rTrans[x] = rTrans;

		float3 zRot = state.d_precondionerRot[x] * rRot;
		state.d_zRot[x] = zRot;

		float3 zTrans = state.d_precondionerTrans[x] * rTrans;
		state.d_zTrans[x] = zTrans;

		b = dot(zRot, rRot) + dot(zTrans, rTrans);
	}
	b = warpReduce(b);
	if (threadIdx.x % WARP_SIZE == 0)
	{
		atomicAdd(&state.d_scanAlpha[1], b);
	}
}

template<bool lastIteration>
__global__ void PCGStep_Kernel3(SolverInput input, SolverState state)
{
	const unsigned int N = input.numberOfImages;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x > 0 && x < N)
	{
		const float rDotzNew = state.d_scanAlpha[1];
		const float rDotzOld = state.d_rDotzOld[x];

		float beta = 0.0f;
		if (rDotzOld > FLOAT_EPSILON) beta = rDotzNew / rDotzOld;

		state.d_rDotzOld[x] = rDotzNew;
		state.d_pRot[x] = state.d_zRot[x] + beta*state.d_pRot[x];
		state.d_pTrans[x] = state.d_zTrans[x] + beta*state.d_pTrans[x];


		state.d_Ap_XRot[x] = make_float3(0.0f, 0.0f, 0.0f);
		state.d_Ap_XTrans[x] = make_float3(0.0f, 0.0f, 0.0f);

		if (lastIteration)
		{
#ifdef USE_LIE_SPACE
			float3 rot, trans;
			computeLieUpdate(state.d_deltaRot[x], state.d_deltaTrans[x], state.d_xRot[x], state.d_xTrans[x], rot, trans);
			state.d_xRot[x] = rot;
			state.d_xTrans[x] = trans;
#else
			state.d_xRot[x] = state.d_xRot[x] + state.d_deltaRot[x];
			state.d_xTrans[x] = state.d_xTrans[x] + state.d_deltaTrans[x];
#endif
		}
	}
}

template<bool useSparse, bool useDense>
bool PCGIteration(SolverInput& input, SolverState& state, SolverParameters& parameters, SolverStateAnalysis& analysis, bool lastIteration, CUDATimer *timer)
{
	const unsigned int N = input.numberOfImages;
	const int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	if (blocksPerGrid > THREADS_PER_BLOCK)
	{
		std::cout << "Too many variables for this block size. Maximum number of variables for two kernel scan: " << THREADS_PER_BLOCK*THREADS_PER_BLOCK << std::endl;
		while (1);
	}
	if (timer) timer->startEvent("PCGIteration");

	cutilSafeCall(cudaMemset(state.d_scanAlpha, 0, sizeof(float) * 2));

	if (useSparse) {
		const unsigned int Ncorr = input.numberOfCorrespondences;
		const int blocksPerGridCorr = (Ncorr + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		PCGStep_Kernel0 << <blocksPerGridCorr, THREADS_PER_BLOCK >> >(input, state, parameters);
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif
		PCGStep_Kernel1a << < N, THREADS_PER_BLOCK_JT >> >(input, state, parameters);
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif
	}
	if (useDense) {
		PCGStep_Kernel_Dense << < N, THREADS_PER_BLOCK_JT_DENSE >> >(input, state, parameters);
#ifdef _DEBUG
		cutilSafeCall(cudaDeviceSynchronize());
		cutilCheckMsg(__FUNCTION__);
#endif
	}


	PCGStep_Kernel1b << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state, parameters);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	PCGStep_Kernel2 << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
#ifdef ENABLE_EARLY_OUT
	float scanAlpha; cutilSafeCall(cudaMemcpy(&scanAlpha, state.d_scanAlpha, sizeof(float), cudaMemcpyDeviceToHost));
	if (fabs(scanAlpha) < 5e-7) { lastIteration = true; }  //todo check this part
#endif
	if (lastIteration) {
		PCGStep_Kernel3<true> << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state);
	}
	else {
		PCGStep_Kernel3<false> << <blocksPerGrid, THREADS_PER_BLOCK >> >(input, state);
	}

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
	if (timer) timer->endEvent();

	return lastIteration;
}

#ifdef USE_LIE_SPACE
__global__ void convertLiePosesToMatricesCU_Kernel(const float3* d_rot, const float3* d_trans, unsigned int numTransforms, float4x4* d_transforms, float4x4* d_transformInvs)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numTransforms) {
		poseToMatrix(d_rot[idx], d_trans[idx], d_transforms[idx]);
		d_transformInvs[idx] = d_transforms[idx].getInverse();
	}
}
extern "C"
void convertLiePosesToMatricesCU(const float3* d_rot, const float3* d_trans, unsigned int numTransforms, float4x4* d_transforms, float4x4* d_transformInvs)
{
	convertLiePosesToMatricesCU_Kernel << <(numTransforms + 8 - 1) / 8, 8 >> >(d_rot, d_trans, numTransforms, d_transforms, d_transformInvs);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}
#endif


__global__ void printPosesMatricesCU_kernel(const float4x4* d_transforms, const int numTransforms)
{
	const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	printf("Rporting d_transforms\n");
	for (int i=0;i<numTransforms;i++)
	{
		printf("pose id=%d\n",i);
		d_transforms[i].print();
	}
}

void printPosesMatricesCU(const float4x4* d_transforms, const int numTransforms)
{
	printPosesMatricesCU_kernel<<<1,1>>>(d_transforms,numTransforms);
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}


extern "C" void solveBundlingStub(SolverInput& input, SolverState& state, SolverParameters& parameters, SolverStateAnalysis& analysis, float* convergenceAnalysis, CUDATimer *timer)
{
	if (convergenceAnalysis) {
		float initialResidual = EvalResidual(input, state, parameters, timer);
		convergenceAnalysis[0] = initialResidual;
	}

#ifdef PRINT_RESIDUALS_SPARSE
	if (parameters.weightSparse > 0) {
		if (input.numberOfCorrespondences == 0) { printf("ERROR: %d correspondences\n", input.numberOfCorrespondences); getchar(); }
		float initialResidual = EvalResidual(input, state, parameters, timer);
		printf("initial sparse = %f*%f = %f\n", parameters.weightSparse, initialResidual / parameters.weightSparse, initialResidual);
	}
#endif

	for (unsigned int nIter = 0; nIter < parameters.nNonLinearIterations; nIter++)
	{
		parameters.weightSparse = input.weightsSparse[nIter];
		parameters.weightDenseDepth = input.weightsDenseDepth[nIter];
		parameters.weightDenseColor = input.weightsDenseColor[nIter];
		parameters.useDense = (parameters.weightDenseDepth > 0 || parameters.weightDenseColor > 0);
#ifdef USE_LIE_SPACE
		convertLiePosesToMatricesCU(state.d_xRot, state.d_xTrans, input.numberOfImages, state.d_xTransforms, state.d_xTransformInverses);
#endif
		if (parameters.useDense)
			parameters.useDense = BuildDenseSystem(input, state, parameters, timer);
		Initialization(input, state, parameters, timer);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

		if (parameters.weightSparse > 0.0f) {
			if (parameters.useDense) {
				for (unsigned int linIter = 0; linIter < parameters.nLinIterations; linIter++)
					if (PCGIteration<true, true>(input, state, parameters, analysis, linIter == parameters.nLinIterations - 1, timer))
					{
						break;
					}
			}
			else {
				for (unsigned int linIter = 0; linIter < parameters.nLinIterations; linIter++)
					if (PCGIteration<true, false>(input, state, parameters, analysis, linIter == parameters.nLinIterations - 1, timer)) {
						//totalLinIters += (linIter+1); numLin++;
						break;
					}
			}
		}
		else {
			for (unsigned int linIter = 0; linIter < parameters.nLinIterations; linIter++)
				if (PCGIteration<false, true>(input, state, parameters, analysis, linIter == parameters.nLinIterations - 1, timer)) break;
		}

#ifdef PRINT_RESIDUALS_SPARSE
		if (parameters.weightSparse > 0) {
			float residual = EvalResidual(input, state, parameters, timer);
			printf("[niter %d] weight * sparse = %f*%f = %f\t[#corr = %d]\n", nIter, parameters.weightSparse, residual / parameters.weightSparse, residual, input.numberOfCorrespondences);
		}
#endif
		if (convergenceAnalysis) {
			float residual = EvalResidual(input, state, parameters, timer);
			convergenceAnalysis[nIter + 1] = residual;
		}

#ifdef ENABLE_EARLY_OUT
		if (nIter < parameters.nNonLinearIterations - 1 && EvalGNConvergence(input, state, analysis, timer) < 0.005f) {
			break;
		}
#endif
		}

}


__global__ void BuildVariablesToCorrespondencesTableDevice(EntryJ *d_correspondences, unsigned int numberOfCorrespondences,
																													 unsigned int maxNumCorrespondencesPerImage, int *d_variablesToCorrespondences, int *d_numEntriesPerRow)
{
	const unsigned int N = numberOfCorrespondences;
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;

	if (x < N)
	{
		EntryJ &corr = d_correspondences[x];
		if (corr.isValid())
		{
			int offset0 = atomicAdd(&d_numEntriesPerRow[corr.imgIdx_i], 1);
			int offset1 = atomicAdd(&d_numEntriesPerRow[corr.imgIdx_j], 1);
			if (offset0 < maxNumCorrespondencesPerImage && offset1 < maxNumCorrespondencesPerImage)
			{
				d_variablesToCorrespondences[corr.imgIdx_i * maxNumCorrespondencesPerImage + offset0] = x;
				d_variablesToCorrespondences[corr.imgIdx_j * maxNumCorrespondencesPerImage + offset1] = x;
			}
			else
			{
				printf("EXCEEDED MAX NUM CORR PER IMAGE IN SOLVER, INVALIDATING %d(%d,%d) [%d,%d | %d]\n", x, corr.imgIdx_i, corr.imgIdx_j, offset0, offset1, maxNumCorrespondencesPerImage);
				corr.setInvalid();
			}
		}
	}
}

extern "C" void buildVariablesToCorrespondencesTableCUDA(EntryJ* d_correspondences, unsigned int numberOfCorrespondences, unsigned int maxNumCorrespondencesPerImage, int* d_variablesToCorrespondences, int* d_numEntriesPerRow, CUDATimer* timer)
{
	const unsigned int N = numberOfCorrespondences;

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	if (timer) timer->startEvent(__FUNCTION__);

	BuildVariablesToCorrespondencesTableDevice<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_correspondences, numberOfCorrespondences, maxNumCorrespondencesPerImage, d_variablesToCorrespondences, d_numEntriesPerRow);

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	if (timer) timer->endEvent();
}
