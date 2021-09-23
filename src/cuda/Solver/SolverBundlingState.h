#pragma once

#ifndef _SOLVER_STATE_
#define _SOLVER_STATE_

#include "cuda_runtime.h"
#include "../SIFTImageManager.h"
#include "../CUDACacheUtil.h"

struct SolverInput
{
	EntryJ* d_correspondences;
	int* d_variablesToCorrespondences;
	int* d_numEntriesPerRow;

	unsigned int numberOfCorrespondences;
	unsigned int numberOfImages;

	unsigned int maxNumberOfImages;
	unsigned int maxCorrPerImage;

	const int* d_validImages;
	const CUDACachedFrame* d_cacheFrames;
	unsigned int denseDepthWidth;
	unsigned int denseDepthHeight;
	float4 intrinsics;
	unsigned int maxNumDenseImPairs;
	float2 colorFocalLength;

	const float* weightsSparse;
	const float* weightsDenseDepth;
	const float* weightsDenseColor;
};

struct SolverState
{
	float3*	d_deltaRot;
	float3*	d_deltaTrans;

	float3* d_xRot;
	float3* d_xTrans;

	float3*	d_rRot;
	float3*	d_rTrans;

	float3*	d_zRot;
	float3*	d_zTrans;

	float3*	d_pRot;
	float3*	d_pTrans;

	float3*	d_Jp;

	float3*	d_Ap_XRot;
	float3*	d_Ap_XTrans;

	float*	d_scanAlpha;

	float*	d_rDotzOld;

	float3*	d_precondionerRot;
	float3*	d_precondionerTrans;

	float*	d_sumResidual;


	int* d_countHighResidual;

	__host__ float getSumResidual() const {
		float residual;
		cudaMemcpy(&residual, d_sumResidual, sizeof(float), cudaMemcpyDeviceToHost);
		return residual;
	}

	float* d_denseJtJ;
	float* d_denseJtr;
	float* d_denseCorrCounts;

	float4x4* d_xTransforms;
	float4x4* d_xTransformInverses;

	uint2* d_denseOverlappingImages;
	int* d_numDenseOverlappingImages;

	int* d_corrCount;
	int* d_corrCountColor;
	float* d_sumResidualColor;
};

struct SolverStateAnalysis
{
	int*	d_maxResidualIndex;
	float*	d_maxResidual;

	int*	h_maxResidualIndex;
	float*	h_maxResidual;
};

#endif
