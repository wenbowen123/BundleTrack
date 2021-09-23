#pragma once

#ifndef _SOLVER_EQUATIONS_LIE_
#define _SOLVER_EQUATIONS_LIE_


#include "GlobalDefines.h"
#ifdef USE_LIE_SPACE

#define THREADS_PER_BLOCK_JT_DENSE 128
#define THREADS_PER_BLOCK_JT 128

#include "cutil_inline.h"
#include "cutil_math.h"

#include "../cuda_SimpleMatrixUtil.h"

#include "SolverBundlingUtil.h"
#include "SolverBundlingState.h"
#include "SolverBundlingParameters.h"

#include "ICPUtil.h"
#include "LieDerivUtil.h"


__inline__ __device__ float evalAbsMaxResidualDevice(unsigned int corrIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	float3 r = make_float3(0.0f, 0.0f, 0.0f);

	const EntryJ& corr = input.d_correspondences[corrIdx];
	if (corr.isValid()) {
		float4x4 TI = poseToMatrix(state.d_xRot[corr.imgIdx_i], state.d_xTrans[corr.imgIdx_i]);
		float4x4 TJ = poseToMatrix(state.d_xRot[corr.imgIdx_j], state.d_xTrans[corr.imgIdx_j]);
		r = parameters.weightSparse * fabs((TI*corr.pos_i) - (TJ*corr.pos_j));

		return max(r.z, max(r.x, r.y));
	}
	return 0.0f;
}

__inline__ __device__ float evalFDevice(unsigned int corrIdx, SolverInput& input, SolverState& state, SolverParameters& parameters)
{
	float3 r = make_float3(0.0f, 0.0f, 0.0f);

	const EntryJ& corr = input.d_correspondences[corrIdx];
	if (corr.isValid()) {
		float4x4 TI = poseToMatrix(state.d_xRot[corr.imgIdx_i], state.d_xTrans[corr.imgIdx_i]);
		float4x4 TJ = poseToMatrix(state.d_xRot[corr.imgIdx_j], state.d_xTrans[corr.imgIdx_j]);

		r = (TI*corr.pos_i) - (TJ*corr.pos_j);

		float res = parameters.weightSparse * dot(r, r);
		return res;
	}
	return 0.0f;
}



template<bool useDense>
__inline__ __device__ void evalMinusJTFDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, SolverParameters& parameters, float3& resRot, float3& resTrans)
{
	float3 rRot = make_float3(0.0f, 0.0f, 0.0f);
	float3 rTrans = make_float3(0.0f, 0.0f, 0.0f);

	float3 pRot = make_float3(0.0f, 0.0f, 0.0f);
	float3 pTrans = make_float3(0.0f, 0.0f, 0.0f);

	state.d_deltaRot[variableIdx] = make_float3(0.0f, 0.0f, 0.0f);
	state.d_deltaTrans[variableIdx] = make_float3(0.0f, 0.0f, 0.0f);

	int N = min(input.d_numEntriesPerRow[variableIdx], input.maxCorrPerImage);

	for (int i = 0; i < N; i++)
	{
		int corrIdx = input.d_variablesToCorrespondences[variableIdx*input.maxCorrPerImage + i];
		const EntryJ &corr = input.d_correspondences[corrIdx];
		if (corr.isValid())
		{
			const float4x4 TI = state.d_xTransforms[corr.imgIdx_i];
			const float4x4 TJ = state.d_xTransforms[corr.imgIdx_j];

			float3 worldP;
			float variableSign = 1;
			if (variableIdx != corr.imgIdx_i)
			{
				variableSign = -1;
				worldP = TJ * corr.pos_j;
			}
			else
			{
				worldP = TI * corr.pos_i;
			}
			const float3 da = evalLie_dAlpha(worldP);
			const float3 db = evalLie_dBeta(worldP);
			const float3 dc = evalLie_dGamma(worldP);

			const float3 r = (TI * corr.pos_i) - (TJ * corr.pos_j);
			float3 rho;
			const float e = dot(r,r);
			huberLoss(e, parameters.robust_delta, rho);


			rRot += rho.y * variableSign * make_float3(dot(da, r), dot(db, r), dot(dc, r));
			rTrans += rho.y * variableSign * r;

			pRot += rho.y * make_float3(dot(da, da), dot(db, db), dot(dc, dc));
			pTrans += rho.y * make_float3(1.0f, 1.0f, 1.0f);
		}
	}
	resRot = -parameters.weightSparse * rRot;
	resTrans = -parameters.weightSparse * rTrans;

	if (useDense) {
		uint3 transIndices = make_uint3(variableIdx * 6 + 0, variableIdx * 6 + 1, variableIdx * 6 + 2);
		uint3 rotIndices = make_uint3(variableIdx * 6 + 3, variableIdx * 6 + 4, variableIdx * 6 + 5);
		resRot -= make_float3(state.d_denseJtr[rotIndices.x], state.d_denseJtr[rotIndices.y], state.d_denseJtr[rotIndices.z]);
		resTrans -= make_float3(state.d_denseJtr[transIndices.x], state.d_denseJtr[transIndices.y], state.d_denseJtr[transIndices.z]);
	}
	if (pRot.x > FLOAT_EPSILON)   state.d_precondionerRot[variableIdx].x = 1.0f / pRot.x;
	else					      state.d_precondionerRot[variableIdx].x = 1.0f;

	if (pRot.y > FLOAT_EPSILON)   state.d_precondionerRot[variableIdx].y = 1.0f / pRot.y;
	else					      state.d_precondionerRot[variableIdx].y = 1.0f;

	if (pRot.z > FLOAT_EPSILON)   state.d_precondionerRot[variableIdx].z = 1.0f / pRot.z;
	else						  state.d_precondionerRot[variableIdx].z = 1.0f;

	if (pTrans.x > FLOAT_EPSILON) state.d_precondionerTrans[variableIdx].x = 1.0f / pTrans.x;
	else					      state.d_precondionerTrans[variableIdx].x = 1.0f;

	if (pTrans.y > FLOAT_EPSILON) state.d_precondionerTrans[variableIdx].y = 1.0f / pTrans.y;
	else					      state.d_precondionerTrans[variableIdx].y = 1.0f;

	if (pTrans.z > FLOAT_EPSILON) state.d_precondionerTrans[variableIdx].z = 1.0f / pTrans.z;
	else					      state.d_precondionerTrans[variableIdx].z = 1.0f;
}


__inline__ __device__ void applyJTDevice(unsigned int variableIdx, SolverInput& input, SolverState& state, const SolverParameters& parameters,
	float3& outRot, float3& outTrans, unsigned int threadIdx, unsigned int lane)
{
	outRot = make_float3(0.0f, 0.0f, 0.0f);
	outTrans = make_float3(0.0f, 0.0f, 0.0f);

	int N = min(input.d_numEntriesPerRow[variableIdx], input.maxCorrPerImage);

	for (int i = threadIdx; i < N; i += THREADS_PER_BLOCK_JT)
	{
		int corrIdx = input.d_variablesToCorrespondences[variableIdx*input.maxCorrPerImage + i];
		const EntryJ& corr = input.d_correspondences[corrIdx];
		if (corr.isValid()) {
			const float4x4 TI = state.d_xTransforms[corr.imgIdx_i];
			const float4x4 TJ = state.d_xTransforms[corr.imgIdx_j];

			float3 worldP;
			float  variableSign = 1;
			if (variableIdx != corr.imgIdx_i)
			{
				variableSign = -1;
				worldP = TJ * corr.pos_j;
			}
			else {
				worldP = TI * corr.pos_i;
			}
			const float3 da = evalLie_dAlpha(worldP);
			const float3 db = evalLie_dBeta(worldP);
			const float3 dc = evalLie_dGamma(worldP);

			outRot += variableSign * make_float3(dot(da, state.d_Jp[corrIdx]), dot(db, state.d_Jp[corrIdx]), dot(dc, state.d_Jp[corrIdx]));
			outTrans += variableSign * state.d_Jp[corrIdx];
		}
	}

	outRot.x = warpReduce(outRot.x);	 outRot.y = warpReduce(outRot.y);	  outRot.z = warpReduce(outRot.z);
	outTrans.x = warpReduce(outTrans.x); outTrans.y = warpReduce(outTrans.y); outTrans.z = warpReduce(outTrans.z);
}

__inline__ __device__ float3 applyJDevice(unsigned int corrIdx, SolverInput& input, SolverState& state, const SolverParameters& parameters)
{
	float3 b = make_float3(0.0f, 0.0f, 0.0f);
	const EntryJ& corr = input.d_correspondences[corrIdx];

	if (corr.isValid()) {
		if (corr.imgIdx_i > 0)
		{
			const float4x4 TI = state.d_xTransforms[corr.imgIdx_i];
			const float3 worldP = TI * corr.pos_i;
			const float3 da = evalLie_dAlpha(worldP);
			const float3 db = evalLie_dBeta(worldP);
			const float3 dc = evalLie_dGamma(worldP);

			const float3  pp0 = state.d_pRot[corr.imgIdx_i];
			b += da*pp0.x + db*pp0.y + dc*pp0.z + state.d_pTrans[corr.imgIdx_i];
		}

		if (corr.imgIdx_j > 0)
		{
			const float4x4 TJ = state.d_xTransforms[corr.imgIdx_j];
			const float3 worldP = TJ * corr.pos_j;
			const float3 da = evalLie_dAlpha(worldP);
			const float3 db = evalLie_dBeta(worldP);
			const float3 dc = evalLie_dGamma(worldP);

			const float3  pp1 = state.d_pRot[corr.imgIdx_j];
			b -= da*pp1.x + db*pp1.y + dc*pp1.z + state.d_pTrans[corr.imgIdx_j];
		}
		b *= parameters.weightSparse;
	}
	return b;
}


__inline__ __device__ void computeJacobianBlockRow_i(matNxM<1, 6>& jacBlockRow, const float4x4& transform_i,
	const float4x4& invTransform_j, const float3& camPosSrc, const float3& normalTgt)
{
	matNxM<3, 6> jac = evalLie_derivI(invTransform_j, transform_i, camPosSrc);
	for (unsigned int i = 0; i < 6; i++) {
		jacBlockRow(i) = -dot(make_float3(jac(0, i), jac(1, i), jac(2, i)), normalTgt);
	}
}

__inline__ __device__ void computeJacobianBlockRow_j(matNxM<1, 6>& jacBlockRow, const float4x4& invTransform_i,
	const float4x4& transform_j, const float3& camPosSrc, const float3& normalTgt)
{
	matNxM<3, 6> jac = evalLie_derivJ(invTransform_i, transform_j, camPosSrc);
	for (unsigned int i = 0; i < 6; i++) {
		jacBlockRow(i) = -dot(make_float3(jac(0, i), jac(1, i), jac(2, i)), normalTgt);
	}
}

__inline__ __device__ float computeColorDProjLookup(const float4& dx, const float3& camPosSrcToTgt, const float2& intensityDerivTgt, const float2& colorFocalLength)
{
	mat3x1 dcdx; dcdx(0) = dx.x; dcdx(1) = dx.y; dcdx(2) = dx.z;
	mat2x3 dProjectionC = dCameraToScreen(camPosSrcToTgt, colorFocalLength.x, colorFocalLength.y);
	mat1x2 dColorB(intensityDerivTgt);
	mat1x1 dadx = dColorB * dProjectionC * dcdx;

	return dadx(0);
}
__inline__ __device__ void computeJacobianBlockIntensityRow_i(matNxM<1, 6>& jacBlockRow, const float2& colorFocal, const float4x4& transform_i,
	const float4x4& invTransform_j, const float3& camPosSrc, const float3& camPosSrcToTgt, const float2& intensityDerivTgt)
{
	matNxM<3, 6> jac = evalLie_derivI(invTransform_j, transform_i, camPosSrc);
	mat2x3 dProj = dCameraToScreen(camPosSrcToTgt, colorFocal.x, colorFocal.y);
	mat1x2 dColorB(intensityDerivTgt);
	jacBlockRow = dColorB * (dProj * jac);
}
__inline__ __device__ void computeJacobianBlockIntensityRow_j(matNxM<1, 6>& jacBlockRow, const float2& colorFocal, const float4x4& invTransform_i,
	const float4x4& transform_j, const float3& camPosSrc, const float3& camPosSrcToTgt, const float2& intensityDerivTgt)
{
	matNxM<3, 6> jac = evalLie_derivJ(invTransform_i, transform_j, camPosSrc);
	mat2x3 dProj = dCameraToScreen(camPosSrcToTgt, colorFocal.x, colorFocal.y);
	mat1x2 dColorB(intensityDerivTgt);
	jacBlockRow = dColorB * (dProj * jac);

}
#endif

#endif
