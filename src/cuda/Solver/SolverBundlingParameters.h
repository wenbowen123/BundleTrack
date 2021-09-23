#pragma once

#ifndef _SOLVER_PARAMETERS_
#define _SOLVER_PARAMETERS_

struct SolverParameters
{
	unsigned int nNonLinearIterations;
	unsigned int nLinIterations;

	float verifyOptDistThresh;
	float verifyOptPercentThresh;

	float highResidualThresh;
	float robust_delta;

	float denseDistThresh;
	float denseNormalThresh;
	float denseColorThresh;
	float denseColorGradientMin;
	float denseDepthMin;
	float denseDepthMax;

	bool useDenseDepthAllPairwise;
	unsigned int denseOverlapCheckSubsampleFactor;

	float weightSparse;
	float weightDenseDepth;
	float weightDenseColor;
	bool useDense;
};

#endif
