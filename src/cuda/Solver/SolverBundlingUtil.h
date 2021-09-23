#pragma once

#ifndef _SOLVER_Stereo_UTIL_
#define _SOLVER_Stereo_UTIL_

#include "../SolverUtil.h"

#include "cutil_inline.h"
#include <cutil_math.h>

#define THREADS_PER_BLOCK 512
#define WARP_SIZE 32

__inline__ __device__ float warpReduce(float val) {
	int offset = 32 >> 1;
	while (offset > 0) {
		val = val + __shfl_down_sync(FULL_MASK, val, offset, 32);
		offset = offset >> 1;
	}
	return val;
}


__inline__ __device__ void huberLoss(const float e, const float delta, float3 &rho)
{
	float dsqr = delta * delta;
	if (e <= dsqr)
	{
		rho.x = e;
		rho.y = 1.;
		rho.z = 0.;
	}
	else
	{
		double sqrte = sqrt(e);
		rho.x = 2 * sqrte * delta - dsqr;
		rho.y = delta / sqrte;
		rho.z = -0.5 * rho.y / e;
	}
}


#endif
