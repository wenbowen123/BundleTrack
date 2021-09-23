
#include "common.h"
#include "cutil_inline.h"
#include "cutil_math.h"
#include "cuda_SimpleMatrixUtil.h"

#define MLIB_CUDA_SAFE_CALL(b) { if(b != cudaSuccess) throw MLIB_EXCEPTION(std::string(cudaGetErrorString(b)) + ":" + std::string(__FUNCTION__)); }
#define MLIB_CUDA_CHECK_ERR(msg) {  cudaError_t err = cudaGetLastError();	if (err != cudaSuccess) { throw MLIB_EXCEPTION(cudaGetErrorString( err )); } }


#define MINF __int_as_float(0xff800000)
