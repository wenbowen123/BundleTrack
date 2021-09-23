#include "mLib.h"
#include "GlobalDefines.h"
#include "CUDACache.h"
#include "MatrixConversion.h"

#ifdef CUDACACHE_UCHAR_NORMALS
extern "C" void fuseCacheFramesCU(const CUDACachedFrame* d_frames, const int* d_validImages, const float4& intrinsics, const float4x4* d_transforms,
	unsigned int numFrames, unsigned int width, unsigned int height, float* d_output, float2* d_tmp, const uchar4* d_normals);
#else
extern "C" void fuseCacheFramesCU(const CUDACachedFrame* d_frames, const int* d_validImages, const float4& intrinsics, const float4x4* d_transforms,
	unsigned int numFrames, unsigned int width, unsigned int height, float* d_output, float2* d_tmp, const float4* d_normals);
#endif

CUDACache::CUDACache(unsigned int widthDepthInput, unsigned int heightDepthInput, unsigned int widthDownSampled, unsigned int heightDownSampled, unsigned int maxNumImages, const mat4f& inputIntrinsics)
{
	m_width = widthDownSampled;
	m_height = heightDownSampled;
	m_maxNumImages = maxNumImages;

	m_intrinsics = inputIntrinsics;
	m_intrinsics._m00 *= (float)widthDownSampled / (float)widthDepthInput;
	m_intrinsics._m11 *= (float)heightDownSampled / (float)heightDepthInput;
	m_intrinsics._m02 *= (float)(widthDownSampled -1)/ (float)(widthDepthInput-1);
	m_intrinsics._m12 *= (float)(heightDownSampled-1) / (float)(heightDepthInput-1);
	m_intrinsicsInv = m_intrinsics.getInverse();

	m_filterIntensitySigma = 2.5;
	m_filterDepthSigmaD = 1.0;
	m_filterDepthSigmaR = 0.05;

	m_inputDepthWidth = widthDepthInput;
	m_inputDepthHeight = heightDepthInput;
	m_inputIntrinsics = inputIntrinsics;
	m_inputIntrinsicsInv = m_inputIntrinsics.getInverse();

	alloc();
	m_currentFrame = 0;
}

CUDACache::~CUDACache()
{
	free();
}

void CUDACache::alloc()
{
	m_cache.resize(m_maxNumImages);
	for (CUDACachedFrame &f : m_cache)
	{
		f.alloc(m_width, m_height);
	}
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_cache, sizeof(CUDACachedFrame) * m_maxNumImages));
	MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_cache, m_cache.data(), sizeof(CUDACachedFrame) * m_cache.size(), cudaMemcpyHostToDevice));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_intensityHelper, sizeof(float) * m_width * m_height));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_filterHelper, sizeof(float) * m_inputDepthWidth * m_inputDepthHeight));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_helperCamPos, sizeof(float4) * m_inputDepthWidth * m_inputDepthHeight));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_helperNormals, sizeof(float4) * m_inputDepthWidth * m_inputDepthHeight));
}

void CUDACache::free()
{
	for (CUDACachedFrame &f : m_cache)
	{
		f.free();
	}
	m_cache.clear();
	cutilSafeCall(cudaFree(d_cache));
	cutilSafeCall(cudaFree(d_intensityHelper));
	cutilSafeCall(cudaFree(d_filterHelper));
	cutilSafeCall(cudaFree(d_helperCamPos));
	cutilSafeCall(cudaFree(d_helperNormals));

	m_currentFrame = 0;
}

void CUDACache::storeFrame(unsigned int inputDepthWidth, unsigned int inputDepthHeight, const float* d_depth, const uchar4* d_color, const float4 *d_normals)
{
	CUDACachedFrame& frame = m_cache[m_currentFrame];

	CUDAImageUtil::convertDepthFloatToCameraSpaceFloat4(d_helperCamPos, d_depth, *(float4x4*)&m_inputIntrinsicsInv, inputDepthWidth, inputDepthHeight);
	CUDAImageUtil::resampleFloat4(frame.d_cameraposDownsampled, m_width, m_height, d_helperCamPos, inputDepthWidth, inputDepthHeight);
	CUDAImageUtil::resampleFloat4(frame.d_normalsDownsampled, m_width, m_height, d_normals, inputDepthWidth, inputDepthHeight);
	CUDAImageUtil::resampleFloat(frame.d_depthDownsampled, m_width, m_height, d_depth, inputDepthWidth, inputDepthHeight);

	CUDAImageUtil::countNumValidDepth(frame.d_num_valid_points, frame.d_depthDownsampled, m_height, m_width);

	m_currentFrame++;
}

void CUDACache::fuseDepthFrames(CUDACache* globalCache, const int* d_validImages, const float4x4* d_transforms) const
{
	assert(globalCache->m_currentFrame > 0);
	const unsigned int numFrames = m_currentFrame;
	const unsigned int globalFrameIdx = globalCache->m_currentFrame - 1;

	CUDACachedFrame &globalFrame = globalCache->m_cache[globalFrameIdx];
	if (globalFrameIdx + 1 == globalCache->m_maxNumImages)
	{
		std::cerr << "CUDACache reached max # images!" << std::endl;
		while (1);
	}
	CUDACachedFrame &tmpFrame = globalCache->m_cache[globalFrameIdx + 1];

	float4 intrinsics = make_float4(m_intrinsics(0, 0), m_intrinsics(1, 1), m_intrinsics(0, 2), m_intrinsics(1, 2));
#ifdef CUDACACHE_UCHAR_NORMALS
	fuseCacheFramesCU(d_cache, d_validImages, intrinsics, d_transforms, numFrames, m_width, m_height,
										globalFrame.d_depthDownsampled, tmpFrame.d_intensityDerivsDownsampled, globalFrame.d_normalsDownsampledUCHAR4);
#else
	fuseCacheFramesCU(d_cache, d_validImages, intrinsics, d_transforms, numFrames, m_width, m_height,
										globalFrame.d_depthDownsampled, tmpFrame.d_intensityDerivsDownsampled, globalFrame.d_normalsDownsampled);
#endif
	CUDAImageUtil::convertDepthFloatToCameraSpaceFloat4(globalFrame.d_cameraposDownsampled, globalFrame.d_depthDownsampled, MatrixConversion::toCUDA(m_intrinsicsInv), m_width, m_height);
#ifdef CUDACACHE_UCHAR_NORMALS
	CUDAImageUtil::computeNormals(d_helperNormals, globalFrame.d_cameraposDownsampled, m_width, m_height);
	CUDAImageUtil::convertNormalsFloat4ToUCHAR4(globalFrame.d_normalsDownsampledUCHAR4, d_helperNormals, m_width, m_height);
#endif
#ifdef CUDACACHE_FLOAT_NORMALS
	CUDAImageUtil::computeNormals(globalFrame.d_normalsDownsampled, globalFrame.d_cameraposDownsampled, m_width, m_height);
#endif
}
