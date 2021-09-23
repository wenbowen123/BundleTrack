#pragma once

#include "CUDACacheUtil.h"
#include "CUDAImageUtil.h"
#include "core-math/matrix4x4.h"
#include "mLib.h"

class CUDACache {
public:

	CUDACache(unsigned int widthDepthInput, unsigned int heightDepthInput, unsigned int widthDownSampled, unsigned int heightDownSampled, unsigned int maxNumImages, const ml::mat4f& inputIntrinsics);
	~CUDACache();
	void alloc();
	void free();

	void storeFrame(unsigned int inputDepthWidth, unsigned int inputDepthHeight, const float* d_depth, const uchar4* d_color, const float4 *d_normals);

	void reset() {
		m_currentFrame = 0;
	}

	const std::vector<CUDACachedFrame>& getCacheFrames() const { return m_cache; }
	const CUDACachedFrame* getCacheFramesGPU() const { return d_cache; }

	void copyCacheFrameFrom(CUDACache* other, unsigned int frameFrom) {
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_depthDownsampled, other->m_cache[frameFrom].d_depthDownsampled, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_cameraposDownsampled, other->m_cache[frameFrom].d_cameraposDownsampled, sizeof(float4) * m_width * m_height, cudaMemcpyDeviceToDevice));

		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_intensityDownsampled, other->m_cache[frameFrom].d_intensityDownsampled, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice));
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_intensityDerivsDownsampled, other->m_cache[frameFrom].d_intensityDerivsDownsampled, sizeof(float2) * m_width * m_height, cudaMemcpyDeviceToDevice));
#ifdef CUDACACHE_UCHAR_NORMALS
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_normalsDownsampledUCHAR4, other->m_cache[frameFrom].d_normalsDownsampledUCHAR4, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice));
#endif
#ifdef CUDACACHE_FLOAT_NORMALS
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[m_currentFrame].d_normalsDownsampled, other->m_cache[frameFrom].d_normalsDownsampled, sizeof(float4) * m_width * m_height, cudaMemcpyDeviceToDevice));
#endif
		m_currentFrame++;
	}

	void incrementCache() {
		m_currentFrame++;
	}

	unsigned int getWidth() const { return m_width; }
	unsigned int getHeight() const { return m_height; }

	const ml::mat4f& getIntrinsics() const { return m_intrinsics; }
	const ml::mat4f& getIntrinsicsInv() const { return m_intrinsicsInv; }

	unsigned int getNumFrames() const { return m_currentFrame; }

	void saveToFile(const std::string& filename) const {
		BinaryDataStreamFile s(filename, true);
		s << m_width;
		s << m_height;
		s << m_intrinsics;
		s << m_intrinsicsInv;
		s << m_currentFrame;
		s << m_maxNumImages;

		DepthImage32 depth(m_width, m_height);
		ColorImageR32G32B32A32 camPos(m_width, m_height), normals(m_width, m_height);
		ColorImageR32 intensity(m_width, m_height);
		BaseImage<vec2f> intensityDerivative(m_width, m_height);
		ColorImageR32 intensityOrig(m_width, m_height);
		for (unsigned int i = 0; i < m_currentFrame; i++) {
			const CUDACachedFrame& f = m_cache[i];
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(depth.getData(), f.d_depthDownsampled, sizeof(float)*depth.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(camPos.getData(), f.d_cameraposDownsampled, sizeof(float4)*camPos.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(intensity.getData(), f.d_intensityDownsampled, sizeof(float)*intensity.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(intensityDerivative.getData(), f.d_intensityDerivsDownsampled, sizeof(float2)*intensityDerivative.getNumPixels(), cudaMemcpyDeviceToHost));
#ifdef CUDACACHE_UCHAR_NORMALS
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(intensityOrig.getData(), f.d_normalsDownsampledUCHAR4, sizeof(float)*intensityOrig.getNumPixels(), cudaMemcpyDeviceToHost));
#endif
#ifdef CUDACACHE_FLOAT_NORMALS
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(normals.getData(), f.d_normalsDownsampled, sizeof(float4)*normals.getNumPixels(), cudaMemcpyDeviceToHost));
#endif
			s << depth;
			s << camPos;
			s << normals;
			s << intensity;
			s << intensityDerivative;
			s << intensityOrig;
		}
		s.close();
	}
	void loadFromFile(const std::string& filename) {
		unsigned int oldMaxNumImages = m_maxNumImages;
		unsigned int oldWidth = m_width;
		unsigned int oldHeight = m_height;
		BinaryDataStreamFile s(filename, false);
		s >> m_width;
		s >> m_height;
		s >> m_intrinsics;
		s >> m_intrinsicsInv;
		s >> m_currentFrame;
		s >> m_maxNumImages;

		if (m_maxNumImages > oldMaxNumImages || m_width > oldWidth || m_height > oldHeight) {
			free();
			alloc();
		}

		DepthImage32 depth(m_width, m_height);
		ColorImageR32G32B32A32 camPos(m_width, m_height), normals(m_width, m_height);
		ColorImageR32 intensity(m_width, m_height);
		BaseImage<vec2f> intensityDerivative(m_width, m_height);
		ColorImageR32 intensityOrig(m_width, m_height);
		for (unsigned int i = 0; i < m_currentFrame; i++) {
			const CUDACachedFrame& f = m_cache[i];
			s >> depth;
			s >> camPos;
			s >> normals;
			s >> intensity;
			s >> intensityDerivative;
			s >> intensityOrig;
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_depthDownsampled, depth.getData(), sizeof(float)*depth.getNumPixels(), cudaMemcpyHostToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_cameraposDownsampled, camPos.getData(), sizeof(float4)*camPos.getNumPixels(), cudaMemcpyHostToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_intensityDownsampled, intensity.getData(), sizeof(float)*intensity.getNumPixels(), cudaMemcpyHostToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_intensityDerivsDownsampled, intensityDerivative.getData(), sizeof(float2)*intensityDerivative.getNumPixels(), cudaMemcpyHostToDevice));
#ifdef CUDACACHE_UCHAR_NORMALS
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_normalsDownsampledUCHAR4, intensityOrig.getData(), sizeof(float)*intensityOrig.getNumPixels(), cudaMemcpyHostToDevice));
#endif
#ifdef CUDACACHE_FLOAT_NORMALS
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(f.d_normalsDownsampled, normals.getData(), sizeof(float4)*normals.getNumPixels(), cudaMemcpyHostToDevice));
#endif
		}
		s.close();
	}

	void printCacheImages(std::string outDir) const {
		if (m_cache.empty()) return;
		if (!(outDir.back() == '/' || outDir.back() == '\\')) outDir.push_back('/');
		if (!util::directoryExists(outDir)) util::makeDirectory(outDir);

		ColorImageR32 intensity(m_width, m_height); DepthImage32 depth(m_width, m_height);
		ColorImageR32G32B32A32 image(m_width, m_height); ColorImageR8G8B8A8 image8(m_width, m_height);
		image.setInvalidValue(vec4f(-std::numeric_limits<float>::infinity()));
		for (unsigned int i = 0; i < m_cache.size(); i++) {
			const CUDACachedFrame& f = m_cache[i];
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(depth.getData(), f.d_depthDownsampled, sizeof(float)*depth.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(intensity.getData(), f.d_intensityDownsampled, sizeof(float)*intensity.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(image8.getData(), f.d_normalsDownsampledUCHAR4, sizeof(uchar4)*image8.getNumPixels(), cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(image.getData(), f.d_normalsDownsampled, sizeof(float4)*image.getNumPixels(), cudaMemcpyDeviceToHost));
                        for (const auto& p : image)                    {
				if (p.value.x != -std::numeric_limits<float>::infinity()) {
					p.value.w = 1.0f;
					image8(p.x, p.y).w = 255;
				}
			}
		}
	}

	std::vector<CUDACachedFrame>& getCachedFramesDEBUG() { return m_cache; }
	void setCurrentFrame(unsigned int c) { m_currentFrame = c; }
	void setIntrinsics(const ml::mat4f& inputIntrinsics, const ml::mat4f& intrinsics) {
		m_inputIntrinsics = inputIntrinsics; m_inputIntrinsicsInv = inputIntrinsics.getInverse();
		m_intrinsics = intrinsics; m_intrinsicsInv = intrinsics.getInverse();
	}

	void setCachedFrames(const std::vector<CUDACachedFrame>& cachedFrames) {
		MLIB_ASSERT(cachedFrames.size() <= m_cache.size());
		for (unsigned int i = 0; i < cachedFrames.size(); i++) {
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_depthDownsampled, cachedFrames[i].d_depthDownsampled, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_cameraposDownsampled, cachedFrames[i].d_cameraposDownsampled, sizeof(float4) * m_width * m_height, cudaMemcpyDeviceToDevice));

			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_intensityDownsampled, cachedFrames[i].d_intensityDownsampled, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_intensityDerivsDownsampled, cachedFrames[i].d_intensityDerivsDownsampled, sizeof(float2) * m_width * m_height, cudaMemcpyDeviceToDevice));
#ifdef CUDACACHE_UCHAR_NORMALS
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_normalsDownsampledUCHAR4, cachedFrames[i].d_normalsDownsampledUCHAR4, sizeof(float) * m_width * m_height, cudaMemcpyDeviceToDevice));
#endif
#ifdef CUDACACHE_FLOAT_NORMALS
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(m_cache[i].d_normalsDownsampled, cachedFrames[i].d_normalsDownsampled, sizeof(float4) * m_width * m_height, cudaMemcpyDeviceToDevice));
#endif
		}
	}

	void fuseDepthFrames(CUDACache* globalCache, const int* d_validImages, const float4x4* d_transforms) const;

private:
	unsigned int m_width;
	unsigned int m_height;
	ml::mat4f		 m_intrinsics;
	ml::mat4f		 m_intrinsicsInv;

	unsigned int m_currentFrame;
	unsigned int m_maxNumImages;

	std::vector < CUDACachedFrame > m_cache;
	CUDACachedFrame*				d_cache;

	float* d_filterHelper;
	float4* d_helperCamPos, *d_helperNormals;
	unsigned int m_inputDepthWidth;
	unsigned int m_inputDepthHeight;
	ml::mat4f		 m_inputIntrinsics;
	ml::mat4f		 m_inputIntrinsicsInv;

	float* d_intensityHelper;
	float m_filterIntensitySigma;
	float m_filterDepthSigmaD;
	float m_filterDepthSigmaR;
};
