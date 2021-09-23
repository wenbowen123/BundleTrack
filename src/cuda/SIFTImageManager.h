

#pragma once

#ifndef _IMAGE_MANAGER_H_
#define _IMAGE_MANAGER_H_

#include <cuda_runtime.h>
#include "cutil_inline.h"

#include <vector>
#include <cassert>
#include <iostream>
#include <vector>

#include "GlobalDefines.h"
#include "cuda_SimpleMatrixUtil.h"
#include "../CUDACacheUtil.h"
#include "CUDATimer.h"


struct SIFTKeyPoint {
	float2 pos;
	float scale;
	float depth;
};

struct SIFTKeyPointDesc {
	unsigned char feature[128];
};

struct SIFTImageGPU {
	SIFTKeyPoint*			d_keyPoints;
	SIFTKeyPointDesc*		d_keyPointDescs;
};

struct ImagePairMatch {
	int*		d_numMatches;
	float*		d_distances;
	uint2*		d_keyPointIndices;
};


struct EntryJ {
	unsigned int imgIdx_i;
	unsigned int imgIdx_j;
	float3 pos_i;
	float3 pos_j;

	__host__ __device__
	void setInvalid() {
		imgIdx_i = (unsigned int)-1;
		imgIdx_j = (unsigned int)-1;
	}
	__host__ __device__
	bool isValid() const {
		return imgIdx_i != (unsigned int)-1;
	}
};



class SIFTImageManager {
public:
	friend class SIFTMatchFilter;
	friend class TestMatching;

	SIFTImageManager(unsigned int maxImages = 500,
		unsigned int maxKeyPointsPerImage = 4096);

	~SIFTImageManager();


	SIFTImageGPU& getImageGPU(unsigned int imageIdx);

	const SIFTImageGPU& getImageGPU(unsigned int imageIdx) const;

	unsigned int getNumImages() const;

	unsigned int getNumKeyPointsPerImage(unsigned int imageIdx) const;
	unsigned int getMaxNumKeyPointsPerImage() const { return m_maxKeyPointsPerImage; }

	SIFTImageGPU& createSIFTImageGPU();

	void finalizeSIFTImageGPU(unsigned int numKeyPoints);

	ImagePairMatch& /*SIFTImageManager::*/getImagePairMatch(unsigned int prevImageIdx, unsigned int curImageIdx, uint2& keyPointOffset);//

	ImagePairMatch& getImagePairMatchDEBUG(unsigned int prevImageIdx, unsigned int curImageIdx, uint2& keyPointOffset)
	{
		assert(prevImageIdx < getNumImages());
		assert(curImageIdx < getNumImages());
		keyPointOffset = make_uint2(m_numKeyPointsPerImagePrefixSum[prevImageIdx], m_numKeyPointsPerImagePrefixSum[curImageIdx]);
		return m_currImagePairMatches[prevImageIdx];
	}

	void reset() {
		m_SIFTImagesGPU.clear();
		m_numKeyPointsPerImage.clear();
		m_numKeyPointsPerImagePrefixSum.clear();
		m_numKeyPoints = 0;
		m_globNumResiduals = 0;
		m_bFinalizedGPUImage = false;
		MLIB_CUDA_SAFE_CALL(cudaMemset(d_globNumResiduals, 0, sizeof(int)));

		m_validImages.clear();
		m_validImages.resize(m_maxNumImages, 0);
		m_validImages[0] = 1;
	}

	void SortKeyPointMatchesCU(unsigned int curFrame, unsigned int startFrame, unsigned int numFrames);

	void FilterKeyPointMatchesCU(unsigned int curFrame, unsigned int startFrame, unsigned int numFrames, const float4x4& siftIntrinsicsInv, unsigned int minNumMatches, float maxKabschRes2);

	void FilterMatchesBySurfaceAreaCU(unsigned int curFrame, unsigned int startFrame, unsigned int numFrames, const float4x4& colorIntrinsicsInv, float areaThresh);

	void FilterMatchesByDenseVerifyCU(unsigned int curFrame, unsigned int startFrame, unsigned int numFrames, unsigned int imageWidth, unsigned int imageHeight,
		const float4x4& intrinsics, const CUDACachedFrame* d_cachedFrames,
		float distThresh, float normalThresh, float colorThresh, float errThresh, float corrThresh, float sensorDepthMin, float sensorDepthMax);

	int VerifyTrajectoryCU(unsigned int numImages, float4x4* d_trajectory,
		unsigned int imageWidth, unsigned int imageHeight,
		const float4x4& intrinsics, const CUDACachedFrame* d_cachedFrames,
		float distThresh, float normalThresh, float colorThresh, float errThresh, float corrThresh,
		float sensorDepthMin, float sensorDepthMax);

	void AddCurrToResidualsCU(unsigned int curFrame, unsigned int startFrame, unsigned int numFrames, const float4x4& colorIntrinsicsInv);

	void InvalidateImageToImageCU(const uint2& imageToImageIdx);

	void CheckForInvalidFramesSimpleCU(const int* d_varToCorrNumEntriesPerRow, unsigned int numVars);
	void CheckForInvalidFramesCU(const int* d_varToCorrNumEntriesPerRow, unsigned int numVars);

	unsigned int filterFrames(unsigned int curFrame, unsigned int startFrame, unsigned int numFrames);

	const std::vector<int>& getValidImages() const { return m_validImages; }
	void invalidateFrame(unsigned int frame) { m_validImages[frame] = 0; }

	void updateGPUValidImages() {
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_validImages, m_validImages.data(), sizeof(int)*getNumImages(), cudaMemcpyHostToDevice));
	}
	const int* getValidImagesGPU() const { return d_validImages; }

	int* debugGetNumRawMatchesGPU() {
		return d_currNumMatchesPerImagePair;
	}
	int* debugGetNumFiltMatchesGPU() {
		return d_currNumFilteredMatchesPerImagePair;
	}

	unsigned int getTotalNumKeyPoints() const { return m_numKeyPoints; }
	void setNumImagesDEBUG(unsigned int numImages) {
		MLIB_ASSERT(numImages <= m_SIFTImagesGPU.size());
		if (numImages == m_SIFTImagesGPU.size()) return;
		m_SIFTImagesGPU.resize(numImages);
		m_numKeyPointsPerImage.resize(numImages);
		m_numKeyPointsPerImagePrefixSum.resize(numImages);
		m_numKeyPoints = m_numKeyPointsPerImagePrefixSum.back();
	}
	void setGlobalCorrespondencesDEBUG(const std::vector<EntryJ>& correspondences) {
		MLIB_ASSERT(correspondences.size() < MAX_MATCHES_PER_IMAGE_PAIR_FILTERED * (m_maxNumImages*(m_maxNumImages - 1)) / 2);
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_globMatches, correspondences.data(), sizeof(EntryJ)*correspondences.size(), cudaMemcpyHostToDevice));
		m_globNumResiduals = (unsigned int)correspondences.size();
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(d_globNumResiduals, &m_globNumResiduals, sizeof(unsigned int), cudaMemcpyHostToDevice));
	}
	void setValidImagesDEBUG(const std::vector<int>& valid) {
		m_validImages = valid;
	}
	void getNumRawMatchesDEBUG(std::vector<unsigned int>& numMatches) const {
		MLIB_ASSERT(getNumImages() > 1);
		if (getCurrentFrame() + 1 == getNumImages()) numMatches.resize(getNumImages() - 1);
		else										 numMatches.resize(getNumImages());
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(numMatches.data(), d_currNumMatchesPerImagePair, sizeof(unsigned int)*numMatches.size(), cudaMemcpyDeviceToHost));
	}
	void getNumFiltMatchesDEBUG(std::vector<unsigned int>& numMatches) const {
		MLIB_ASSERT(getNumImages() > 1);
		if (getCurrentFrame() + 1 == getNumImages()) numMatches.resize(getNumImages() - 1);
		else										 numMatches.resize(getNumImages());
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(numMatches.data(), d_currNumFilteredMatchesPerImagePair, sizeof(unsigned int)*numMatches.size(), cudaMemcpyDeviceToHost));
	}
	void getSIFTKeyPointsDEBUG(std::vector<SIFTKeyPoint>& keys) const {
		keys.resize(m_numKeyPoints);
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(keys.data(), d_keyPoints, sizeof(SIFTKeyPoint) * keys.size(), cudaMemcpyDeviceToHost));
	}
	void getSIFTKeyPointDescsDEBUG(std::vector<SIFTKeyPointDesc>& descs) const {
		descs.resize(m_numKeyPoints);
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(descs.data(), d_keyPointDescs, sizeof(SIFTKeyPointDesc) * descs.size(), cudaMemcpyDeviceToHost));
	}
	void getRawKeyPointIndicesAndMatchDistancesDEBUG(unsigned int imagePairIndex, std::vector<uint2>& keyPointIndices, std::vector<float>& matchDistances) const
	{
		unsigned int numMatches;
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&numMatches, d_currNumMatchesPerImagePair + imagePairIndex, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		if (numMatches > MAX_MATCHES_PER_IMAGE_PAIR_RAW) numMatches = MAX_MATCHES_PER_IMAGE_PAIR_RAW;
		keyPointIndices.resize(numMatches);
		matchDistances.resize(numMatches);
		if (numMatches > 0) {
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(keyPointIndices.data(), d_currMatchKeyPointIndices + imagePairIndex * MAX_MATCHES_PER_IMAGE_PAIR_RAW, sizeof(uint2) * numMatches, cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(matchDistances.data(), d_currMatchDistances + imagePairIndex * MAX_MATCHES_PER_IMAGE_PAIR_RAW, sizeof(float) * numMatches, cudaMemcpyDeviceToHost));
		}
	}
	void getFiltKeyPointIndicesAndMatchDistancesDEBUG(unsigned int imagePairIndex, std::vector<uint2>& keyPointIndices, std::vector<float>& matchDistances) const
	{
		unsigned int numMatches;
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&numMatches, d_currNumFilteredMatchesPerImagePair + imagePairIndex, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		MLIB_ASSERT(numMatches <= MAX_MATCHES_PER_IMAGE_PAIR_FILTERED);
		keyPointIndices.resize(numMatches);
		matchDistances.resize(numMatches);
		if (numMatches > 0) {
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(keyPointIndices.data(), d_currFilteredMatchKeyPointIndices + imagePairIndex * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, sizeof(uint2) * numMatches, cudaMemcpyDeviceToHost));
			MLIB_CUDA_SAFE_CALL(cudaMemcpy(matchDistances.data(), d_currFilteredMatchKeyPointIndices + imagePairIndex * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, sizeof(float) * numMatches, cudaMemcpyDeviceToHost));
		}
	}
	void getCurrMatchKeyPointIndicesDEBUG(std::vector<uint2>& keyPointIndices, std::vector<unsigned int>& numMatches, bool filtered) const
	{
		numMatches.resize(getNumImages());
		if (filtered)	{ MLIB_CUDA_SAFE_CALL(cudaMemcpy(numMatches.data(), d_currNumFilteredMatchesPerImagePair, sizeof(unsigned int)*numMatches.size(), cudaMemcpyDeviceToHost)); }
		else			{ MLIB_CUDA_SAFE_CALL(cudaMemcpy(numMatches.data(), d_currNumMatchesPerImagePair, sizeof(unsigned int)*numMatches.size(), cudaMemcpyDeviceToHost)); }
		if (filtered)	{ keyPointIndices.resize(numMatches.size() * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED); MLIB_CUDA_SAFE_CALL(cudaMemcpy(keyPointIndices.data(), d_currFilteredMatchKeyPointIndices, sizeof(uint2) * keyPointIndices.size(), cudaMemcpyDeviceToHost)); }
		else			{ keyPointIndices.resize(numMatches.size() * MAX_MATCHES_PER_IMAGE_PAIR_RAW); MLIB_CUDA_SAFE_CALL(cudaMemcpy(keyPointIndices.data(), d_currMatchKeyPointIndices, sizeof(uint2) * keyPointIndices.size(), cudaMemcpyDeviceToHost)); }
	}
	void getFiltKeyPointIndicesDEBUG(unsigned int imagePairIndex, std::vector<uint2>& keyPointIndices) const
	{
		unsigned int numMatches;
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(&numMatches, d_currNumFilteredMatchesPerImagePair + imagePairIndex, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		keyPointIndices.resize(numMatches);
		MLIB_CUDA_SAFE_CALL(cudaMemcpy(keyPointIndices.data(), d_currFilteredMatchKeyPointIndices + imagePairIndex * MAX_MATCHES_PER_IMAGE_PAIR_FILTERED, sizeof(uint2) * numMatches, cudaMemcpyDeviceToHost));
	}
	const EntryJ* getGlobalCorrespondencesGPU() const { return d_globMatches; }
	EntryJ* getGlobalCorrespondencesGPU() { return d_globMatches; }
	unsigned int getNumGlobalCorrespondences() const { return m_globNumResiduals; }
	const float4x4* getFiltTransformsToWorldGPU() const { return d_currFilteredTransformsInv; }
	const int* getNumFiltMatchesGPU() const { return d_currNumFilteredMatchesPerImagePair; }

	void fuseToGlobal(SIFTImageManager* global, const float4x4& colorIntrinsics, const float4x4* d_transforms,
		const float4x4& colorIntrinsicsInv) const;
	void computeTracks(const std::vector<float4x4>& trajectory, const std::vector<EntryJ>& correspondences, const std::vector<uint2>& correspondenceKeyIndices,
		std::vector< std::vector<std::pair<uint2, float3>> >& tracks) const;

	//try to match previously invalidated images
	bool getTopRetryImage(unsigned int& idx) {
		if (m_imagesToRetry.empty()) return false;
		idx = m_imagesToRetry.front();
		m_imagesToRetry.pop_front();
		return true;
	}
	void addToRetryList(unsigned int idx) {
		m_imagesToRetry.push_front(idx);
	}
	unsigned int getCurrentFrame() const { return m_currentImage; }
	void setCurrentFrame(unsigned int idx) { m_currentImage = idx; }

	static void TestSVDDebugCU(const float3x3& m);

	void saveToFile(const std::string& s);

	void loadFromFile(const std::string& s);

	void evaluateTimings() {
		if (m_timer) m_timer->evaluate(true);
	}
	void setTimer(CUDATimer* timer) {
		m_timer = timer;
	}

	void lock() {
		m_mutex.lock();
	}

	void unlock() {
		m_mutex.unlock();
	}

public:
	std::mutex m_mutex;

	void alloc();
	void free();
	void initializeMatching();

	void fuseLocalKeyDepths(std::vector<SIFTKeyPoint>& globalKeys, const std::vector<float*>& depthFrames,
		unsigned int depthWidth, unsigned int depthHeight,
		const std::vector<float4x4>& transforms, const std::vector<float4x4>& transformsInv,
		const float4x4& siftIntrinsicsInv, const float4x4& depthIntrinsics, const float4x4& depthIntrinsicsInv) const;

	std::vector<SIFTImageGPU>	m_SIFTImagesGPU;
	bool						m_bFinalizedGPUImage;

	unsigned int				m_numKeyPoints;
	std::vector<unsigned int>	m_numKeyPointsPerImage;
	std::vector<unsigned int>	m_numKeyPointsPerImagePrefixSum;

	SIFTKeyPoint*			d_keyPoints;
	SIFTKeyPointDesc*		d_keyPointDescs;


	std::vector<ImagePairMatch>	m_currImagePairMatches;

	int*			d_currNumMatchesPerImagePair;
	float*			d_currMatchDistances;
	uint2*			d_currMatchKeyPointIndices;

	int*			d_currNumFilteredMatchesPerImagePair;
	float*			d_currFilteredMatchDistances;
	uint2*			d_currFilteredMatchKeyPointIndices;
	float4x4*		d_currFilteredTransforms;
	float4x4*		d_currFilteredTransformsInv;

	std::vector<int> m_validImages;
	int*			 d_validImages;

	unsigned int	m_globNumResiduals;
	int*			d_globNumResiduals;
	EntryJ*			d_globMatches;
	uint2*			d_globMatchesKeyPointIndices;
	int*			d_validOpt;

	unsigned int m_maxNumImages;
	unsigned int m_maxKeyPointsPerImage;

	std::list<unsigned int> m_imagesToRetry;
	unsigned int			m_currentImage;

	CUDATimer *m_timer;
};


#endif

