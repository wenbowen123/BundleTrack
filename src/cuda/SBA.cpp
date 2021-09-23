
//#include "stdafx.h"
#include "SBA.h"
#include "CUDACache.h"
#include "TimingLog.h"
#include <chrono>

#define POSESIZE 6

extern "C" void convertMatricesToPosesCU(const float4x4* d_transforms, unsigned int numTransforms,
	float3* d_rot, float3* d_trans, const int* d_validImages);

extern "C" void convertPosesToMatricesCU(const float3* d_rot, const float3* d_trans, unsigned int numImages, float4x4* d_transforms, const int* d_validImages);



SBA::SBA(int maxNumImages, int maxNumResiduals, int max_corr_per_image, std::shared_ptr<YAML::Node> yml1)
{
	yml = yml1;
#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	m_bUseComprehensiveFrameInvalidation = false;

	const unsigned int maxNumIts = (*yml)["bundle"]["num_iter_outter"].as<int>();
	m_localWeightsSparse.resize(maxNumIts, 1.0f);

	m_localWeightsDenseDepth.resize(maxNumIts,1);

	m_localWeightsDenseColor.resize(maxNumIts, 0.0f);

	m_maxResidual = -1.0f;

	m_bUseGlobalDenseOpt = true;
	m_bUseLocalDense = true;



	init(maxNumImages, maxNumResiduals, max_corr_per_image);
}

SBA::~SBA()
{
	SAFE_DELETE(m_solver);
	cutilSafeCall(cudaFree(d_xRot));
	cutilSafeCall(cudaFree(d_xTrans));

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif
}

void SBA::init(unsigned int maxImages, unsigned int maxNumResiduals, unsigned int max_corr_per_image)
{

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	unsigned int maxNumImages = maxImages;
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_xRot, sizeof(float3)*maxNumImages));
	MLIB_CUDA_SAFE_CALL(cudaMalloc(&d_xTrans, sizeof(float3)*maxNumImages));

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	m_solver = new CUDASolverBundling(maxImages, maxNumResiduals,max_corr_per_image,yml);
	m_bVerify = false;

	m_bUseComprehensiveFrameInvalidation = false;
	m_bUseLocalDense = true;
}


bool SBA::align(const std::vector<EntryJ> &global_corres, const std::vector<int> &n_match_per_pair, int n_images, const CUDACache* cudaCache, float4x4* d_transforms, bool useVerify, bool isLocal, bool recordConvergence, bool isStart, bool isEnd, bool isScanDoneOpt, unsigned int revalidateIdx)
{
	if (recordConvergence) m_recordedConvergence.push_back(std::vector<float>());
	_global_corres = global_corres;
	_n_match_per_pair = n_match_per_pair;
	_n_images = n_images;
	m_bVerify = false;
	m_maxResidual = -1.0f;

	bool usePairwise = true;
	const CUDACache* cache = cudaCache;
	std::vector<float> weightsDenseDepth, weightsDenseColor, weightsSparse;

	weightsSparse = m_localWeightsSparse;

	weightsDenseDepth = m_localWeightsDenseDepth;
	weightsDenseColor = m_localWeightsDenseColor;

	unsigned int numImages = n_images;
	std::vector<int> valid_images_cpu(n_images,1);
	cutilSafeCall(cudaMalloc(&d_validImages, n_images*sizeof(int)));
	cutilSafeCall(cudaMemcpy(d_validImages, valid_images_cpu.data(), sizeof(int)*n_images, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMalloc(&d_correspondences, sizeof(EntryJ)*_global_corres.size()));
	cudaMemcpy(d_correspondences, _global_corres.data(), sizeof(EntryJ)*_global_corres.size(), cudaMemcpyHostToDevice);

	convertMatricesToPosesCU(d_transforms, numImages, d_xRot, d_xTrans, d_validImages);

	bool removed = alignCUDA(cache, usePairwise, weightsSparse, weightsDenseDepth, weightsDenseColor, isStart, isEnd, revalidateIdx);
	if (recordConvergence) {
		const std::vector<float>& conv = m_solver->getConvergenceAnalysis();
		m_recordedConvergence.back().insert(m_recordedConvergence.back().end(), conv.begin(), conv.end());
	}


	convertPosesToMatricesCU(d_xRot, d_xTrans, numImages, d_transforms, d_validImages);

	cutilSafeCall(cudaFree(d_validImages));
	cutilSafeCall(cudaFree(d_correspondences));

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

	return removed;
}

bool SBA::alignCUDA(const CUDACache* cudaCache, bool useDensePairwise, const std::vector<float>& weightsSparse, const std::vector<float>& weightsDenseDepth, const std::vector<float>& weightsDenseColor, bool isStart, bool isEnd, unsigned int revalidateIdx)
{
	m_numCorrespondences = _global_corres.size();
	unsigned int numImages = _n_images;
	auto begin = std::chrono::steady_clock::now();
	m_solver->solve(d_correspondences, m_numCorrespondences, d_validImages, numImages, cudaCache, weightsSparse, weightsDenseDepth, weightsDenseColor, useDensePairwise, d_xRot, d_xTrans, isStart, isEnd, revalidateIdx);
	auto end = std::chrono::steady_clock::now();
	std::cout << "m_solver->solve Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()/1000.0 << "[ms]" << std::endl;
	bool removed = false;

	return removed;
}


#ifdef NEW_GUIDED_REMOVE
template<>
struct std::hash<ml::vec2ui> : public std::unary_function < ml::vec2ui, size_t > {
	size_t operator()(const ml::vec2ui& v) const {
		const size_t p0 = 73856093;
		const size_t p1 = 19349669;
		const size_t res = ((size_t)v.x * p0) ^ ((size_t)v.y * p1);
		return res;
	}
};
#endif

namespace std {
template<>
struct hash<vec2ui> : public std::unary_function < vec2ui, size_t > {
	size_t operator()(const vec2ui& v) const {
		const size_t p0 = 73856093;
		const size_t p1 = 19349669;
		const size_t res = ((size_t)v.x * p0) ^ ((size_t)v.y * p1);
		return res;
	}
};
}
bool SBA::removeMaxResidualCUDA(unsigned int numImages, unsigned int curFrame)
{

	return false;
}

void SBA::printConvergence(const std::string& filename) const
{
	if (m_recordedConvergence.empty()) return;
	std::ofstream s(filename);
	for (unsigned int i = 0; i < m_recordedConvergence.size(); i++) {
		for (unsigned int k = 0; k < m_recordedConvergence[i].size(); k++)
			s << m_recordedConvergence[i][k] << std::endl;
	}
	s.close();
}

