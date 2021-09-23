#pragma once


#include "cudaUtil.h"
#include "SIFTImageManager.h"
#include "core-math/vec3.h"
#include "core-util/timer.h"
#include "Solver/CUDASolverBundling.h"
#include "core-util/binaryDataStream.h"
#include "yaml-cpp/yaml.h"

struct JacobianBlock {
	ml::vec3f data[6];
};

class SBA
{
public:
	SBA(int maxNumImages, int maxNumResiduals, int max_corr_per_image, std::shared_ptr<YAML::Node> yml1);
	~SBA();
	void init(unsigned int maxImages, unsigned int maxNumResiduals, unsigned int max_corr_per_image);

	bool align(const std::vector<EntryJ> &global_corres, const std::vector<int> &n_match_per_pair, int n_images, const CUDACache* cudaCache, float4x4* d_transforms, bool useVerify, bool isLocal, bool recordConvergence, bool isStart, bool isEnd, bool isScanDoneOpt, unsigned int revalidateIdx);

	float getMaxResidual() const { return m_maxResidual; }
	const std::vector<float>& getLinearConvergenceAnalysis() const { return m_solver->getLinearConvergenceAnalysis(); }
	bool useVerification() const { return m_bVerify; }

	void evaluateSolverTimings() {
		m_solver->evaluateTimings();
	}
	void printConvergence(const std::string& filename) const;


	void setGlobalWeights(const std::vector<float>& weightsSparse, const std::vector<float>& weightsDenseDepth, const std::vector<float>& weightsDenseColor, bool useGlobalDenseOpt) {
		m_globalWeightsMutex.lock();
		m_globalWeightsSparse = weightsSparse;
		m_globalWeightsDenseDepth = weightsDenseDepth;
		m_globalWeightsDenseColor = weightsDenseColor;
		m_bUseGlobalDenseOpt = useGlobalDenseOpt;
		m_globalWeightsMutex.unlock();
	}

	void saveLogRemovedCorrToFile(const std::string& prefix) const {
		ml::BinaryDataStreamFile s(prefix + ".bin", true);
		s << _logRemovedImImCorrs.size();
		if (!_logRemovedImImCorrs.empty()) s.writeData((const BYTE*)_logRemovedImImCorrs.data(), sizeof(std::pair<ml::vec2ui, float>)*_logRemovedImImCorrs.size());
		s.close();

		std::ofstream os(prefix + ".txt");
		os << "# remove im-im correspondences = " << _logRemovedImImCorrs.size() << std::endl;
		for (unsigned int i = 0; i < _logRemovedImImCorrs.size(); i++)
			os << _logRemovedImImCorrs[i].first << "\t\t" << _logRemovedImImCorrs[i].second << std::endl;
		os.close();
	}

private:

	bool alignCUDA(const CUDACache* cudaCache, bool useDensePairwise,
		const std::vector<float>& weightsSparse, const std::vector<float>& weightsDenseDepth, const std::vector<float>& weightsDenseColor, bool isStart, bool isEnd, unsigned int revalidateIdx);
	bool removeMaxResidualCUDA(unsigned int numImages, unsigned int curFrame);

	float3*			d_xRot;
	float3*			d_xTrans;
	unsigned int	m_numCorrespondences;
	EntryJ*			d_correspondences;
	int* d_validImages;


	std::vector<EntryJ> _global_corres;
	std::vector<int> _n_match_per_pair;
	int _n_images;

	bool m_bUseLocalDense;
	bool m_bUseGlobalDenseOpt;
	std::vector<float> m_localWeightsSparse;
	std::vector<float> m_localWeightsDenseDepth;
	std::vector<float> m_localWeightsDenseColor;
	std::vector<float> m_globalWeightsSparse;
	std::vector<float> m_globalWeightsDenseDepth;
	std::vector<float> m_globalWeightsDenseColor;
	std::mutex m_globalWeightsMutex;

	CUDASolverBundling* m_solver;

	bool m_bUseComprehensiveFrameInvalidation;

	float m_maxResidual;
	bool m_bVerify;

	std::vector< std::vector<float> > m_recordedConvergence;

	static ml::Timer s_timer;
	std::shared_ptr<YAML::Node> yml;

	std::vector<std::pair<ml::vec2ui, float>> _logRemovedImImCorrs;
};

