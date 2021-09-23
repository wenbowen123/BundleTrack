#pragma once

#include <cuda_runtime.h>
//#include <cuda_d3d11_interop.h>
#include "core-math/vec2.h"
#include "../cudaUtil.h"
#include "SolverBundlingParameters.h"
#include "SolverBundlingState.h"

#include "../cuda_SimpleMatrixUtil.h"
#include "../CUDATimer.h"
#include "yaml-cpp/yaml.h"


class CUDACache;


class CUDASolverBundling
{
public:

	CUDASolverBundling(unsigned int maxNumberOfImages, unsigned int maxNumResiduals, const int max_corr_per_image, std::shared_ptr<YAML::Node> yml1);
	~CUDASolverBundling();

	void solve(EntryJ* d_correspondences, unsigned int numberOfCorrespondences,	const int* d_validImages, unsigned int numberOfImages, const CUDACache* cudaCache, const std::vector<float>& weightsSparse, const std::vector<float>& weightsDenseDepth, const std::vector<float>& weightsDenseColor, bool usePairwiseDense, float3* d_rotationAnglesUnknowns, float3* d_translationUnknowns, bool rebuildJT, bool findMaxResidual, unsigned int revalidateIdx);
	const std::vector<float>& getConvergenceAnalysis() const { return m_convergence; }
	const std::vector<float>& getLinearConvergenceAnalysis() const { return m_linConvergence; }

	void getMaxResidual(float& max, int& index) const {
		max = m_solverExtra.h_maxResidual[0];
		index = m_solverExtra.h_maxResidualIndex[0];
	};
	bool getMaxResidual(unsigned int curFrame, EntryJ* d_correspondences, ml::vec2ui& imageIndices, float& maxRes);
	bool useVerification(EntryJ* d_correspondences, unsigned int numberOfCorrespondences);

	const int* getVariablesToCorrespondences() const { return d_variablesToCorrespondences; }
	const int* getVarToCorrNumEntriesPerRow() const { return d_numEntriesPerRow; }

	void evaluateTimings() {
		if (m_timer) {
			m_timer->evaluate(true);
			std::cout << std::endl << std::endl;
		}
	}

	void resetTimer() {
		if (m_timer) m_timer->reset();
	}

#ifdef NEW_GUIDED_REMOVE
	const std::vector<ml::vec2ui>& getGuidedMaxResImagesToRemove() const { return m_maxResImPairs; }
#endif
private:

	static bool isSimilarImagePair(const ml::vec2ui& pair0, const ml::vec2ui& pair1) {
		if ((std::abs((int)pair0.x - (int)pair1.x) < 10 && std::abs((int)pair0.y - (int)pair1.y) < 10) ||
			(std::abs((int)pair0.x - (int)pair1.y) < 10 && std::abs((int)pair0.y - (int)pair1.x) < 10))
			return true;
		return false;
	}


	void buildVariablesToCorrespondencesTable(EntryJ* d_correspondences, unsigned int numberOfCorrespondences);
	void computeMaxResidual(SolverInput& solverInput, SolverParameters& parameters, unsigned int revalidateIdx);

	SolverState	m_solverState;
	SolverStateAnalysis m_solverExtra;

	unsigned int m_maxNumberOfImages;
	unsigned int m_maxCorrPerImage;

	unsigned int m_maxNumDenseImPairs;

	int* d_variablesToCorrespondences;
	int* d_numEntriesPerRow;

	std::vector<float> m_convergence;
	std::vector<float> m_linConvergence;

	float m_verifyOptDistThresh;
	float m_verifyOptPercentThresh;

	bool		m_bRecordConvergence;
	CUDATimer *m_timer;

	SolverParameters m_defaultParams;
	float			 m_maxResidualThresh;
	std::shared_ptr<YAML::Node> yml;

#ifdef NEW_GUIDED_REMOVE
	std::vector<vec2ui> m_maxResImPairs;

	float4x4*	d_transforms;
#endif
};


