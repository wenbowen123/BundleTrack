
#ifndef EXT_NEARESTNEIGHBORSEARCHFLANN_H_
#define EXT_NEARESTNEIGHBORSEARCHFLANN_H_

namespace ml
{

	template<class FloatType>
	class NearestNeighborSearchFLANN : public NearestNeighborSearch<FloatType>
	{
	public:
		// checkCount is the total number of entries that are checked each query (the higher the more accuracte).  Typically max(50, 4 * maxK) is good.
		// trees: number of parallel kd-trees to use
		NearestNeighborSearchFLANN(int checkCount, int trees)
		{
			m_trees = trees;
			m_checkCount = checkCount;
			m_flatPoints = nullptr;
			m_FLANNIndex = nullptr;
		}

		~NearestNeighborSearchFLANN()
		{
			if (m_FLANNIndex) {
 				delete[] m_queryStorage.ptr();
				delete[] m_indicesStorage.ptr();
				delete[] m_distsStorage.ptr();
				delete m_flatPoints;
				delete m_FLANNIndex;
				m_flatPoints = nullptr;
				m_FLANNIndex = nullptr;
			}
		}

		std::vector<FloatType> getDistances(unsigned int k) const
		{
			std::vector<FloatType> dists(k);
			for (unsigned int i = 0; i < k; i++) {
				dists[i] = (FloatType)sqrt(m_distsStorage[0][i]);	//TODO i believe this should be the sqrt
			}
			return dists;
		}

	private:
		void initInternal(const std::vector< const FloatType* > &points, UINT dimension, UINT maxK)
		{
			m_dimension = dimension;
			//std::cout << "Initializing FLANN index with " << points.size() << " points" << std::endl;
			
			// FLANN requires that all the points be flat. We could make a different init function
			// so that if you provide everything in a flat array we can use that instead without
			// duplicating the dataset.
			m_flatPoints = new FloatType[points.size() * dimension];
			for (size_t pointIndex = 0; pointIndex < points.size(); pointIndex++)
			{
				for (size_t dim = 0; dim < dimension; dim++)
				{
					m_flatPoints[pointIndex * dimension + dim] = points[pointIndex][dim];
				}
			}
				
			flann::Matrix<FloatType> dataset(m_flatPoints, points.size(), dimension);

			// TODO: FLANN can easily save/load its index, if creating the index is taking a long time.
			m_FLANNIndex = new flann::Index<flann::L2<FloatType> >(dataset, flann::KDTreeIndexParams((int)m_trees));
			m_FLANNIndex->buildIndex();

			m_queryStorage = flann::Matrix<float>(new float[dimension], 1, dimension);
			m_indicesStorage = flann::Matrix<int>(new int[maxK], 1, maxK);
			m_distsStorage = flann::Matrix<float>(new float[maxK], 1, maxK);

			//std::cout << "FLANN index created" << std::endl;
		}

		void kNearestInternal(const FloatType* query, UINT k, FloatType epsilon, std::vector<UINT> &result) const
		{
			memcpy(m_queryStorage.ptr(), query, m_dimension * sizeof(FloatType));
			int res = m_FLANNIndex->knnSearch(m_queryStorage, m_indicesStorage, m_distsStorage, k, flann::SearchParams((int)m_checkCount));

			result.resize(res);
			for (size_t i = 0; i < result.size(); i++) {
				result[i] = m_indicesStorage[0][i];
			}
		}

		void fixedRadiusInternal(const FloatType* query, UINT k, FloatType radius, FloatType epsilon, std::vector<UINT> &result) const
		{
			memcpy(m_queryStorage.ptr(), query, m_dimension * sizeof(FloatType));
			int res = m_FLANNIndex->radiusSearch(m_queryStorage, m_indicesStorage, m_distsStorage, radius*radius, flann::SearchParams((int)m_checkCount));

			result.resize(res);
			for (size_t i = 0; i < result.size(); i++) {
				result[i] = m_indicesStorage[0][i];
			} 
		}

		void fixedRadiusInternalDist(const FloatType* query, UINT k, FloatType radius, FloatType epsilon, std::vector< std::pair<UINT, FloatType> > &result) const
		{
			memcpy(m_queryStorage.ptr(), query, m_dimension * sizeof(FloatType));
			int res = m_FLANNIndex->radiusSearch(m_queryStorage, m_indicesStorage, m_distsStorage, radius*radius, flann::SearchParams((int)m_checkCount));

			result.resize(res);
			for (size_t i = 0; i < result.size(); i++) {
				result[i].first = m_indicesStorage[0][i];
				result[i].second = m_distsStorage[0][i];
				result[i].second = sqrt(result[i].second);
			}
		}

		//TODO: abstract over different search metrics
		flann::Index< flann::L2<float> > *m_FLANNIndex;
		float *m_flatPoints;
		mutable flann::Matrix<FloatType> m_queryStorage;
		mutable flann::Matrix<int> m_indicesStorage;
		mutable flann::Matrix<FloatType> m_distsStorage;
		size_t m_dimension, m_checkCount, m_trees;
	};

	typedef NearestNeighborSearchFLANN<float> NearestNeighborSearchFLANNf;
	typedef NearestNeighborSearchFLANN<double> NearestNeighborSearchFLANNd;

}  // namespace ml

#endif  // EXT_NEARESTNEIGHBORSEARCHFLANN_H_
