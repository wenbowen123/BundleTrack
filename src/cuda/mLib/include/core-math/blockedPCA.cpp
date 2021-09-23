#include <string>

template<class T>
void BlockedPCA<T>::init(const DenseMatrix<T> &points, size_t subsetCount, const EigenSolverFunc &eigenSolver)
{
	const size_t n = points.rows();
	const size_t dimension = points.cols();
    std::cout << "Initializing blocked PCA, " << n << " points, " << dimension << " total dims, " << subsetCount << " subsets" << std::endl;
	
	_subsets.resize(subsetCount);

	int allocatedDims = 0;
	int subsetIndex = 0;
	while (allocatedDims < dimension)
	{
		_subsets[subsetIndex].dimCount++;
		allocatedDims++;
		subsetIndex = (subsetIndex + 1) % subsetCount;
	}

	_totalDimensions = 0;
	for (int i = 0; i < subsetCount; i++)
	{
		_subsets[i].startDim = _totalDimensions;
		_totalDimensions += _subsets[i].dimCount;
	}

	for (Subset &s : _subsets)
	{
		DenseMatrix<T> subpoints(n, s.dimCount);
		for (int p = 0; p < n; p++)
			for (int d = 0; d < s.dimCount; d++)
				subpoints(p, d) = points(p, d + s.startDim);
		s.pca.init(subpoints, eigenSolver);
	}
}

template<class T>
void BlockedPCA<T>::transform(const std::vector<T> &input, size_t reducedTotalDimension, std::vector<T> &result) const
{
	if (result.size() != reducedTotalDimension)
		result.resize(reducedTotalDimension);
	transform(input.data(), reducedTotalDimension, result.data());
}

/*template<class T>
void BlockedPCA<T>::inverseTransform(const std::vector<T> &input, std::vector<T> &result) const
{
	if (result.size() != _totalDimensions)
		result.resize(_totalDimensions);
	inverseTransform(input.data(), input.size() / _subsets.size() + 1, result.data());
}*/

template<class T>
void BlockedPCA<T>::transform(const T *input, size_t reducedTotalDimension, T *result) const
{
	const size_t subsetCount = _subsets.size();
	const int reducedSubsetDimension = reducedTotalDimension / subsetCount;
	for (int i = 0; i < reducedTotalDimension; i++)
		result[i] = 0;

	for (int subsetIndex = 0; subsetIndex < subsetCount; subsetIndex++)
	{
		const Subset &s = _subsets[subsetIndex];
		
		int reducedDim = std::min((int)reducedSubsetDimension, s.dimCount);
		if (subsetIndex * reducedSubsetDimension + reducedDim >= reducedTotalDimension)
			reducedDim = reducedTotalDimension - subsetIndex * reducedSubsetDimension;

		s.pca.transform(input + s.startDim, reducedDim, result + subsetIndex * reducedSubsetDimension);
	}
}

/*template<class T>
void BlockedPCA<T>::inverseTransform(const T *input, size_t reducedSubsetDimension, T *result) const
{
	const size_t subsetCount = _subsets.size();
	for (int i = 0; i < reducedSubsetDimension * subsetCount; i++)
		result[i] = 0;

	for (int subsetIndex = 0; subsetIndex < subsetCount; subsetIndex++)
	{
		const Subset &s = _subsets[subsetIndex];
		const int reducedDim = std::min((int)reducedSubsetDimension, s.dimCount);
		s.pca.transform(input + s.startDim, reducedDim, result + subsetIndex * reducedSubsetDimension);
	}
}*/

template<class T>
void BlockedPCA<T>::save(const std::string &baseFilename) const
{
    BinaryDataStreamFile file(baseFilename + ".dat", true);
	size_t subsetCount = _subsets.size();
    file << subsetCount << _totalDimensions;
	for (int i = 0; i < subsetCount; i++)
	{
		file << _subsets[i].startDim << _subsets[i].dimCount;
		_subsets[i].pca.save(baseFilename + "_" + std::to_string(i) + ".dat");
	}
    file.close();


}

template<class T>
void BlockedPCA<T>::load(const std::string &baseFilename)
{
    BinaryDataStreamFile file(baseFilename + ".dat", false);
	size_t subsetCount;
	file >> subsetCount >> _totalDimensions;
	_subsets.resize(subsetCount);
	for (int i = 0; i < subsetCount; i++)
	{
		file >> _subsets[i].startDim >> _subsets[i].dimCount;
		_subsets[i].pca.load(baseFilename + "_" + std::to_string(i) + ".dat");
	}
    //file.closeStream();
}
