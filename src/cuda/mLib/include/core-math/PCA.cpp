
template<class T>
void PCA<T>::init(const std::vector<const T*> &points, size_t dimension, const EigenSolverFunc &eigenSolver)
{
    //std::cout << "Initializing PCA, " << points.size() << " points, " << dimension << " dimensions" << std::endl;
	
    const size_t n = points.size();
    DenseMatrix<T> B(n, dimension);
	
    _means.clear();
    _means.resize(dimension, (T)0.0);

    for (const T *point : points)
		for(size_t dimIndex = 0; dimIndex < dimension; dimIndex++)
		    _means[dimIndex] += point[dimIndex];
	
    for(T &x : _means)
		x /= (T)n;	

	for(size_t pointIndex = 0; pointIndex < n; pointIndex++)
	{
		const T *point = points[pointIndex];
		for(size_t dimIndex = 0; dimIndex < dimension; dimIndex++)
		{
            B((UINT)pointIndex, (UINT)dimIndex) = point[dimIndex] - _means[dimIndex];
		}
	}

    //std::cout << "Building cross-correlation matrix..." << std::endl;
	
    DenseMatrix<T> C = B.getTranspose() * B;
	//DenseMatrix<T>::MultiplyMMTranspose(C, B);

    const T norm = T(1.0) / T(n);	//TODO (Matthias): I think this is wrong; I believe it  must be n-1
    for (auto &x : C)
        x *= norm;

    initFromCorrelationMatrix(C, eigenSolver);
}

template<class T>
void PCA<T>::init(DenseMatrix<T> &points, const EigenSolverFunc &eigenSolver)
{
    const size_t n = points.rows();
    const size_t dimension = points.cols();
    //std::cout << "Initializing PCA, " << n << " points, " << dimension << " dimensions" << std::endl;

    _means.clear();
    _means.resize(dimension, (T)0.0);

    for (size_t pointIndex = 0; pointIndex < n; pointIndex++)
        for (size_t dimIndex = 0; dimIndex < dimension; dimIndex++)
            _means[dimIndex] += points(pointIndex, dimIndex);

    for (T &x : _means)
        x /= (T)n;

    for (size_t pointIndex = 0; pointIndex < n; pointIndex++)
    {
        for (size_t dimIndex = 0; dimIndex < dimension; dimIndex++)
        {
            points(pointIndex, dimIndex) -= _means[dimIndex];
        }
    }

    //std::cout << "Building cross-correlation matrix..." << std::endl;

	//DenseMatrix<T> C = points * points.transpose();
	DenseMatrix<T> C = points.getTranspose() * points;
    //DenseMatrix<T>::MultiplyMMTranspose(C, B);

    const T norm = T(1.0) / T(n);
    for (auto &x : C)
        x *= norm;

    initFromCorrelationMatrix(C, eigenSolver);
}

template<class T>
void PCA<T>::initFromCorrelationMatrix(const DenseMatrix<T> &C, const EigenSolverFunc &eigenSolver)
{
    const size_t dimension = C.rows();
	//std::cout << "Computing eigensystem..." << std::endl;
	_system = eigenSolver(C);
    //_system = C.eigenSystem();
	//_system = C.eigenSystemUsingEigen();
    //std::cout << C.EigenTest(_system.eigenvalues, _system.eigenvectors) << std::endl;

    finalizeFromEigenSystem();
}

template<class T>
void PCA<T>::finalizeFromEigenSystem()
{
    const size_t dimension = _system.eigenvalues.size();
    double sum = 0.0;
	for(size_t dimIndex = 0; dimIndex < dimension; dimIndex++)
	{
        sum += std::max(T(0.0), _system.eigenvalues[dimIndex]);
	}
	double cumulativeEnergy = 0.0;
	for(size_t dimIndex = 0; dimIndex < dimension; dimIndex++)
	{
		cumulativeEnergy += _system.eigenvalues[dimIndex];
		//std::cout << "Energy at " << dimIndex + 1 << " terms: " << cumulativeEnergy / sum * 100.0f << "%" << std::endl;
	}
}

template<class T>
size_t PCA<T>::reducedDimension(double energyPercent) const
{
	double sum = 0.0;
	for(size_t dimIndex = 0; dimIndex < _system.eigenvalues.size(); dimIndex++)
	{
		sum += _system.eigenvalues[dimIndex];
	}
	double cumulativeEnergy = 0.0;
	for(size_t dimIndex = 0; dimIndex < _system.eigenvalues.size(); dimIndex++)
	{
		cumulativeEnergy += _system.eigenvalues[dimIndex];
		if(cumulativeEnergy / sum >= energyPercent)
		{
			return dimIndex + 1;
		}
	}
	return _system.eigenvalues.size();
}

template<class T>
void PCA<T>::transform(const std::vector<T> &input, size_t reducedDimension, std::vector<T> &result) const
{
    if(result.size() != reducedDimension)
	    result.resize(reducedDimension);
    transform(input.data(), reducedDimension, result.data());
}

template<class T>
void PCA<T>::inverseTransform(const std::vector<T> &input, std::vector<T> &result) const
{
    if (result.size() != _means.size())
	    result.resize(_means.size());
    inverseTransform(input.data(), input.size(), result.data());
}

template<class T>
void PCA<T>::transform(const T *input, size_t reducedDimension, T *result) const
{
	const size_t dimension = _means.size();
	const T *means = _means.data();
	const T *eigenVectorsPtr = _system.eigenvectors.getData();
	for(size_t row = 0; row < reducedDimension; row++)
    {
		T total = 0.0;
		const T *rowStart = eigenVectorsPtr + row * _system.eigenvectors.cols();
		for(size_t index = 0; index < dimension; index++)
		{
			total += rowStart[index] * (input[index] - means[index]);
		}
		result[row] = total;
    }
}

template<class T>
void PCA<T>::inverseTransform(const T *input, size_t reducedDimension, T *result) const
{
	size_t dimension = _means.size();
	for(size_t col = 0; col < dimension; col++)
    {
		T total = 0.0;
		for(size_t index = 0; index < reducedDimension; index++)
		{
			total += _system.eigenvectors(index, col) * input[index];
		}
		result[col] = total + _means[col];
    }
}

template<class T>
void PCA<T>::save(const std::string &filename) const
{
    BinaryDataStreamFile file(filename, true);
    file << _means << _system.eigenvalues << _system.eigenvectors;
    file.close();
}

template<class T>
void PCA<T>::load(const std::string &filename)
{
    BinaryDataStreamFile file(filename, false);
    file >> _means >> _system.eigenvalues >> _system.eigenvectors;
    file.close();
}
