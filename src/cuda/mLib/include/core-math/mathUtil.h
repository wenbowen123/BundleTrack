
#ifndef CORE_MATH_MATHUTIL_H_
#define CORE_MATH_MATHUTIL_H_

namespace ml {
namespace math {

template<class T>
Matrix3x3<T> covarianceMatrix(const std::vector< vec3<T> >& points) 
{
	auto mean = std::accumulate(points.begin(), points.end(), vec3<T>::origin) / (T)points.size();

	Matrix3x3<T> covariance;
	covariance.setZero();

	for (const auto& p : points)
	{
		auto recenteredPt = p - mean;
		auto tensor = Matrix3x3<T>::tensorProduct(recenteredPt, recenteredPt);
		for (int y = 0; y < 3; y++)
			for (int x = 0; x < 3; x++)
				covariance(y, x) += tensor(y, x);
	}

	covariance /= (T)(points.size() - 1);

	return covariance;
}

template<class T>
Matrix2x2<T> covarianceMatrix(const std::vector< vec2<T> >& points)
{
	auto mean = std::accumulate(points.begin(), points.end(), vec2<T>::origin) / (T)points.size();

	Matrix2x2<T> covariance;
	covariance.setZero();

	for (const auto& p : points)
	{
		auto recenteredPt = p - mean;
		auto tensor = Matrix2x2<T>::tensorProduct(recenteredPt, recenteredPt);
		for (int y = 0; y < 2; y++)
			for (int x = 0; x < 2; x++)
				covariance(y, x) += tensor(y, x);
	}

	covariance /= (T)(points.size() - 1);

	return covariance;
}


//
// returns the <axis, eigenvalue> pairs for the PCA of the given 3D points.
//
template <class T>
vector< std::pair<vec3<T>, T> > pointSetPCA(const std::vector< vec3<T> > &points)
{
	auto system = covarianceMatrix(points).eigenSystem();
    const auto &v = system.eigenvectors;
    
    vector< std::pair<vec3<T>, T> > result;
    result.push_back(std::make_pair(vec3<T>(v(0, 0), v(0, 1), v(0, 2)), system.eigenvalues[0]));
    result.push_back(std::make_pair(vec3<T>(v(1, 0), v(1, 1), v(1, 2)), system.eigenvalues[1]));
    result.push_back(std::make_pair(vec3<T>(v(2, 0), v(2, 1), v(2, 2)), system.eigenvalues[2]));
    return result;
}

//
// returns the <axis, eigenvalue> pairs for the PCA of the given 2D points.
//
template <class T>
vector< std::pair<vec2<T>, T> > pointSetPCA(const std::vector< vec2<T> > &points)
{
	auto system = covarianceMatrix(points).eigenSystem();
    const auto &v = system.eigenvectors;

    vector< std::pair<vec2<T>, T> > result;
    result.push_back(std::make_pair(vec2<T>(v(0, 0), v(1, 0)), system.eigenvalues[0]));
    result.push_back(std::make_pair(vec2<T>(v(0, 1), v(1, 1)), system.eigenvalues[1]));
    return result;
}


}
}  // namespace ml

#endif  // CORE_MATH_SPARSEMATRIX_H_
