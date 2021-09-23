
#ifndef EXT_EIGEN_EIGENSOLVER_H_
#define EXT_EIGEN_EIGENSOLVER_H_

namespace ml
{


//linear system solver
template<class D> class LinearSolverEigen : public LinearSolver<D>
{
public:
	enum Method
	{
		LLT,
		LDLT,
		LU, //Inferior to LLT for symmetric problems
		QR, //Extremely slow
		ConjugateGradient_Diag,
		BiCGSTAB_Diag,
		BiCGSTAB_LUT,
		Profile,
	};

	LinearSolverEigen(Method method = ConjugateGradient_Diag, double tolerance = 1e-10)
	{
		m_method = method;
        m_tolerance = tolerance;
	}

	MathVector<D> solve(const SparseMatrix<D> &A, const MathVector<D> &b)
	{
		MLIB_ASSERT_STR(A.square() && b.size() == A.rows(), "invalid solve dimensions");
		Eigen::SparseMatrix<D> eigenMatrix;
        std::cout << "Making eigen matrix...";
		eigenutil::makeEigenMatrix(A, eigenMatrix);
        std::cout << "done" << std::endl;
		return solve(eigenMatrix, b);
	}

	MathVector<D> solve(const Eigen::SparseMatrix<D> &A, const MathVector<D> &b)
	{
		return solveUsingMethod(A, b, m_method);
	}

	MathVector<D> solveLeastSquares(const SparseMatrix<D> &A, const MathVector<D> &b)
	{
		Eigen::SparseMatrix<D> eigenMatrix;
		eigenutil::makeEigenMatrix(A, eigenMatrix);
        return solveLeastSquaresQR(eigenMatrix, b);
	}

    MathVector<D> solveLeastSquaresNormalEquations(const SparseMatrix<D> &A, const MathVector<D> &b)
    {
        Eigen::SparseMatrix<D> eigenMatrix;
        eigenutil::makeEigenMatrix(A, eigenMatrix);
        return solveUsingMethod(eigenMatrix.transpose() * eigenMatrix, A.transpose() * b, m_method);
    }

    MathVector<D> solveLeastSquaresManualCG(const SparseMatrix<D> &A, const MathVector<D> &bBase, UINT maxIterations, bool verbose = false)
    {
        SparseMatrix<D> ATranspose = A.transpose();
        const MathVector<D> &b = ATranspose * bBase;

        Eigen::SparseMatrix<D> eigenA, eigenAt;
        eigenutil::makeEigenMatrix(A, eigenA);
        eigenAt = eigenA.transpose();

        const UINT n = (UINT)b.size();

        MathVector<D> dInverse = A.selfTransposeDiagonal();
        auto invert = [](D& x) { x = (D)1.0 / x; };
        for_each(dInverse.begin(), dInverse.end(), invert);

        auto eigenMultiply = [&](const MathVector<D> &x) {
            Eigen::VectorXf temp = eigenAt * (eigenA * eigenutil::makeEigenVector(x));
            return eigenutil::dumpEigenVector(temp);
        };

        MathVector<D> x(n, 0.0);
        MathVector<D> r = b - eigenMultiply(x);
        MathVector<D> z = dInverse * r;
        MathVector<D> p = z;

        for (UINT iteration = 0; iteration < maxIterations; iteration++)
        {
            const D gamma = r | z;

            if (fabs(gamma) < 1e-20) break;
            
            //FloatType alpha = gamma / SparseMatrix<FloatType>::quadratic(A, p);
            const D alphaDenom = MathVector<D>::dot(p, eigenMultiply(p));
            
            std::cout << "alphaDenom: " << alphaDenom << std::endl;

            const D alpha = gamma / alphaDenom;

            x = x + alpha * p;
            r = r - alpha * eigenMultiply(p);

            if (*std::max_element(r.begin(), r.end()) <= m_tolerance && *std::min_element(r.begin(), r.end()) >= -m_tolerance)	break;

            z = dInverse * r;
            const D beta = (z | r) / gamma;
            p = z + beta * p;
        }
        return x;
    }

	MathVector<D> solveLeastSquaresQR(const Eigen::SparseMatrix<D> &A, const MathVector<D> &b)
	{
		std::cout << "Solving least-squares problem using QR" << std::endl;

        const Eigen::Matrix<D, Eigen::Dynamic, 1> bEigen = eigenutil::makeEigenVector(b);
        Eigen::SparseQR< Eigen::SparseMatrix<D>, Eigen::COLAMDOrdering<int> > factorization(A);
        //Eigen::SparseQR< Eigen::SparseMatrix<D>, Eigen::NaturalOrdering<int> > factorization(A);
		Eigen::Matrix<D, Eigen::Dynamic, 1> x = factorization.solve(bEigen);

		return eigenutil::dumpEigenVector(x);
	}

private:
	MathVector<D> solveUsingMethod(const Eigen::SparseMatrix<D> &A, const MathVector<D> &b, Method method)
	{
		ComponentTimer timer("Solving using method: " + getMethodName(method));
		
		const auto bEigen = eigenutil::makeEigenVector(b);
        Eigen::Matrix<D, Eigen::Dynamic, 1> x;

		if(method == LLT)
		{
			Eigen::SimplicialLLT< Eigen::SparseMatrix<D> > factorization(A);
			x = factorization.solve(bEigen);
		}
		else if(method == LDLT)
		{
			Eigen::SimplicialLDLT< Eigen::SparseMatrix<D> > factorization(A);
			x = factorization.solve(bEigen);
		}
		else if(method == LU)
		{
			Eigen::SparseLU< Eigen::SparseMatrix<D> > factorization(A);
			x = factorization.solve(bEigen);
		}
		else if(method == QR)
		{
			Eigen::SparseQR< Eigen::SparseMatrix<D>, Eigen::COLAMDOrdering<int> > factorization(A);
			x = factorization.solve(bEigen);
		}
		else if(method == ConjugateGradient_Diag)
		{
			Eigen::ConjugateGradient< Eigen::SparseMatrix<D>, Eigen::Lower, Eigen::DiagonalPreconditioner<D> > solver;
            solver.setTolerance((D)m_tolerance);
			solver.compute(A);
			x = solver.solve(bEigen);
			//Console::log("Iterations: " + std::string(solver.iterations()));
			//Console::log("Error: " + std::string(solver.error()));
		}
		else if(method == BiCGSTAB_Diag)
		{
			Eigen::BiCGSTAB< Eigen::SparseMatrix<D>, Eigen::DiagonalPreconditioner<D > > solver;
            solver.setTolerance((D)m_tolerance);
			solver.compute(A);
			x = solver.solve(bEigen);
			//Console::log("Iterations: " + std::string(solver.iterations()));
			//Console::log("Error: " + std::string(solver.error()));
		}
		else if(method == BiCGSTAB_LUT)
		{
			Eigen::BiCGSTAB< Eigen::SparseMatrix<D>, Eigen::IncompleteLUT<D > > solver;
            solver.setTolerance((D)m_tolerance);
			solver.compute(A);
			x = solver.solve(bEigen);
			//Console::log("Iterations: " + std::string(solver.iterations()));
			//Console::log("Error: " + std::string(solver.error()));
		}
		else if(method == Profile)
		{
			std::cout << "Profiling all eigen linear solvers" << std::endl;
			const int methodCount = (int)Profile;
			std::vector< MathVector<D> > results(methodCount);
			for(int methodIndex = 0; methodIndex < methodCount; methodIndex++)
			{
				results[methodIndex] = solveUsingMethod(A, b, (Method)methodIndex);
				if(methodIndex != 0)
				{
					double maxDeviation = 0.0;
					for(UINT variableIndex = 0; variableIndex < b.size(); variableIndex++)
						maxDeviation = std::max<double>(maxDeviation, fabs(results[methodIndex][variableIndex] - results[0][variableIndex]));
					std::cout << "Max deviation from LLT: " << std::to_string(maxDeviation) << std::endl;
				}
			}
			return results[0];
		}
		else
		{
			MLIB_ERROR("Unknown method");
		}

		return eigenutil::dumpEigenVector(x);
	}

	static std::string getMethodName(Method m)
	{
		switch(m)
		{
		case LLT: return "LLT";
		case LDLT: return "LDLT";
		case LU: return "LU";
		case QR: return "QR";
		case ConjugateGradient_Diag: return "ConjugateGradient_Diag";
		case BiCGSTAB_Diag: return "BiCGSTAB_Diag";
		case BiCGSTAB_LUT: return "BiCGSTAB_LUT";
		case Profile: return "Profile";
		default: return "Unknown";
		}
	}

	Method m_method;
    double m_tolerance;
};


//eigenvalue decomposition
template<class FloatType> class EigenSolverEigen : public EigenSolver < FloatType >
{
public:
	void eigenSystemInternal(const DenseMatrix<FloatType> &M, FloatType **eigenvectors, FloatType *eigenvalues) const {
		Eigen::MatrixXd mat;
		eigenutil::makeEigenMatrix(M, mat);
		Eigen::EigenSolver<Eigen::MatrixXd> eigenSolver(mat);
		
		for (unsigned int i = 0; i < M.rows(); i++) {
			eigenvalues[i] = (FloatType)eigenSolver.eigenvalues()[i].real();
			eigenSolver.eigenvalues();
			for (unsigned int j = 0; j < M.cols(); j++) {
				eigenvectors[i][j] = (FloatType)eigenSolver.eigenvectors()(i, j).real();
			}
		}
	}
};

typedef EigenSolverEigen<float> EigenSolverEigenf;
typedef EigenSolverEigen<double> EigenSolverEigend;


template<class FloatType> class EigenWrapper 
{
public:
	//! given a set of 3d correspondences determine a rotation and translation vector
	static Matrix4x4<FloatType> kabsch(const std::vector<vec3<FloatType>>& source, const std::vector<vec3<FloatType>>& target, vec3<FloatType>& eigenvalues, bool printDebug = false) {
		if (source.size() != target.size()) throw MLIB_EXCEPTION("invalid dimensions");
		if (source.size() < 3) throw MLIB_EXCEPTION("need at least 3 points");
		//{
		//	const auto& P = source;
		//	const auto& Q = target;

		//	//compute mean p0
		//	vec3<FloatType> p0(0, 0, 0);
		//	for (size_t i = 0; i < P.size(); i++) {
		//		p0 += P[i];
		//	}
		//	p0 /= (FloatType)P.size();

		//	//compute mean p1
		//	vec3<FloatType> q0(0, 0, 0);
		//	for (size_t i = 0; i < Q.size(); i++) {
		//		q0 += Q[i];
		//	}
		//	q0 /= (FloatType)Q.size();

		//	//compute covariance matrix
		//	Matrix3x3<FloatType> C;	C.setZero();
		//	for (size_t i = 0; i < source.size(); i++) {
		//		C = C + Matrix3x3<FloatType>::tensorProduct(P[i] - p0, Q[i] - q0);
		//	}

		//	//convert to eigen
		//	Eigen::Matrix3d Ce;
		//	for (unsigned int i = 0; i < 3; i++) {
		//		for (unsigned int j = 0; j < 3; j++) {
		//			Ce(i, j) = C(i, j);
		//		}
		//	}

		//	// SVD
		//	Eigen::JacobiSVD<Eigen::Matrix3d> svd(Ce, Eigen::ComputeThinU | Eigen::ComputeThinV);

		//	Eigen::Matrix3d V = svd.matrixU();
		//	Eigen::Vector3d S = svd.singularValues();
		//	Eigen::Matrix3d W = svd.matrixV();
		//	Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

		//	if ((V * W.transpose()).determinant() < 0)
		//		I(3 - 1, 3 - 1) = -1;

		//	// Recover the rotation and translation
		//	Eigen::Matrix3d resRot = W * I * V.transpose();
		//	Eigen::Vector3d resTrans = Eigen::Vector3d(q0.x, q0.y, q0.z) - resRot*Eigen::Vector3d(p0.x, p0.y, p0.z);

		//	Matrix4x4<FloatType> ret;
		//	for (unsigned int i = 0; i < 3; i++) {
		//		for (unsigned int j = 0; j < 3; j++) {
		//			ret(i, j) = (FloatType)resRot(i, j);
		//		}
		//	}
		//	ret(3, 0) = ret(3, 1) = ret(3, 2) = 0;	ret(3, 3) = 1;
		//	ret(0, 3) = (FloatType)resTrans(0);
		//	ret(1, 3) = (FloatType)resTrans(1);
		//	ret(2, 3) = (FloatType)resTrans(2);
		//	return ret;
		//}


		Eigen::MatrixXd P(3, source.size());
		for (size_t i = 0; i < source.size(); i++) {
			P(0, i) = source[i].x;
			P(1, i) = source[i].y;
			P(2, i) = source[i].z;
		}
		Eigen::MatrixXd Q(3, target.size());
		for (size_t i = 0; i < target.size(); i++) {
			Q(0, i) = target[i].x;
			Q(1, i) = target[i].y;
			Q(2, i) = target[i].z;
		}
		Eigen::VectorXd weights(source.size());
		for (unsigned int i = 0; i < weights.size(); i++) {
			weights[i] = 1.0;
		}
			 
		//if (P.cols() != Q.cols() || P.rows() != Q.rows())
		//	Helpers::ExitWithMessage("Helpers::Kabsch: P and Q have different dimensions");
		size_t D = P.rows(); // dimension of the space
		size_t N = P.cols(); // number of points
		Eigen::VectorXd	normalizedWeights = Eigen::VectorXd(weights.size());

		// normalize weights to sum to 1
		{
			double	sumWeights = 0;
			for (unsigned int i = 0; i < weights.size(); i++)
			{
				sumWeights += weights(i);
			}
			normalizedWeights = weights * (1.0 / sumWeights);
		}

		// Centroids
		Eigen::VectorXd	p0 = P * normalizedWeights;
		Eigen::VectorXd	q0 = Q * normalizedWeights;
		Eigen::VectorXd v1 = Eigen::VectorXd::Ones(N);


		Eigen::MatrixXd P_centred = P - p0*v1.transpose(); // translating P to center the origin
		Eigen::MatrixXd Q_centred = Q - q0*v1.transpose(); // translating Q to center the origin

		// Covariance between both matrices
		Eigen::MatrixXd C = P_centred * normalizedWeights.asDiagonal() * Q_centred.transpose();

		// SVD
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(C, Eigen::ComputeThinU | Eigen::ComputeThinV);

		Eigen::MatrixXd V = svd.matrixU();
		Eigen::VectorXd S = svd.singularValues();
		Eigen::MatrixXd W = svd.matrixV();
		Eigen::MatrixXd I = Eigen::MatrixXd::Identity(D, D);
		if (printDebug) {
			std::cout << "AtA:" << std::endl;
			for (unsigned int i = 0; i < C.rows(); i++) {
				for (unsigned int j = 0; j < C.cols(); j++)
					std::cout << C(i, j) << " ";
				std::cout << std::endl;
			}
			std::cout << "eigenvalues: " << S[0] << ", " << S[1] << ", " << S[2] << std::endl;
		}

		eigenvalues[0] = (FloatType)S[0];
		eigenvalues[1] = (FloatType)S[1];
		eigenvalues[2] = (FloatType)S[2];

		if ((V * W.transpose()).determinant() < 0)
			I(D - 1, D - 1) = -1;

		// Recover the rotation and translation
		Eigen::MatrixXd resRot = W * I * V.transpose();
		Eigen::VectorXd resTrans = q0 - resRot*p0;

		Matrix4x4<FloatType> ret;
		for (unsigned int i = 0; i < 3; i++) {
			for (unsigned int j = 0; j < 3; j++) {
				ret(i, j) = (FloatType)resRot(i, j);
			}
		}
		ret(3, 0) = ret(3, 1) = ret(3, 2) = 0;	ret(3, 3) = 1;
		ret(0, 3) = (FloatType)resTrans(0);
		ret(1, 3) = (FloatType)resTrans(1);
		ret(2, 3) = (FloatType)resTrans(2);
		return ret;
	}


	static FloatType reProjectionError(const std::vector < vec3<FloatType> >& source, const std::vector < vec3<FloatType> >& target, const Matrix4x4<FloatType>& trans) {
		if (source.size() != target.size()) throw MLIB_EXCEPTION("invalid dimension");

		FloatType res = 0;
		for (size_t i = 0; i < source.size(); i++) {
			FloatType distSq = vec3<FloatType>::distSq(trans * source[i], target[i]);
			res += distSq;
		}
		return res;
	}

	//! squared re-projection errors
	static void reProjectionErrors(const std::vector < vec3<FloatType> >& source, const std::vector < vec3<FloatType> >& target, const Matrix4x4<FloatType>& trans, std::vector<FloatType>& residuals) {
		if (source.size() != target.size()) throw MLIB_EXCEPTION("invalid dimension");

		residuals.resize(source.size(), 0);
		for (size_t i = 0; i < source.size(); i++) {
			residuals[i] = vec3<FloatType>::distSq(trans * source[i], target[i]);
		}
	}

	//! returns eigenvalues
	static vec3<FloatType> covarianceSVD(const std::vector<vec3<FloatType>>& points) {
		if (points.size() < 3) throw MLIB_EXCEPTION("need at least 3 points");
		vec3<FloatType> eigenvalues;

		Eigen::MatrixXd P(3, points.size());
		for (size_t i = 0; i < points.size(); i++) {
			P(0, i) = points[i].x;
			P(1, i) = points[i].y;
			P(2, i) = points[i].z;
		}
		Eigen::VectorXd weights(points.size());
		for (unsigned int i = 0; i < weights.size(); i++) {
			weights[i] = 1.0;
		}

		size_t D = P.rows(); // dimension of the space
		size_t N = P.cols(); // number of points
		Eigen::VectorXd	normalizedWeights = Eigen::VectorXd(weights.size());

		// normalize weights to sum to 1
		{
			double	sumWeights = 0;
			for (unsigned int i = 0; i < weights.size(); i++)
			{
				sumWeights += weights(i);
			}
			normalizedWeights = weights * (1.0 / sumWeights);
		}

		// Centroids
		Eigen::VectorXd	p0 = P * normalizedWeights;
		Eigen::VectorXd v1 = Eigen::VectorXd::Ones(N);
		Eigen::MatrixXd P_centred = P - p0*v1.transpose(); // translating P to center the origin

		// Covariance between both matrices
		Eigen::MatrixXd C = P_centred * normalizedWeights.asDiagonal() * P_centred.transpose();

		// SVD
		Eigen::JacobiSVD<Eigen::MatrixXd> svd(C, Eigen::ComputeThinU | Eigen::ComputeThinV);

		//Eigen::MatrixXd V = svd.matrixU();
		Eigen::VectorXd S = svd.singularValues();
		//Eigen::MatrixXd W = svd.matrixV();

		eigenvalues[0] = (FloatType)S[0];
		eigenvalues[1] = (FloatType)S[1];
		eigenvalues[2] = (FloatType)S[2];

		return eigenvalues;
	}
};
typedef EigenWrapper<float> EigenWrapperf;
typedef EigenWrapper<double> EigenWrapperd;

}  // namespace ml

#endif  // EXT_EIGEN_EIGENSOLVER_H_