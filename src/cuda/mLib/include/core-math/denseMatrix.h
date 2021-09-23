
#ifndef CORE_MATH_DENSEMATRIX_H_
#define CORE_MATH_DENSEMATRIX_H_

#include "mathVector.h"
#include "sparseMatrix.h"
#include "denseMatrix.h"
#include "matrix4x4.h"
#include "matrix3x3.h"
#include "matrix2x2.h"



namespace ml {

template<class T>
class Matrix4x4;

template<class T>
class Matrix3x3;

template<class T>
class Matrix2x2;

template <class T> class DenseMatrix
{
public:
	DenseMatrix()
	{
		m_rows = 0;
		m_cols = 0;
	}

	DenseMatrix(const DenseMatrix<T>& s)
	{
		m_rows = s.m_rows;
		m_cols = s.m_cols;
		m_data = s.m_data;
	}

    DenseMatrix(DenseMatrix<T> &&s)
	{
		m_rows = s.m_rows;
		m_cols = s.m_cols;
		s.m_rows = 0;
		s.m_cols = 0;
		m_data = std::move(s.m_data);
	}

    explicit DenseMatrix(size_t squareDimension)
	{
		m_rows = (UINT)squareDimension;
        m_cols = (UINT)squareDimension;
		m_data.resize(m_rows * m_cols);
	}

	explicit DenseMatrix(const MathVector<T> &diagonal)
	{
		m_rows = diagonal.size();
		m_cols = diagonal.size();
		m_data.resize(m_rows * m_cols);
		for(UINT row = 0; row < m_rows; row++)
		{
			for(UINT col = 0; col < m_cols; col++)
				(*this)(row, col) = 0.0;
			(*this)(row, row) = diagonal[row];
		}

	}

    DenseMatrix(size_t rows, size_t cols, T clearValue = (T)0.0)
	{
		m_rows = rows;
        m_cols = cols;
        m_data.resize(m_rows * m_cols, clearValue);
	}

	DenseMatrix(size_t rows, size_t cols, const T* values) {
		m_rows = rows;
		m_cols = cols;
		m_data.resize(m_rows * m_cols);

		for (size_t i = 0; i < rows*cols; i++) {
			m_data[i] = values[i];
		}
	}

	DenseMatrix(const std::string &s, MatrixStringFormat format)
	{
		if(format == MatrixStringFormatMathematica)
		{
			//
			// this is really a dense format and should be loaded as such, then cast into a SparseMatrix
			//
			std::vector<std::string> data = ml::util::split(s,"},{");
			m_rows = data.size();
			m_cols = util::split(data[0], ",").size();
			m_data.resize(m_rows * m_cols);

			for (size_t row = 0; row < m_rows; row++) {
				std::vector<std::string> values = ml::util::split(data[row], ",");
				for (size_t col = 0; col < values.size(); col++) {
					const std::string s = ml::util::replace(ml::util::replace(values[col], "{",""), "}","");
					(*this)(row, col) = (T)std::stod(s);
				}
			}
		}
		else
		{
			MLIB_ERROR("invalid matrix string format");
		}
	}

    DenseMatrix(const Matrix4x4<T> &m)
    {
        m_rows = 4;
        m_cols = 4;
        m_data.resize(16);
        for (unsigned int element = 0; element < m_data.size(); element++)
            m_data[element] = m[element];
    }

	DenseMatrix(const Matrix3x3<T> &m)
	{
		m_rows = 3;
		m_cols = 3;
		m_data.resize(9);
		for (unsigned int element = 0; element < m_data.size(); element++)
			m_data[element] = m[element];
	}

	DenseMatrix(const Matrix2x2<T> &m)
	{
		m_rows = 2;
		m_cols = 2;
		m_data.resize(4);
		for (unsigned int element = 0; element < m_data.size(); element++)
			m_data[element] = m[element];
	}

    void resize(size_t rows, size_t cols, T clearValue = (T)0.0)
	{
		m_rows = rows;
		m_cols = cols;
		m_data.resize(m_rows * m_cols, clearValue);
	}


	void operator=(const DenseMatrix<T>& s)
	{
		m_rows = s.m_rows;
		m_cols = s.m_cols;
		m_data = s.m_data;
	}

	void operator=(DenseMatrix<T>&& s)
	{
		m_rows = s.m_rows;
		m_cols = s.m_cols;
		s.m_rows = 0;
		s.m_cols = 0;
		m_data = std::move(s.m_data);
	}

	//
	// Accessors
	//
    T& operator()(size_t row, size_t col)
	{
		MLIB_ASSERT(row < m_rows && col < m_cols);
		return m_data[row * m_cols + col];
	}

    T operator()(size_t row, size_t col) const
	{
		MLIB_ASSERT(row < m_rows && col < m_cols);
		return m_data[row * m_cols + col];
	}

    size_t rows() const
	{
		return m_rows;
	}

    size_t cols() const
	{
		return m_cols;
	}

	bool square() const
	{
		return (m_rows == m_cols);
	}

	//! Access i-th element of the Matrix for constant access
	inline T operator[] (unsigned int i) const {
		MLIB_ASSERT(i < m_cols*m_rows);
		return m_data[i];
	}

	//! Access i-th element of the Matrix
	inline  T& operator[] (unsigned int i) {
		assert(i < m_cols*m_rows);
		return m_data[i];
	}

	std::vector<T> diagonal() const
	{
		MLIB_ASSERT_STR(square(), "diagonal called on non-square matrix");
		std::vector<T> result(m_rows);
		for(UINT row = 0; row < m_rows; row++)
			result[row] = m_data[row * m_cols + row];
		return result;
	}

    const T* getData() const
    {
        return &m_data[0];
    }


	//
	// math functions
	//
	DenseMatrix<T> getTranspose() const;
	T maxMagnitude() const;
	DenseMatrix<T> inverse();
	void invertInPlace();
	bool valid() const;

	//
	// overloaded operator helpers
	//
	static DenseMatrix<T> add(const DenseMatrix<T> &A, const DenseMatrix<T> &B);
	static DenseMatrix<T> subtract(const DenseMatrix<T> &A, const DenseMatrix<T> &B);
	static DenseMatrix<T> multiply(const DenseMatrix<T> &A, T c);
	static std::vector<T> multiply(const DenseMatrix<T> &A, const std::vector<T> &v);
	static DenseMatrix<T> multiply(const DenseMatrix<T> &A, const DenseMatrix<T> &B);

	//
	// common matrices
	//
	static DenseMatrix<T> identity(int n)
	{
		return DenseMatrix<T>(MathVector<T>(n, (T)1.0));
	}

	unsigned int rank(T eps = (T)0.00001) const {
		if (!square())	throw MLIB_EXCEPTION("");
		return util::rank<DenseMatrix<T>, T>(*this, m_rows, eps);
	}

	// checks whether the matrix is symmetric
	bool isSymmetric(T eps = (T)0.00001) const {
		if (!square())	return false;
		for (unsigned int i = 1; i < m_rows; i++) {
			for (unsigned int j = 0; j < m_cols/2; j++) {
				if (!math::floatEqual((*this)(i, j), (*this)(j, i), eps)) return false;
			}
		}
		return true;
	}

	EigenSystem<T> eigenSystem() const {
		return EigenSolver<T>::solve<EigenSolver<T>::TYPE_DEFAULT>(*this);
	}

	// TODO: figure out a better way to do this.
	EigenSystem<T> eigenSystemUsingEigen() const {
		return EigenSolver<T>::solve<EigenSolver<T>::TYPE_EIGEN>(*this);
	}

    //
    // in-place operators
    //
    void operator /= (T x)
    {
        T rcp = (T)1.0 / x;
        for (T &e : m_data)
            e *= rcp;
    }

    T* begin()
    {
        return &m_data[0];
    }
    T* end()
    {
        return begin() + (m_rows * m_cols);
    }

	inline operator Matrix4x4<T>(){
		MLIB_ASSERT(m_rows == 4 && m_cols == 4);
		Matrix4x4<T> res;
		for (unsigned int i = 0; i < 12; i++) {
			res[i] = m_data[i];
		}
		return res;
	}
	inline operator Matrix3x3<T>(){
		MLIB_ASSERT(m_rows == 3 && m_cols == 3);
		Matrix3x3<T> res;
		for (unsigned int i = 0; i < 9; i++) {
			res[i] = m_data[i];
		}
		return res;
	}
	inline operator Matrix2x2<T>(){
		MLIB_ASSERT(m_rows == 2 && m_cols == 2);
		Matrix2x2<T> res;
		for (unsigned int i = 0; i < 4; i++) {
			res[i] = m_data[i];
		}
		return res;
	}

private:
	size_t m_rows, m_cols;
    std::vector< T > m_data;
};

template<class T>
DenseMatrix<T> operator + (const DenseMatrix<T> &A, const DenseMatrix<T> &B)
{
	return DenseMatrix<T>::add(A, B);
}

template<class T>
DenseMatrix<T> operator - (const DenseMatrix<T> &A, const DenseMatrix<T> &B)
{
	return DenseMatrix<T>::subtract(A, B);
}

template<class T>
DenseMatrix<T> operator * (const DenseMatrix<T> &A, const DenseMatrix<T> &B)
{
	return DenseMatrix<T>::multiply(A, B);
}

template<class T>
MathVector<T> operator * (const DenseMatrix<T> &A, const MathVector<T> &B)
{
	return DenseMatrix<T>::multiply(A, B);
}

template<class T>
DenseMatrix<T> operator * (const DenseMatrix<T> &A, T val)
{
	return DenseMatrix<T>::multiply(A, val);
}

//! writes to a stream
template <class T>
inline std::ostream& operator<<(std::ostream& s, const DenseMatrix<T>& m)
{
	for (unsigned int i = 0; i < m.rows(); i++) {
		for (unsigned int j = 0; j < m.cols(); j++) {
			s << m(i,j) << " ";
		}
		std::cout << std::endl;
	}
	return s;
}

template<class BinaryDataBuffer, class BinaryDataCompressor, class T>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator << (BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const DenseMatrix<T>& m) {
	const UINT rows = m.rows();
	const UINT cols = m.cols();
	s << rows << cols;
	s.writeData((const BYTE *)m.getData(), sizeof(T) * rows * cols);
	return s;
}

template<class BinaryDataBuffer, class BinaryDataCompressor, class T>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator >> (BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, DenseMatrix<T>& m) {
	UINT rows, cols;
	s >> rows >> cols;
	m.resize(rows, cols);
	s.readData((BYTE *)m.getData(), sizeof(T) * rows * cols);
	return s;
}

typedef DenseMatrix<float> DenseMatrixf;
typedef DenseMatrix<double> DenseMatrixd;

}  // namespace ml

#include "denseMatrix.cpp"

#endif  // CORE_MATH_DENSEMATRIX_H_