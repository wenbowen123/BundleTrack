
#ifndef CORE_MATH_POINT1D_H_
#define CORE_MATH_POINT1D_H_

#include <iostream>
#include <cmath>
#include <iostream>
#include <cassert>
#include "core-util/binaryDataSerialize.h"

namespace ml {

//! 1D vector (I know it's a joke, but we need it for compatibility reasons)
template <class T>
class vec1 : public BinaryDataSerialize< vec1<T> >
{
public:
	vec1(T v) {
		array[0] = v;
	}

	vec1() {
		array[0] = 0;
	}

	template <class U>
	vec1(const vec1<U>& other) {
		array[0] = (T)other.array[0];
	}

	vec1(const vec1& other) {
		array[0] = other.array[0];
	}

	inline const vec1<T>& operator=(const vec1& other) {
		array[0] = other.array[0];
		return *this;
	}

	inline vec1<T> operator-() const {
		return vec1<T>(-array[0]);
	}

	inline vec1<T> operator+(const vec1& other) const {
		return vec1<T>(array[0]+other.array[0]);
	}

	inline vec1<T> operator+(T val) const {
		return vec1<T>(array[0]+val);
	}

	inline void operator+=(const vec1& other) {
		array[0] += other.array[0];
	}

	inline void operator-=(const vec1& other) {
		array[0] -= other.array[0];
	}

	inline void operator+=(T val) {
		array[0] += val;
	}

	inline void operator-=(T val) {
		array[0] -= val;
	}

	inline void operator*=(T val) {
		array[0] *= val;
	}

	inline void operator/=(T val) {
		array[0] /= val;
	}

	inline vec1<T> operator*(T val) const {
		return vec1<T>(array[0]*val);
	}

	inline vec1<T> operator/(T val) const {
		return vec1<T>(array[0]/val);
	}

	inline vec1<T> operator-(const vec1& other) const {
		return vec1<T>(array[0]-other.array[0]);
	}

	inline vec1<T> operator-(T val) const {
		return vec1<T>(array[0]-val);
	}

	inline bool operator==(const vec1& other) const {
		if ((array[0] == other.array[0]))
			return true;

		return false;
	}

	inline bool operator!=(const vec1& other) const {
		return !(*this == other);
	}


	//! dot product
	inline T operator|(const vec1& other) const {
		return (array[0]*other.array[0]);
	}

	inline T& operator[](unsigned int i) {
		assert(i < 1);
		return array[i];
	}

	inline const T& operator[](unsigned int i) const {
		assert(i < 1);
		return array[i];
	}

	inline T lengthSq() const {
		return (array[0]*array[0]);
	}

	inline T length() const {
		return array[0];
	}

	static T distSq(const vec1& v0, const vec1& v1) {
		return (v0.array[0] - v1.array[1])*(v0.array[0] - v1.array[1]);
	}

	static T dist(const vec1& v0, const vec1& v1) {
		return std::abs(v0.array[0] - v1.array[1]);
	}

	static vec1<T> randomUniform(T min, T max) {
		return vec1<T>(math::randomUniform(min, max));
	}

	inline vec1 getNormalized() const {
		return vec1<T>();
	}

	inline void normalize() const {
		array[0] /= length();
	}

	inline void print() const {
		std::cout << "(" << array[0] << ")" << std::endl;
	}

    inline bool isValid() const {
        return (x == x);
    }

	inline T* getData() {
		return &array[0];
	}

	inline std::vector<T> toStdVector() const {
		std::vector<T> result(1);
		result[0] = x;
		return result;
	}

	inline std::string toString() const {
		return std::to_string(x);
	}

	static const vec1<T> origin;
	static const vec1<T> eX;
	static const vec1<T> eY;

	union {
		struct {
			T x;        // standard names for components
		};
		struct {
			T r;		// colors
		};
		T array[1];     // array access
	};
};

//! operator for scalar * vector
template <class T>
inline vec1<T> operator*(T s, const vec1<T>& v)
{
	return v * s;
}
template <class T>
inline vec1<T> operator/(T s, const vec1<T>& v)
{
	return vec1<T>(s/v.x);
}
template <class T>
inline vec1<T> operator+(T s, const vec1<T>& v)
{
	return v + s;
}
template <class T>
inline vec1<T> operator-(T s, const vec1<T>& v)
{
	return vec1<T>(s-v.x);
}

namespace math {
	template<class T>
	inline vec1<int> sign(const vec1<T>& v) {
		return vec1<int>(sign(v.x));
	}
}


//! write a vec1 to a stream
template <class T> inline std::ostream& operator<<(std::ostream& s, const vec1<T>& v)
{ return (s << v[0]);}

//! read a vec1 from a stream
template <class T> inline std::istream& operator>>(std::istream& s, vec1<T>& v)
{ return (s >> v[0]); }


typedef vec1<double> vec1d;
typedef vec1<float> vec1f;
typedef vec1<int> vec1i;
typedef vec1<short> vec1s;
typedef vec1<short> vec1us;
typedef vec1<unsigned int> vec1ui;
typedef vec1<unsigned char> vec1uc;
typedef vec1<UINT64> vec1ul;
typedef vec1<INT64> vec1l;

}  // namespace ml

#endif  // CORE_MATH_POINT2D_H_
