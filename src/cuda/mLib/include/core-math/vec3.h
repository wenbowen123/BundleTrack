
#ifndef CORE_MATH_POINT3D_H_
#define CORE_MATH_POINT3D_H_


#include <iostream>
#include <cmath>

#include "vec2.h"

namespace ml {

//! 3D vector.
template <class T>
class vec3 : public BinaryDataSerialize< vec3<T> >
{
public:
	explicit vec3(T v) {
		array[0] = array[1] = array[2] = v;
	}

	vec3() {
		array[0] = array[1] = array[2] = 0;
	}

	vec3(T x, T y, T z) {
		array[0] = x;
		array[1] = y;
		array[2] = z;
	}

	template <class U>
	vec3(const vec3<U>& other) {
		array[0] = (T)other.array[0];
		array[1] = (T)other.array[1];
		array[2] = (T)other.array[2];
	}

    explicit vec3(const vec2<T>& other, T z) {
		array[0] = other.array[0];
		array[1] = other.array[1];
		array[2] = z;
	}

	vec3(const vec3& other) {
		array[0] = other.array[0];
		array[1] = other.array[1];
		array[2] = other.array[2];
	}

	vec3(const T* other) {
		array[0] = other[0];
		array[1] = other[1];
		array[2] = other[2];
	}

	inline const vec3<T>& operator=(const vec3& other) {
		array[0] = other.array[0];
		array[1] = other.array[1];
		array[2] = other.array[2];
		return *this;
	}

	inline const vec3<T>& operator=(T other) {
		array[0] = other;
		array[1] = other;
		array[2] = other;
		return *this;
	}

	inline vec3<T> operator-() const {
		return vec3<T>(-array[0], -array[1], -array[2]);
	}

	inline vec3<T> operator+(const vec3& other) const {
		return vec3<T>(array[0]+other.array[0], array[1]+other.array[1], array[2]+other.array[2]);
	}

	inline vec3<T> operator+(T val) const {
		return vec3<T>(array[0]+val, array[1]+val, array[2]+val);
	}

	inline void operator+=(const vec3& other) {
		array[0] += other.array[0];
		array[1] += other.array[1];
		array[2] += other.array[2];
	}

	inline void operator-=(const vec3& other) {
		array[0] -= other.array[0];
		array[1] -= other.array[1];
		array[2] -= other.array[2];
	}

	inline void operator+=(T val) {
		array[0] += val;
		array[1] += val;
		array[2] += val;
	}

	inline void operator-=(T val) {
		array[0] -= val;
		array[1] -= val;
		array[2] -= val;
	}

	inline void operator*=(T val) {
		array[0] *= val;
		array[1] *= val;
		array[2] *= val;
	}

	inline bool isValid() const
	{
		return (x == x && y == y && z == z);
	}

	inline bool isFinite() const
	{
		return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
	}

	inline void operator/=(T val) {
		//optimized version for float/double (doesn't work for int) -- assumes compiler statically optimizes if
		if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
			T inv = (T)1 / val;
			array[0] *= inv;
			array[1] *= inv;
			array[2] *= inv;
		}
		else {
			array[0] /= val;
			array[1] /= val;
			array[2] /= val;
		}
	}

	inline vec3<T> operator*(T val) const {
		return vec3<T>(array[0]*val, array[1]*val, array[2]*val);
	}

	inline vec3<T> operator/(T val) const {
		//optimized version for float/double (doesn't work for int) -- assumes compiler statically optimizes if
		if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
			T inv = (T)1 / val;
			return vec3<T>(array[0] * inv, array[1] * inv, array[2] * inv);
		}
		else {
			return vec3<T>(array[0] / val, array[1] / val, array[2] / val);
		}
	}

	//! Cross product
	inline vec3<T> operator^(const vec3& other) const {
		return vec3<T>(array[1]*other.array[2] - array[2]*other.array[1], array[2]*other.array[0] - array[0]*other.array[2], array[0]*other.array[1] - array[1]*other.array[0]);
	}

	//! Dot product
	inline T operator|(const vec3& other) const {
		return (array[0]*other.array[0] + array[1]*other.array[1] + array[2]*other.array[2]);
	}

	static inline T dot(const vec3& l, const vec3& r) {
		return(l.array[0] * r.array[0] + l.array[1] * r.array[1] + l.array[2] * r.array[2]);
	}

    static inline vec3 cross(const vec3& l, const vec3& r) {
        return (l ^ r);
    }

	inline vec3<T> operator-(const vec3& other) const {
		return vec3<T>(array[0]-other.array[0], array[1]-other.array[1], array[2]-other.array[2]);
	}

	inline vec3<T> operator-(T val) const {
		return vec3<T>(array[0]-val, array[1]-val, array[2]-val);
	}

	inline bool operator==(const vec3& other) const {
		if ((array[0] == other.array[0]) && (array[1] == other.array[1]) && (array[2] == other.array[2]))
			return true;

		return false;
	}

	inline bool operator!=(const vec3& other) const {
		return !(*this == other);
	}


	inline T lengthSq() const {
		return (array[0]*array[0] + array[1]*array[1] + array[2]*array[2]);
	}

	inline T length() const {
		return sqrt(lengthSq());
	}

	static T distSq(const vec3& v0, const vec3& v1) {
		return ((v0.array[0]-v1.array[0])*(v0.array[0]-v1.array[0]) + (v0.array[1]-v1.array[1])*(v0.array[1]-v1.array[1]) + (v0.array[2]-v1.array[2])*(v0.array[2]-v1.array[2]));
	}

	static T dist(const vec3& v0, const vec3& v1) {
		return sqrt((v0.array[0]-v1.array[0])*(v0.array[0]-v1.array[0]) + (v0.array[1]-v1.array[1])*(v0.array[1]-v1.array[1]) + (v0.array[2]-v1.array[2])*(v0.array[2]-v1.array[2]));
	}

	static vec3<T> randomUniform(T min, T max) {
		return vec3<T>(math::randomUniform(min, max),
			math::randomUniform(min, max),
			math::randomUniform(min, max));
	}

	inline operator T*() {
		return array;
	}
	
	inline operator const T*() const {
		return array;
	}

	inline void print() const {
		std::cout << "(" << array[0] << " / " << array[1] << " / " << array[2] << ")" << std::endl;
	}

	const T& operator[](int i) const {
		assert(i < 3);
		return array[i];
	}

	T& operator[](int i) {
		assert(i < 3);
		return array[i];
	}

	static inline vec3<T> normalize(const vec3<T> &v) {
		return v.getNormalized();
	}
	inline void normalize() {
		T val = (T)1.0 / length();
		array[0] *= val;
		array[1] *= val;
		array[2] *= val;
	}

  //! If this vec3 is non-zero, then normalize it, else return
	inline void normalizeIfNonzero() {
		const T l = length();
    if (l == static_cast<T>(0)) { return; }  // TODO: Better to check against epsilon tolerance
    const T val = static_cast<T>(1) / l;
		array[0] *= val;
		array[1] *= val;
		array[2] *= val;
	}

	inline vec3<T> getNormalized() const {
		T val = (T)1.0 / length();
		return vec3<T>(array[0] * val, array[1] * val, array[2] * val);
	}

    // returns the angle between two vectors *in degrees*
    static T angleBetween(const vec3<T> &v0, const vec3<T> &v1) {
        T l0 = v0.length();
        T l1 = v1.length();
        if(l0 <= 0.0f || l1 <= 0.0f)
            return 0.0f;
        else
            return math::radiansToDegrees(acosf(math::clamp(vec3<T>::dot(v0, v1) / l0 / l1, -1.0f, 1.0f)));
    }

	inline T* getData() {
		return &array[0];
	}

	inline std::vector<T> toStdVector() const {
		std::vector<T> result(3);
		result[0] = x;
		result[1] = y;
		result[2] = z;
		return result;
	}

    inline std::string toString(char separator = ' ') const {
        return toString(std::string(1, separator));
    }

    inline std::string toString(const std::string &separator) const {
        return std::to_string(x) + separator + std::to_string(y) + separator + std::to_string(z);
    }

	static const vec3<T> origin;
	static const vec3<T> eX;
	static const vec3<T> eY;
	static const vec3<T> eZ;

	inline vec1<T> getVec1() const {
		return vec1<T>(x);
	}
	inline vec2<T> getVec2() const {
		return vec2<T>(x,y);
	}

	union {
		struct {
			T x,y,z;    // standard names for components
		};
		struct {
			T r,g,b;	// colors
		};
		T array[3];     // array access
	};
};

//! operator for scalar * vector
template <class T>
inline vec3<T> operator*(T s, const vec3<T>& v) {
	return v * s;
}
template <class T>
inline vec3<T> operator/(T s, const vec3<T>& v)
{
	return vec3<T>(s/v.x, s/v.y, s/v.z);
}
template <class T>
inline vec3<T> operator+(T s, const vec3<T>& v)
{
	return v + s;
}
template <class T>
inline vec3<T> operator-(T s, const vec3<T>& v)
{
	return vec3<T>(s-v.x, s-v.y, s-v.z);
}
 
namespace math {
	template<class T>
	inline vec3<int> sign(const vec3<T>& v) {
		return vec3<int>(sign(v.x), sign(v.y), sign(v.z));
	}
}


//! write a vec3 to a stream
template <class T> 
inline std::ostream& operator<<(std::ostream& s, const vec3<T>& v) {
  return (s << v[0] << " " << v[1] << " " << v[2]);
}

//! read a vec3 from a stream
template <class T> 
inline std::istream& operator>>(std::istream& s, vec3<T>& v) {
  return (s >> v[0] >> v[1] >> v[2]);
}

typedef vec3<double> vec3d;
typedef vec3<float> vec3f;
typedef vec3<int> vec3i;
typedef vec3<short> vec3s;
typedef vec3<unsigned short> vec3us;
typedef vec3<unsigned int> vec3ui;
typedef vec3<unsigned char> vec3uc;
typedef vec3<UINT64> vec3ul;
typedef vec3<INT64> vec3l;

namespace math {
	template<class T>
	inline bool floatEqual(const vec3<T>& v0, const vec3<T>& v1) {
		return
			floatEqual(v0.x, v1.x) &&
			floatEqual(v0.y, v1.y) &&
			floatEqual(v0.z, v1.z);
	}
	template<class T>
	inline vec3<T> frac(const vec3<T>& f) {
		return vec3<T>(frac(f.x), frac(f.y), frac(f.z));
	}
	template<class T>
	inline vec3i round(const vec3<T>& f) {
		return vec3i(round(f.x), round(f.y), round(f.z));
	}
	template<class T>
	inline vec3i ceil(const vec3<T>& f) {
		return vec3i(ceil(f.x), ceil(f.y), ceil(f.z));
	}
	template<class T>
	inline vec3i floor(const vec3<T>& f) {
		return vec3i(floor(f.x), floor(f.y), floor(f.z));
	}
	template<class T>
	inline vec3<T> abs(const vec3<T>& p) {
		return vec3<T>(abs(p.x), abs(p.y), abs(p.z));
	}
    template<class T>
    inline vec3<T> max(const vec3<T>& p, T v) {
        return vec3<T>( std::max(p.x, v),
                           std::max(p.y, v),
                           std::max(p.z, v));
    }
	template<class T>
	inline vec3<T> max(const vec3<T>& p, const vec3<T>& v) {
		return vec3<T>(
			std::max(p.x, v.x),
			std::max(p.y, v.y),
			std::max(p.z, v.z));
	}
	template<class T>
	inline vec3<T> min(const vec3<T>& p, T v) {
		return vec3<T>(	std::min(p.x, v),
							std::min(p.y, v),
							std::min(p.z, v));
	}
	template<class T>
	inline vec3<T> min(const vec3<T>& p, const vec3<T>& v) {
		return vec3<T>(
			std::min(p.x, v.x),
			std::min(p.y, v.y),
			std::min(p.z, v.z));
	}
	template<class T>
	inline vec3<T> clamp(const vec3<T>& p, T pMin, T pMax) {
		return vec3<T>(
			clamp(p.x, pMin, pMax),
			clamp(p.y, pMin, pMax),
			clamp(p.z, pMin, pMax));
	}

    template<class T>
    inline T triangleArea(const vec3<T>& v0, const vec3<T>& v1, const vec3<T>& v2) {
        return ((v1 - v0) ^ (v2 - v0)).length() * (T)0.5;
    }

    template<class T>
    inline T triangleArea(const vec2<T>& v0, const vec2<T>& v1, const vec2<T>& v2) {
        return triangleArea(vec3<T>(v0, 0.0f),
                            vec3<T>(v1, 0.0f),
                            vec3<T>(v2, 0.0f));
    }

    template<class T>
    inline vec3<T> triangleNormal(const vec3<T>& v0, const vec3<T>& v1, const vec3<T>& v2) {
        return ((v1 - v0) ^ (v2 - v0)).getNormalized();
    }

    template<class T>
    inline float trianglePointDistSq(const vec3<T>& t0, const vec3<T>& t1, const vec3<T>& t2, const vec3<T>& p) {
        const vec3<T> edge0 = t1 - t0;
        const vec3<T> edge1 = t2 - t0;
        const vec3<T> v0 = t0 - p;

        float a = edge0 | edge0;
        float b = edge0 | edge1;
        float c = edge1 | edge1;
        float d = edge0 | v0;
        float e = edge1 | v0;

        float det = a*c - b*b;
        float s = b*e - c*d;
        float t = b*d - a*e;

        if (s + t < det)
        {
            if (s < 0.f)
            {
                if (t < 0.f)
                {
                    if (d < 0.f)
                    {
                        s = clamp(-d / a, 0.f, 1.f);
                        t = 0.f;
                    }
                    else
                    {
                        s = 0.f;
                        t = clamp(-e / c, 0.f, 1.f);
                    }
                }
                else
                {
                    s = 0.f;
                    t = clamp(-e / c, 0.f, 1.f);
                }
            }
            else if (t < 0.f)
            {
                s = clamp(-d / a, 0.f, 1.f);
                t = 0.f;
            }
            else
            {
                float invDet = 1.f / det;
                s *= invDet;
                t *= invDet;
            }
        }
        else
        {
            if (s < 0.f)
            {
                float tmp0 = b + d;
                float tmp1 = c + e;
                if (tmp1 > tmp0)
                {
                    float numer = tmp1 - tmp0;
                    float denom = a - 2 * b + c;
                    s = clamp(numer / denom, 0.f, 1.f);
                    t = 1 - s;
                }
                else
                {
                    t = clamp(-e / c, 0.f, 1.f);
                    s = 0.f;
                }
            }
            else if (t < 0.f)
            {
                if (a + d > b + e)
                {
                    float numer = c + e - b - d;
                    float denom = a - 2 * b + c;
                    s = clamp(numer / denom, 0.f, 1.f);
                    t = 1 - s;
                }
                else
                {
                    s = clamp(-e / c, 0.f, 1.f);
                    t = 0.f;
                }
            }
            else
            {
                float numer = c + e - b - d;
                float denom = a - 2 * b + c;
                s = clamp(numer / denom, 0.f, 1.f);
                t = 1.f - s;
            }
        }

        const vec3f pos = t0 + s * edge0 + t * edge1;
        return vec3f::distSq(pos, p);
    }

    template<class T>
    inline float trianglePointDist(const vec3<T>& t0, const vec3<T>& t1, const vec3<T>& t2, const vec3<T>& p)
    {
        return sqrtf(trianglePointDistSq(t0, t1, t2, p));
    }
}

}  // namespace ml

#endif  // CORE_MATH_POINT3D_H_
