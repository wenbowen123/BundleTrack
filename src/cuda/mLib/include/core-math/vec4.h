
#ifndef CORE_MATH_POINT4D_H_
#define CORE_MATH_POINT4D_H_

#include "vec3.h"

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cassert>

namespace ml {

//! 4D vector.
template <class T>
class vec4 : public BinaryDataSerialize< vec4<T> >
{
    public:
        explicit vec4(T v) {
            array[0] = array[1] = array[2] = array[3] = v;
        }

        vec4() {
            array[0] = array[1] = array[2] = array[3] = 0;
        }

        vec4(T x, T y, T z, T w) {
            array[0] = x;
            array[1] = y;
            array[2] = z;
            array[3] = w;
        }
		
		template <class U>
		vec4(const vec4<U>& other) {
			array[0] = (T)other.array[0];
			array[1] = (T)other.array[1];
			array[2] = (T)other.array[2];
			array[3] = (T)other.array[3];
		}

        explicit vec4(const vec3<T>& other, T w = (T)1) {
            array[0] = other.array[0];
            array[1] = other.array[1];
            array[2] = other.array[2];
            array[3] = w;
        }

        vec4(const vec4& other) {
            array[0] = other.array[0];
            array[1] = other.array[1];
            array[2] = other.array[2];
            array[3] = other.array[3];
        }

        explicit vec4(const T* other) {
			array[0] = other[0];
			array[1] = other[1];
			array[2] = other[2];
			array[3] = other[3];
		}

        inline const vec4<T>& operator=(const vec4& other) {
            array[0] = other.array[0];
            array[1] = other.array[1];
            array[2] = other.array[2];
            array[3] = other.array[3];
            return *this;
        }

        inline vec4<T> operator-() const {
            return vec4<T>(-array[0], -array[1], -array[2], -array[3]);
        }

        inline vec4<T> operator+(const vec4& other) const {
            return vec4<T>(array[0] + other.array[0], array[1] + other.array[1],
                              array[2] + other.array[2], array[3] + other.array[3]);
        }

		inline vec4<T> operator+(T val) const {
			return vec4<T>(array[0]+val, array[1]+val, array[2]+val, array[3]+val);
		}

        inline void operator+=(const vec4& other) {
            array[0] += other.array[0];
            array[1] += other.array[1];
            array[2] += other.array[2];
            array[3] += other.array[3];
        }

        inline void operator-=(const vec4& other) {
            array[0] -= other.array[0];
            array[1] -= other.array[1];
            array[2] -= other.array[2];
            array[3] -= other.array[3];
        }

		inline void operator+=(T val) {
			array[0] += val;
			array[1] += val;
			array[2] += val;
			array[3] += val;
		}

		inline void operator-=(T val) {
			array[0] -= val;
			array[1] -= val;
			array[2] -= val;
			array[3] -= val;
		}

        inline void operator*=(T val) {
            array[0] *= val;
            array[1] *= val;
            array[2] *= val;
            array[3] *= val;
        }

        inline void operator/=(T val) {
			//optimized version for float/double (doesn't work for int) -- assumes compiler statically optimizes if
			if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
				T inv = (T)1 / val;
				array[0] *= inv;
				array[1] *= inv;
				array[2] *= inv;
				array[3] *= inv;
			}
			else {
				array[0] /= val;
				array[1] /= val;
				array[2] /= val;
				array[3] /= val;
			}
        }

        inline vec4<T> operator*(T val) const {
            return vec4<T>(array[0] * val, array[1]*val, array[2]*val, array[3]*val);
        }

        inline vec4<T> operator/(T val) const {
			//optimized version for float/double (doesn't work for int) -- assumes compiler statically optimizes if
			if (std::is_same<T, float>::value || std::is_same<T, double>::value) {
				T inv = (T)1 / val;
				return vec4<T>(array[0] * inv, array[1] * inv, array[2] * inv, array[3] * inv);
			}
			else {
				return vec4<T>(array[0] / val, array[1] / val, array[2] / val, array[3] / val);
			}
        }

        //! cross product (of .xyz)
        inline vec4<T> operator^(const vec4& other) const {
            return vec4<T>(array[1] * other.array[2] - array[2] * other.array[1],
                              array[2] * other.array[0] - array[0] * other.array[2],
                              array[0] * other.array[1] - array[1] * other.array[0], T(1));
        }

        //! dot product
        inline T operator|(const vec4& other) const {
            return (array[0] * other.array[0] + array[1] * other.array[1] + array[2] *
                    other.array[2] + array[3] * other.array[3]);
        }

        inline vec4<T> operator-(const vec4& other) const {
            return vec4<T>(array[0]-other.array[0], array[1]-other.array[1], array[2]-other.array[2], array[3]-other.array[3]);
        }

		inline vec4<T> operator-(T val) const {
			return vec4<T>(array[0]-val, array[1]-val, array[2]-val, array[3]-val);
		}

        inline bool operator==(const vec4& other) const {
            if ((array[0] == other.array[0]) && (array[1] == other.array[1]) &&
                (array[2] == other.array[2]) && (array[3] == other.array[3]))
            { return true; }

            return false;
        }

		inline bool operator!=(const vec4& other) const {
			return !(*this == other);
		}

        inline T lengthSq() const {
            return (array[0]*array[0] + array[1]*array[1] + array[2]*array[2] + array[3]*array[3]);
        }

        inline T length() const {
            return sqrt(lengthSq());
        }

        static inline vec4<T> normalize(const vec4<T> &v) {
            return v.getNormalized();
        }

        static T distSq(const vec4& v0, const vec4& v1) {
            return (
                       (v0.array[0] - v1.array[0]) * (v0.array[0] - v1.array[0]) +
                       (v0.array[1] - v1.array[1]) * (v0.array[1] - v1.array[1]) +
                       (v0.array[2] - v1.array[2]) * (v0.array[2] - v1.array[2]) +
                       (v0.array[3] - v1.array[3]) * (v0.array[3] - v1.array[3])
                   );
        }

        static T dist(const vec4& v0, const vec4& v1)  {
            return (v0 - v1).length();
        }

		static vec4<T> randomUniform(T min, T max) {
			return vec4<T>(math::randomUniform(min, max),
				math::randomUniform(min, max),
				math::randomUniform(min, max),
				math::randomUniform(min, max));
		}

        void print() const
        {
            std::cout << "(" << array[0] << " / " << array[1] << " / " << array[2] <<
                           " / " << array[3] << " ) " << std::endl;
        }

        inline const T& operator[](int i) const
        {
            assert(i < 4);
            return array[i];
        }

        inline T& operator[](int i)
        {
            assert(i < 4);
            return array[i];
        }

        inline void normalize()
        {
            T val = (T)1.0 / length();
            array[0] *= val;
            array[1] *= val;
            array[2] *= val;
            array[3] *= val;
        }

        inline bool isValid() const
        {
            return (x == x && y == y && z == z && w == w);
        }

        inline vec4<T> getNormalized() const
        {
            T val = (T)1.0 / length();
            return vec4<T>(array[0] * val, array[1] * val, array[2] * val,
                              array[3] * val);
        }

        inline void dehomogenize()
        {
            array[0] /= array[3];
            array[1] /= array[3];
            array[2] /= array[3];
            array[3] /= array[3];
        }


        inline bool isLinearDependent(const vec4& other) const
        {
            T factor = x / other.x;

            if ((std::fabs(x / factor - other.x) + std::fabs(y / factor - other.y) +
                    std::fabs(z / factor - other.z) + std::fabs(w / factor - other.w)) < 0.00001)
            { return true; }
            else
            { return false; }
        }

		inline T* getData() {
			return &array[0];
		}

		inline std::vector<T> toStdVector() const {
			std::vector<T> result(4);
			result[0] = x;
			result[1] = y;
			result[2] = z;
			result[3] = w;
			return result;
		}

		inline std::string toString(char separator = ' ') const {
			return toString(std::string(1, separator));
		}

		inline std::string toString(const std::string &separator) const {
			return std::to_string(x) + separator + std::to_string(y) + separator + std::to_string(z) + separator + std::to_string(w);
		}


		static const vec4<T> origin;
		static const vec4<T> eX;
		static const vec4<T> eY;
		static const vec4<T> eZ;
		static const vec4<T> eW;

		inline vec1<T> getVec1() const {
			return vec1<T>(x);
		}
		inline vec2<T> getVec2() const {
			return vec2<T>(x,y);
		}
		inline vec3<T> getVec3() const {
			return vec3<T>(x,y,z);
		}

        union
        {
            struct
            {
                T x, y, z, w; // standard names for components
            };
            struct {
				T r, g, b, a;  // colors
            };
            T array[4];     // array access
        };
};

//! operator for scalar * vector
template <class T>
inline vec4<T> operator*(T s, const vec4<T>& v)
{
    return v * s;
}
template <class T>
inline vec4<T> operator/(T s, const vec4<T>& v)
{
	return vec4<T>(s/v.x, s/v.y, s/v.z, s/v.w);
}
template <class T>
inline vec4<T> operator+(T s, const vec4<T>& v)
{
	return v + s;
}
template <class T>
inline vec4<T> operator-(T s, const vec4<T>& v)
{
	return vec4<T>(s-v.x, s-v.y, s-v.z, s-v.w);
}

namespace math {
	template<class T>
	inline vec4<int> sign(const vec4<T>& v) {
		return vec4<int>(sign(v.x), sign(v.y), sign(v.z), sign(v.w));
	}
}


//! write a vec4 to a stream
template <class T>
inline std::ostream& operator<<(std::ostream& s, const vec4<T>& v) {
  return (s << v[0] << " " << v[1] << " " << v[2] << " " << v[3]);
}

//! read a vec4 from a stream
template <class T>
inline std::istream& operator>>(std::istream& s, vec4<T>& v) {
  return (s >> v[0] >> v[1] >> v[2] >> v[3]);
}


typedef vec4<double> vec4d;
typedef vec4<float> vec4f;
typedef vec4<int> vec4i;
typedef vec4<short> vec4s;
typedef vec4<short> vec4us;
typedef vec4<unsigned int> vec4ui;
typedef vec4<unsigned char> vec4uc;
typedef vec4<UINT64> vec4ul;
typedef vec4<INT64> vec4l;

namespace math {
	template<class T>
	inline bool floatEqual(const vec4<T>& v0, const vec4<T>& v1) {
		return
			floatEqual(v0.x, v1.x) &&
			floatEqual(v0.y, v1.y) &&
			floatEqual(v0.z, v1.z) &&
			floatEqual(v0.w, v1.w);
	}
	template<class T>
	inline vec4<T> frac(const vec4<T>& f) {
		return vec4<T>(frac(f.x), frac(f.y), frac(f.z), frac(f.w));
	}
	template<class T>
	inline vec4i round(const vec4<T>& f) {
		return vec4i(round(f.x), round(f.y), round(f.z), round(f.w));
	}
	template<class T>
	inline vec4i ceil(const vec4<T>& f) {
		return vec4i(ceil(f.x), ceil(f.y), ceil(f.z), ceil(f.w));
	}
	template<class T>
	inline vec4i floor(const vec4<T>& f) {
		return vec4i(floor(f.x), floor(f.y), floor(f.z), floor(f.w));
	}
	template<class T>
	inline vec4<T> abs(const vec4<T>& p) {
		return vec4<T>(abs(p.x), abs(p.y), abs(p.z), abs(p.w));
	}
	template<class T>
	inline vec4<T> max(const vec4<T>& p, T v) {
		return vec4<T>(
			std::max(p.x, v),
			std::max(p.y, v),
			std::max(p.z, v),
			std::max(p.w, v));
	}
	template<class T>
	inline vec4<T> max(const vec4<T>& p, const vec4<T>& v) {
		return vec4<T>(
			std::max(p.x, v.x),
			std::max(p.y, v.y),
			std::max(p.z, v.z),
			std::max(p.w, v.w));
	}
	template<class T>
	inline vec4<T> min(const vec4<T>& p, T v) {
		return vec4<T>(
			std::min(p.x, v),
			std::min(p.y, v),
			std::min(p.z, v),
			std::min(p.w, v));
	}
	template<class T>
	inline vec4<T> min(const vec4<T>& p, const vec4<T>& v) {
		return vec4<T>(
			std::min(p.x, v.x),
			std::min(p.y, v.y),
			std::min(p.z, v.z),
			std::min(p.w, v.w));
	}
	template<class T>
	inline vec4<T> clamp(const vec4<T>& p, T pMin, T pMax) {
		return vec4<T>(
			clamp(p.x, pMin, pMax),
			clamp(p.y, pMin, pMax),
			clamp(p.z, pMin, pMax),
			clamp(p.w, pMin, pMax));
	}
}

}  // namespace ml

#endif  // CORE_MATH_POINT4D_H_
