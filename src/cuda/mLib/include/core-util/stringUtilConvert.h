
#ifndef CORE_UTIL_STRINGUTILCONVERT_H_
#define CORE_UTIL_STRINGUTILCONVERT_H_

#include <string>

/////////////////////////////////////////////////////////////////////
// util is already used before; THIS must be included after  //
// all types have been declared for proper conversion              //
/////////////////////////////////////////////////////////////////////

//////////////////////
// native functions //
//////////////////////
namespace ml {

namespace convert {
	inline int toInt(const std::string& s) {
		return std::stoi(s);
	}
    inline long long toInt64(const std::string& s) {
        return std::strtoll(s.c_str(), NULL, 10);
    }
    inline int toInt(bool b) {
        if(b) return 1;
        return 0;
    }
	inline unsigned int toUInt(const std::string& s) {
		//return (unsigned int)toInt(s);
		return std::stoul(s.c_str(), NULL, 10);
	}
	inline long long toUInt64(const std::string& s) {
		return std::strtoull(s.c_str(), NULL, 10);
	}
	inline double toDouble(const std::string& s) {
		return std::stod(s);
	}
	inline float toFloat(const std::string& s) {
		return std::stof(s);
	}
	inline char toChar(const std::string& s) {
		return s[0];
	}
	inline bool toBool(const std::string& s) {
		if (s == "false" || s == "False" || s == "0") { return false; }
		else { return true; }
	}
    template<class T>
    inline std::string toString(const std::vector<T>& val) {
        std::string result;
        for (const auto &e : val)
        {
            result = result + std::to_string(e) + " ";
        }
        return result;
    }
    template<class T>
    inline std::string toString(const vec3<T>& p) {
        std::string result;
        for (const auto &e : p.array)
        {
            result = result + std::to_string(e) + ",";
        }
        return result;
    }
	template<class T>
	inline std::string toString(const T& val) {
		return std::to_string(val);
	}
	template<class U> inline vec2<U> toPoint2D(const std::string& s) {
		vec3<U> ret;
		std::stringstream ss(util::removeChar(s, 'f'));
		ss >> ret.x >> ret.y;
		return ret;
	}
	template<class U> inline vec3<U> toPoint3D(const std::string& s) {
		vec3<U> ret;
		std::stringstream ss(util::removeChar(s, 'f'));
		ss >> ret.x >> ret.y >> ret.z;
		return ret;
	}
	template<class U> inline vec4<U> toPoint4D(const std::string& s) {
		vec4<U> ret;
		std::stringstream ss(util::removeChar(s, 'f'));
		ss >> ret.x >> ret.y >> ret.z >> ret.w;
		return ret;
	}


	template<class T> inline void to(const std::string& s, T& res);

	template<>  inline void to<int>(const std::string& s, int& res) {
		res = toInt(s);
	}
	template<>  inline void to<unsigned int>(const std::string& s, unsigned int& res) {
		res = toUInt(s);
	}
	template<>  inline void to<long long>(const std::string& s, long long& res) {
		res = toInt64(s);
	}
	template<>  inline void to<unsigned long long>(const std::string& s, unsigned long long& res) {
		res = toUInt64(s);
	}
	template<>  inline void to<double>(const std::string& s, double& res) {
		res = toDouble(s);
	}
	template<>  inline void to<float>(const std::string& s, float& res) {
		res = toFloat(s);
	}
	template<>  inline void to<std::string>(const std::string& s, std::string& res) {
		res = s;
	}
	template<>  inline void to<char>(const std::string& s, char& res) {
		res = toChar(s);
	}
	template<> inline void to<bool>(const std::string& s, bool& res) {
		res = toBool(s);
	}
	template<class U> inline void to(const std::string& s, vec2<U>& res) {
		std::stringstream ss(util::removeChar(s, 'f'));
		ss >> res.x >> res.y;
	}
	template<class U> inline void to(const std::string& s, vec3<U>& res) {
        // TODO: abstract and extend to other vecN::"to"
        std::string sFixed = util::removeChar(s, 'f');
        if (util::contains(sFixed, ','))
            sFixed = util::replace(sFixed, ',', ' ');
        std::stringstream ss(sFixed);

		ss >> res.x >> res.y >> res.z;
	}
	template<class U> inline void to(const std::string& s, vec4<U>& res) {
		std::stringstream ss(util::removeChar(s, 'f'));
		ss >> res.x >> res.y >> res.z >> res.w;
	} 
	template<class U> inline void to(const std::string& s, Matrix4x4<U>& res) {
		std::stringstream ss(util::removeChar(s, 'f'));
		ss >> res(0,0) >> res(0,1) >> res(0,2)  >> res(0,3) >>
			  res(1,0) >> res(1,1) >> res(1,2)  >> res(1,3) >>
			  res(2,0) >> res(2,1) >> res(2,2)  >> res(2,3) >>
			  res(3,0) >> res(3,1) >> res(3,2)  >> res(3,3);
	}

}  // namespace Convert

namespace util {

	////////////////////////
	// template overloads //
	////////////////////////
	template<class T> inline T convertTo(const std::string& s) {
		T res;
		convert::to(s, res);
		return res;
	}

	template<class T> inline void convertTo(const std::string& s, T& res) {
		convert::to<T>(s, res);
	}

	template<class U> inline void convertTo(const std::string& s, vec2<U>& res) {
		convert::to(s, res);
	}
	template<class U> inline void convertTo(const std::string& s, vec3<U>& res) {
		convert::to(s, res);
	}
	template<class U> inline void convertTo(const std::string& s, vec4<U>& res) {
		convert::to(s, res);
	}

	template<class U> inline void convertTo(const std::string& s,  Matrix4x4<U>& res) {
		convert::to(s, res);
	}
}  // namespace util

//! stringstream functionality
template<class T>
inline std::string& operator<<(std::string& s, const T& in) {
	s += std::to_string(in);
	return s;
}

}  // namespace ml

#endif  // CORE_UTIL_STRINGUTILCONVERT_H_
