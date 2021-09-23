#ifndef CORE_UTIL_STRINGUTIL_H_
#define CORE_UTIL_STRINGUTIL_H_

#include <string>
#include <vector>
#include <core-base/common.h>
namespace ml
{

namespace util
{
typedef unsigned int UINT;//add by guan
	//TODO TEST
	inline bool startsWith(const std::string& str, const std::string& startCandidate) {
		if (str.length() < startCandidate.length()) { return false; }
		for (size_t i = 0; i < startCandidate.length(); i++) {
			if (str[i] != startCandidate[i]) { return false; }
		}
		return true;
	}

	//TODO TEST
	inline bool endsWith(const std::string& str, const std::string& endCandidate) {
		if (str.length() < endCandidate.length()) { return false; }
		for (size_t i = 0; i < endCandidate.length(); i++) {
			if (str[str.length() - endCandidate.length() + i] != endCandidate[i]) { return false; }
		}
		return true;
	}

	//TODO TEST
	inline bool exactMatchAtOffset(const std::string& str, const std::string& find, size_t offset) {
		size_t MatchLength = 0;
		for (size_t i = 0; i + offset < str.length() && i < find.length(); i++) {
			if (str[i + offset] == find[i]) {
				MatchLength++;
				if (MatchLength == find.length()) { return true; }
			}
		}
		return false;
	}

    //TODO TEST
    inline bool contains(const std::string& str, const std::string& find) {
        for (size_t i = 0; i < str.length(); i++)
        {
            if (exactMatchAtOffset(str, find, i))
                return true;
        }
        return false;
    }

    inline bool contains(const std::string& str, unsigned char find) {
        for (size_t i = 0; i < str.length(); i++)
        {
            if (str[i] == find)
                return true;
        }
        return false;
    }

    //TODO TEST
    inline std::string zeroPad(UINT value, UINT totalLength) {
        std::string result = std::to_string(value);
        while (result.size() < totalLength)
            result = "0" + result;
        return result;
    }

	//TODO TEST
	inline std::string replace(const std::string& str, const std::string& find, const std::string& replace) {
		std::string result;
		for (size_t i = 0; i < str.length(); i++) {
			if (exactMatchAtOffset(str, find, i)) {
				result += replace;
				i += find.length() - 1;
			} else { result += str[i]; }
		}
		return result;
	}

    //TODO TEST
    inline std::string remove(const std::string& str, const std::string& find) {
        return replace(str, find, "");
    }

    inline std::string remove(const std::string& str, const std ::vector< std::string >& find) {
        std::string result = str;
        for (const auto &s : find)
            result = replace(result, s, "");
        return result;
    }

	inline std::string replace(const std::string& str, char find, char replace) {
		return util::replace(str, std::string(1, find), std::string(1, replace));
	}

	//TODO TEST
	inline std::vector<std::string> split(const std::string& str, const std::string& separator, bool pushEmptyStrings = false) {
		MLIB_ASSERT_STR(separator.length() >= 1, "empty separator");
		std::vector<std::string> result;

        if (str.size() == 0)
        {
            result.push_back("");
            return result;
        }

		std::string entry;
		for (size_t i = 0; i < str.length(); i++) {
			bool isSeperator = true;
			for (size_t testIndex = 0; testIndex < separator.length() && i + testIndex < str.length() && isSeperator; testIndex++) {
				if (str[i + testIndex] != separator[testIndex]) {
					isSeperator = false;
				}
			}
			if (isSeperator) {
				if (entry.length() > 0 || pushEmptyStrings) {
					result.push_back(entry);
					entry.clear();
				}
				i += separator.size() - 1;
			} else {
				entry.push_back(str[i]);
			}
		}
		if (entry.length() > 0) { result.push_back(entry); }
		return result;
	}

	inline std::vector<std::string> split(const std::string& str, const char separator, bool pushEmptyStrings = false) {
		return split(str, std::string(1, separator), pushEmptyStrings);
	}


	//! converts all chars of a string to lowercase (returns the result)
	inline std::string toLower(const std::string& str) {
		std::string res(str);
		for (size_t i = 0; i < res.length(); i++) {
			if (res[i] <= 'Z' &&  res[i] >= 'A') {
				res[i] -= ('Z' - 'z');
			}
		}
		return res;
	}
	//! converts all chars of a string to uppercase (returns the result)
	inline std::string toUpper(const std::string& str) {
		std::string res(str);
		for (size_t i = 0; i < res.length(); i++) {
			if (res[i] <= 'z' &&  res[i] >= 'a') {
				res[i] += ('Z' - 'z');
			}
		}
		return res;
	}

	//! removes all characters from a string
	inline std::string removeChar(const std::string& strInput, const char c) {
		std::string str(strInput);
		str.erase(std::remove(str.begin(), str.end(), c), str.end());
		return str;
	}

	//! gets the file extension (ignoring case)
	inline std::string getFileExtension(const std::string& filename) {
		size_t pos = filename.rfind(".");
		if (pos == std::string::npos) return "";
		std::string extension = filename.substr(pos + 1);
		for (unsigned int i = 0; i < extension.size(); i++) {
			extension[i] = tolower(extension[i]);
		}
		return extension;
	}

	//! returns substring from beginning of str up to before last occurrence of delim
	inline std::string getSubstrBeforeLast(const std::string& str, const std::string& delim) {
		std::string trimmed = str;
		return trimmed.erase(str.rfind(delim));
	}

	// TODO: this is broken if the filename has both / and \.
	inline std::string getFilenameFromPath(const std::string& filename) {
		std::string name = filename;
		size_t pos = filename.rfind("/");
		if (pos == std::string::npos) pos = filename.rfind("\\");
		if (pos == std::string::npos) return filename;
		name.erase(name.begin(), name.begin() + pos + 1);
		return name;
	}

	inline bool hasFileExtension(const std::string& filename) {
		std::string filepart = getFilenameFromPath(filename);
		size_t p = filepart.rfind(".");
		if (p == 0 || p == std::string::npos) return false;
		return true;
	}

    //! splits string about the first instance of delim
    inline std::pair<std::string, std::string> splitOnFirst(const std::string& str, const std::string& delim) {
        std::pair<std::string, std::string> result;
        auto firstIndex = str.find(delim);
        result.first = str.substr(0, firstIndex);
        result.second = str.substr(firstIndex + delim.size());
        return result;
    }

	//! splits string about the first instance of delim
	inline std::pair<std::string, std::string> splitOnLast(const std::string& str, const std::string& delim) {
		std::pair<std::string, std::string> result;
		auto lastIndex = str.rfind(delim);
		result.first = str.substr(0, lastIndex);
		result.second = str.substr(lastIndex + delim.size());
		return result;
	}

	//! splits string about the first instance of any string in delims
	inline std::pair<std::string, std::string> splitOnFirst(const std::string& str, const std::vector<std::string>& delims) {
		std::pair<std::string, std::string> result;
		size_t firstIndex = std::string::npos; unsigned int delimIdx = (unsigned int)-1;
		for (unsigned int i = 0; i < delims.size(); i++) {
			auto idx = str.find(delims[i]);
			if (idx < firstIndex) {
				firstIndex = idx;
				delimIdx = i;
			}
		}
		result.first = str.substr(0, firstIndex);
		result.second = (firstIndex == std::string::npos) ? "" : str.substr(firstIndex + delims[delimIdx].size());
		return result;
	}

	//! returns filename with extension removed
	inline std::string dropExtension(const std::string& filename) {
		return getSubstrBeforeLast(filename, ".");
	}

	//! trims any whitespace on right of str and returns
	inline std::string rtrim(const std::string& str) {
		std::string trimmed = str;
		return trimmed.erase(str.find_last_not_of(" \n\r\t") + 1);
	}

	//! returns the integer of the last suffix
	inline unsigned int getNumericSuffix(const std::string& str) {
		std::string suffix;
		unsigned int i = 0;
		while (i < str.length()) {
			char curr = str[str.length()-1-i];
			if (curr >= '0' && curr <= '9') {
				suffix = curr + suffix;
			} else {
				break;
			}
			i++;
		}
		if (suffix.length() > 0) {
			return std::atoi(suffix.c_str());
		} else {
			return (unsigned int)(-1);
		}
	}

	//! returns the first integer found
	inline unsigned int getFirstNumeric(const std::string& str) {
		std::string number;
		unsigned int i = 0;
		bool found = false;
		while (i < str.length()) {
			char curr = str[i];
			if (curr >= '0' && curr <= '9') {
				number = number + curr;
				found = true;
			}
			else if (found) {
				break;
			}
			i++;
		}
		if (number.length() > 0) {
			return std::atoi(number.c_str());
		}
		else {
			return (unsigned int)(-1);
		}
	}

	inline std::string getBaseBeforeNumericSuffix(const std::string& str) {
		std::string base = str;
		unsigned int i = 0;
		while (i < str.length()) {
			char curr = str[str.length()-1-i];
			if (curr >= '0' && curr <= '9') {
				base.pop_back();
			} else {
				break;
			}
			i++;
		}
		return base;
	}

#ifdef UNICODE
	inline std::wstring windowsStr(const std::string& s) 
	{
		std::wstring ret(s.begin(), s.end());
		return ret;
	}
	inline std::wstring windowsStr(const std::wstring& s) 
	{
		return s;
	}
#else
	inline std::string windowsStr(const std::string& s)
	{
		return s;
	}
	inline std::string windowsStr(const std::wstring& s)
	{
		std::string ret(s.begin(), s.end());
		return ret;

	}
#endif

}  // namespace util

template<class T>
inline std::ostream& operator<<(std::ostream& s, const std::vector<T>& v) {
	s << "vector size " << v.size() << "\n";
	for (size_t i = 0; i < v.size(); i++) {
		s << '\t' << v[i];
		if (i != v.size() - 1) s << '\n';
	}
	return s;
}


template<class T>
inline std::ostream& operator<<(std::ostream& s, const std::list<T>& l) {
	s << "list size " << l.size() << "\n";  
	for (auto iter = l.begin(); iter != l.end(); iter++) {
		s << '\t' << *iter;
		if (iter != --l.end()) s << '\n';
	}
	return s;
}

}  // namespace ml

#endif  // CORE_UTIL_STRINGUTIL_H__
