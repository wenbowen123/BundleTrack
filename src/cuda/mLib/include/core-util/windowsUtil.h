
#ifndef WINDOWS_UTIL_H_
#define WINDOWS_UTIL_H_

#ifdef _WIN32

namespace ml {

	namespace util {
		std::string getLastErrorString();
		//! checks an error and exits
		void errorExit(const std::string& functionName);
	}
}

#endif

#endif
