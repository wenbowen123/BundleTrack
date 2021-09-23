
#ifndef CORE_UTIL_PARAMETERFILE_H_
#define CORE_UTIL_PARAMETERFILE_H_
#include <iostream>//add by guan
#include <sstream>
#include <string>
#include <map>
#include <set>
#include <fstream>
#include <list>
#include "core-base/common.h"
namespace ml {

	class ParameterFile {
	public:
		ParameterFile(char separator = '=', bool caseSensitive = true) {
			m_Separator = separator;
			m_CaseSensitive = caseSensitive;
		}

		ParameterFile(const std::string& filename, char separator = '=', bool caseSensitive = true) {
			m_Separator = separator;
			m_CaseSensitive = caseSensitive;
			addParameterFile(filename);
		}

		void addParameterFile(const std::string& filename) {
			std::ifstream file(filename);
			if (!file.is_open()) throw MLIB_EXCEPTION(filename);

			while (!file.eof()) {
				std::string line;
				getline(file, line);
				removeComments(line);
				removeSpecialCharacters(line);
				if (line.length() <= 1) continue;

				size_t separator = line.find(m_Separator);	//split the string at separator
				if (separator == std::string::npos)	{
					MLIB_WARNING("No seperator found in line: " + line); //std::string(1, m_Separator) + ":" + 
					continue;
				}
				std::string attr_name = line.substr(0, separator);
				std::string attr_value = line.substr(separator + 1, line.length() - 1);
				removeSpecialCharacters(attr_name);
				removeSpecialCharacters(attr_value);
				if (attr_name.length() == 0) {
					MLIB_WARNING("Invalid attribute or value");
					continue;
				}
				if (!m_CaseSensitive)	attr_name = util::toLower(attr_name);
				m_Values[attr_name] = attr_value;
			}
			file.close();

			bool found = false;
			for (const auto &file : m_Filenames) {
				if (file == filename)	found = true;
			}
			if (!found)	m_Filenames.push_back(filename);
		}

		void reload() {
			m_Values.clear();
			for (const auto &file : m_Filenames)
			{
				addParameterFile(file);
			}
		}

		std::string readString(const std::string &paramName) const
		{
			std::string value;
			if (!readParameter(paramName, value))
			{
				std::cout << "parameter not found: " << paramName << std::endl;
				value = "<parameter not found>";
			}
			return value;
		}

		template<class T>
		bool readParameter(const std::string& name, T& value) const {
			if (m_CaseSensitive) {
				const auto s = m_Values.find(name);
				if (s == m_Values.end()) {
					return false;
				}
				else {
					util::convertTo(s->second, value);
					return true;
				}
			}
			else {
				std::string lname(name);	lname = util::toLower(lname);
				const auto s = m_Values.find(name);
				if (s == m_Values.end()) {
					return false;
				}
				else {
					util::convertTo(s->second, value);
					return true;
				}
			}
		}
		template<class U>
		bool readParameter(const std::string& name, std::vector<U>& value) const {
			value.clear();
			for (size_t i = 0;; i++) {
				std::stringstream ss;	ss << i;
				std::string currName = name + "[" + ss.str() + "]";
				U currValue;
				if (readParameter(currName, currValue)) {
					value.resize(i + 1);
					value[i] = currValue;
				}
				else {
					break;
				}
			}
			if (value.size() == 0)	return false;
			else return true;
		}

		template<class U>
		bool readParameter(const std::string& name, std::list<U>& value) const {
			value.clear();
			for (size_t i = 0;; i++) {
				std::stringstream ss;	ss << i;
				std::string currName = name + "[" + ss.str() + "]";
				U currValue;
				if (getParameterForValue(currName, currValue)) {
					value.push_back(currValue);
				}
				else {
					break;
				}
			}
			if (value.size() == 0)	return false;
			else return true;
		}

		void print() const {
			for (auto iter = m_Values.begin(); iter != m_Values.end(); iter++) {
				std::cout << iter->first << " " << m_Separator << " " << iter->second << std::endl;
			}
		}

		void saveToFile(const std::string& filename) const {
			//note: unordered_map killed the original ordering
			std::ofstream ofs(filename);
			if (!ofs.is_open()) throw MLIB_EXCEPTION("failed to open file (" + filename + ") for write");
			for (const auto v : m_Values) {
				ofs << v.first << " " << m_Separator << " " << v.second << std::endl;
			}
			ofs.close();
		}

		const std::map<std::string, std::string>& getParameters() const { return m_Values; }

		void overrideParameter(const std::string &parameter, const std::string &newValue)
		{
			m_Values[parameter] = newValue;
		}

		void deleteParameter(const std::string& parameter, bool verbose = false)
		{
			size_t numErased = m_Values.erase(parameter);
			if (verbose && numErased == 0) std::cout << "warning: could not find parameter (" << parameter << ") to delete" << std::endl;
		}

	private:
		//! removes spaces and tabs at the begin and end of the string
		void removeSpecialCharacters(std::string &str) const {
			char characters[] = { ' ', '\t', '\"', ';', '\r', '\n' };
			const unsigned int length = 6;
			bool found = true;
			while (str.length() && found) {
				found = false;
				for (unsigned int i = 0; i < length; i++) {
					if (*str.begin() == characters[i]) {
						str.erase(str.begin());	found = true;	break;
					}
					if (*(--str.end()) == characters[i]) {
						str.erase(--str.end()); found = true;	break;
					};
				}
			}
		}

		//! searches for comments and removes everything after the comment if there is one
		void removeComments(std::string& s) const {
			const char stringDelim = '\"';
			std::vector<size_t> strpos;
			size_t pos = s.find(stringDelim, 0);
			while (pos != std::string::npos) {
				strpos.push_back(pos);
				pos = s.find(stringDelim, pos + 1);
			}

			std::string comments[] = { "//", "#", ";" };
			const unsigned int length = 3;
			for (unsigned int i = 0; i < length; i++) {
				size_t curr = s.find(comments[i]);
				if (curr != std::string::npos) {
					bool instring = false;
					for (size_t j = 0; j + 1 < strpos.size(); j += 2) { //check if enclosed in string
						if (curr > strpos[j] && curr < strpos[j + 1]) {
							instring = true;
							break;
						}
					}
					if (!instring) s = s.substr(0, curr);
				}
			}
		}
		std::map<std::string, std::string> m_Values;
		char m_Separator;
		bool m_CaseSensitive;
		std::list<std::string> m_Filenames;
	};

}  // namespace ml

#endif  // CORE_UTIL_PARAMETERFILE_H_
