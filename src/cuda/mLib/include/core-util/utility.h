#ifndef CORE_UTIL_UTILITY_H_
#define CORE_UTIL_UTILITY_H_

#include <string>
#include <algorithm>
#include <vector>
#include <sstream>
#include <iomanip>
#define LINUX

namespace ml
{

namespace math
{
static const double PI = 3.1415926535897932384626433832795028842;
static const float PIf = 3.14159265358979323846f;

inline float degreesToRadians(float x)
{
	return x * (PIf / 180.0f);
}

inline float radiansToDegrees(float x)
{
	return x * (180.0f / PIf);
}

inline double degreesToRadians(double x)
{
	return x * (PI / 180.0);
}

inline double radiansToDegrees(double x)
{
	return x * (180.0 / PI);
}

template <class T>
inline bool floatEqual(T v0, T v1, T eps = (T)0.000001)
{
	return (std::abs(v0 - v1) <= eps);
}

template <class T>
inline T linearMap(T s1, T e1, T s2, T e2, T start)
{
	return ((start - s1) * (e2 - s2) / (e1 - s1) + s2);
}

template <class T>
inline T linearMapClamped(T s1, T e1, T s2, T e2, T start)
{
	if (start <= s1)
		return s2;
	if (start >= e1)
		return e2;
	return ((start - s1) * (e2 - s2) / (e1 - s1) + s2);
}

template <class T, class U>
#if __cplusplus >= 201103L || defined __cpp_decltype
inline auto lerp(T left, T right, U s) -> decltype(left * s)
{
	return static_cast<decltype(left * s)>(left + (right - left) * s);
}
#else
inline T lerp(T left, T right, U s)
{
	return static_cast<T>(left + (right - left) * s);
}
#endif

inline int mod(int x, size_t M)
{
	if (x >= 0)
	{
		return (x % M);
	}
	else
	{
		return ((x + (x / static_cast<int>(M) + 2) * static_cast<int>(M)) % M);
	}
}

template <class T>
inline T square(T x)
{
	return x * x;
}

template <class T>
inline T min(T A, T B)
{
	if (A < B)
	{
		return A;
	}
	else
	{
		return B;
	}
}

template <class T>
inline T min(T A, T B, T C)
{
	if (A < B && A < C)
	{
		return A;
	}
	else if (B < C)
	{
		return B;
	}
	else
	{
		return C;
	}
}

template <class T>
inline T max(T A, T B)
{
	if (A > B)
	{
		return A;
	}
	else
	{
		return B;
	}
}

template <class T>
inline T max(T A, T B, T C)
{
	if (A > B && A > C)
	{
		return A;
	}
	else if (B > C)
	{
		return B;
	}
	else
	{
		return C;
	}
}

template <class T>
inline T max(T A, T B, T C, T D)
{
	return max(max(A, B), max(C, D));
}

template <class T>
inline unsigned int maxIndex(T A, T B, T C)
{
	if (A > B && A > C)
	{
		return 0;
	}
	else if (B > C)
	{
		return 1;
	}
	else
	{
		return 2;
	}
}

//! returns the clamped value between min and max
template <class T>
inline T clamp(T x, T pMin, T pMax)
{
	if (x < pMin)
	{
		return pMin;
	}
	if (x > pMax)
	{
		return pMax;
	}
	return x;
}

template <class T>
inline long int floor(T x)
{
	return (long int)std::floor(x);
}

template <class T>
inline long int ceil(T x)
{
	return (long int)std::ceil(x);
}

template <class T>
inline T abs(T x)
{
	//TODO compile check that it's not an unsigned variable type
	if (x < 0)
	{
		return -x;
	}
	return x;
}

template <class T>
inline int round(const T &f)
{
	return (f > (T)0.0) ? (int)floor(f + (T)0.5) : (int)ceil(f - (T)0.5);
}

template <class T>
inline T frac(const T &val)
{
	return (val - floor(val));
}

template <class T>
inline bool isPower2(const T &x)
{
	return (x & (x - 1)) == 0;
}

template <class T>
inline T nextLargeestPow2(T x)
{
	x |= (x >> 1);
	x |= (x >> 2);
	x |= (x >> 4);
	x |= (x >> 8);
	x |= (x >> 16);
	return (x + 1);
}

template <class T>
inline T log2Integer(T x)
{
	T r = 0;
	while (x >>= 1)
	{
		r++;
	}
	return r;
}

//! non-zero 32-bit integer value to compute the log base 10 of
template <class T>
inline T log10Integer(T x)
{
	T r; // result goes here

	const unsigned int PowersOf10[] = {
			1, 10, 100, 1000, 10000, 100000,
			1000000, 10000000, 100000000, 1000000000};

	T t = (log2Integer(x) + 1) * 1233 >> 12; // (use a lg2 method from above)
	r = t - (x < PowersOf10[t]);
	return r;
}

//! returns -1 if negative, 0 if 0, +1 if positive
template <typename T>
inline int sign(T val)
{
	return (T(0) < val) - (val < T(0));
}

//! returns -1 if negative; +1 otherwise (includes 0)
template <typename T>
inline int sgn(T val)
{
	return val < 0 ? -1 : 1;
}

//! solves a^2 + bx + c = 0
template <typename T>
inline void quadraticFormula(T a, T b, T c, T &x0, T &x1)
{
	T tmp = (T)-0.5 * (b + (T)sgn(b) * sqrt(b * b - (T)4 * a * c));
	x0 = tmp / a;
	x1 = c / tmp;
}

//! L2 squared distance metric
template <typename T>
inline T distSqL2(const std::vector<T> &a, const std::vector<T> &b)
{
	T result = T(0.0);
	for (size_t i = 0; i < a.size(); i++)
	{
		T diff = a[i] - b[i];
		result += diff * diff;
	}
	return result;
}

//! L2 distance metric
template <typename T>
inline T distL2(const std::vector<T> &a, const std::vector<T> &b)
{
	return sqrt(distSqL2(a, b));
}

//! L1 distance metric
template <typename T>
inline T distL1(const std::vector<T> &a, const std::vector<T> &b)
{
	T result = T(0.0);
	const size_t size = a.size();
	const T *aPtr = a.data();
	const T *bPtr = b.data();
	for (size_t i = 0; i < size; i++)
	{
		const T diff = aPtr[i] - bPtr[i];
		result += std::abs(diff);
	}
	return result;
}

//! computes sin(phi) and cos(phi)
template <typename T>
inline void sincos(T phi, T &sinPhi, T &cosPhi)
{
	sinPhi = std::sin(phi);
	cosPhi = std::cos(phi);
}
//! computes sin(phi) and cos(phi)
template <typename T>
inline void sincos(T phi, T *sinPhi, T *cosPhi)
{
	sincos(phi, *sinPhi, *cosPhi);
}

//! counts the number of bits in an unsigned integer
inline unsigned int numberOfSetBits(unsigned int i)
{
	i = i - ((i >> 1) & 0x55555555);
	i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
	return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

//! counts the number of bits in an integer
inline int numberOfSetBits(int i)
{
	i = i - ((i >> 1) & 0x55555555);
	i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
	return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

//! generates a random number (uniform distribution)
template <typename T>
inline T randomUniform(T _min, T _max)
{
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::uniform_real_distribution<T> dis(_min, _max);
	return dis(gen);
}
template <>
inline int randomUniform(int _min, int _max)
{
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(_min, _max);
	return dis(gen);
}
template <>
inline unsigned int randomUniform(unsigned int _min, unsigned int _max)
{
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(_min, _max);
	return dis(gen);
}

//! generates a random number (normal distribution)
template <typename T>
inline T randomNormal(T mean, T stddev)
{
	static std::random_device rd;
	static std::mt19937 gen(rd());
	std::normal_distribution<T> dis(mean, stddev);
	return dis(gen);
}

//! flips a coin: 50% chance to retrun true; otherwise false
inline bool randomCointoss()
{
	if (randomUniform(0.0f, 1.0f) > 0.5f)
		return false; //there may be some bias
	else
		return true;
}

} // namespace math

namespace util
{
//! reads from a vector with an index that is clamped to be in-bounds.
template <class T>
const T &clampedRead(const std::vector<T> &v, int index)
{
	if (index < 0)
		return v[0];
	if (index >= v.size())
		return v[v.size() - 1];
	return v[index];
}

template <class T>
bool validIndex(const std::vector<T> &v, int index)
{
	return (index >= 0 && index < v.size());
}

//
// iterator helpers
//
template <class container, class assignFunction>
void fill(container &c, assignFunction func)
{
	int i = 0;
	for (auto &x : c)
	{
		x = func(i++);
	}
}

inline std::string getTimeString()
{
	auto t = std::time(nullptr);
	auto tm = *std::localtime(&t);
#if __GNUC__ && __GNUC__ < 5
#pragma message("std::put_time not available in gcc version < 5")
	return std::string();
#else
	std::stringstream ss;
	ss << std::put_time(&tm, "%Y-%m-%d_%H-%M-%S");
	return ss.str();
#endif
}

inline UINT64 getTime()
{
	UINT64 res = std::time(NULL);
	return res;
}

//
// hashing
//
UINT32 hash32(const BYTE *start, UINT length);
UINT64 hash64(const BYTE *start, UINT length);

template <class T>
inline UINT32 hash32(const T &obj)
{
	return hash32((const BYTE *)&obj, sizeof(T));
}

template <class T>
inline UINT64 hash64(const T &obj)
{
	return hash64((const BYTE *)&obj, sizeof(T));
};

//
// casting
//
inline UINT castBoolToUINT(bool b)
{
	if (b)
	{
		return 1;
	}
	else
	{
		return 0;
	}
};

template <class T>
inline BYTE boundToByte(T value)
{
	if (value < 0)
	{
		value = 0;
	}
	if (value > 255)
	{
		value = 255;
	}
	return (BYTE)value;
};

template <class T>
inline short boundToShort(T value)
{
	if (value <= std::numeric_limits<short>::min())
	{
		return std::numeric_limits<short>::min();
	}
	if (value >= std::numeric_limits<short>::max())
	{
		return std::numeric_limits<short>::max();
	}
	return short(value);
}

//i/o
std::istream &safeGetline(std::istream &is, std::string &t);

//
// file utility
//
bool fileExists(const std::string &filename);
//! returns the file size in bytes
size_t getFileSize(const std::string &filename);
void copyFile(const std::string &sourceFile, const std::string &destFile);
void renameFile(const std::string &oldFilename, const std::string &newFilename);

//
// There are OS-specific functions
//
void messageBox(const char *string);
void messageBox(const std::string &S);
void copyStringToClipboard(const std::string &S);
std::string loadStringFromClipboard();
int runCommand(const std::string &command);
int runCommand(const std::string &executablePath, const std::string &commandLine, bool Blocking);
void makeDirectory(const std::string &directory);
void deleteDirectory(const std::string &directory);
void clearDirectory(const std::string &directory);
void deleteFile(const std::string &file);
bool moveFile(const std::string &currentFile, const std::string &newFile);
bool directoryExists(const std::string &directory);
std::string getWorkingDirectory();								//returns the current working directory
bool setWorkingDirectory(const std::string &dir); //sets a new working directory; returns true if successful
std::string getExecutablePath();									//returns the path of the program executable

inline void runSystemCommand(const std::string &s)
{
	//TODO fix it: this should in theroy call s = util::replace(s, "/", "\\");
	system(s.c_str());
}

//
// Returns the next line in the given file
//
std::vector<std::string> splitPath(const std::string &path);
std::string directoryFromPath(const std::string &path);
std::string fileNameFromPath(const std::string &path);
std::string removeExtensions(const std::string &path);
std::string getNextLine(std::ifstream &file);
std::vector<BYTE> getFileData(const std::string &filename);

//
// Returns the set of all lines in the given file
//
std::vector<std::string> getFileLines(std::ifstream &file, UINT minLineLength = 0);
std::vector<std::string> getFileLines(const std::string &filename, UINT minLineLength = 0);

//! Save lines to file
void writeToFile(const std::string &line, const std::string &filename);
void saveLinesToFile(const std::vector<std::string> &lines, const std::string &filename);

//
// FILE wrappers
//
inline FILE *checkedFOpen(const char *filename, const char *mode)
{
	if (!util::fileExists(filename) && std::string(mode) == "rb")
	{
		std::cout << "File not found: " << filename << std::endl;
		return nullptr;
	}
	FILE *file = fopen(filename, mode);
	MLIB_ASSERT_STR(file != nullptr && !ferror(file), std::string("Failed to open file: ") + std::string(filename));
	return file;
}

inline FILE *checkedFOpen(const std::string &filename, const char *mode)
{
	return checkedFOpen(filename.c_str(), mode);
}

inline void checkedFRead(void *dest, UINT64 elementSize, UINT64 elementCount, FILE *file)
{
	UINT64 elementsRead = fread(dest, elementSize, elementCount, file);
	MLIB_ASSERT_STR(!ferror(file) && elementsRead == elementCount, "fread failed");
}

inline void checkedFWrite(const void *Src, UINT64 elementSize, UINT64 elementCount, FILE *file)
{
	UINT64 elementsWritten = fwrite(Src, elementSize, elementCount, file);
	MLIB_ASSERT_STR(!ferror(file) && elementsWritten == elementCount, "fwrite failed");
}

inline void checkedFSeek(UINT offset, FILE *file)
{
	int result = fseek(file, offset, SEEK_SET);
	MLIB_ASSERT_STR(!ferror(file) && result == 0, "fseek failed");
}

template <class T, class U>
void insert(T &vec, const U &iterable)
{
	vec.insert(iterable.begin(), iterable.end());
}

template <class T, class U>
void push_back(T &vec, const U &iterable)
{
	for (const auto &e : iterable)
		vec.push_back(e);
}

template <class T>
void pop_front(T &vec)
{
	MLIB_ASSERT_STR(vec.size() >= 1, "empty pop");
	for (int i = 0; i < (int)vec.size() - 1; i++)
	{
		vec[i] = vec[i + 1];
	}
	vec.resize((int)vec.size() - 1);
}

template <class T>
bool contains(const std::vector<T> &vec, const T &element)
{
	for (const T &e : vec)
		if (e == element)
			return true;
	return false;
}

template <class T>
int indexOf(const std::vector<T> &vec, const T &element)
{
	for (int index = 0; index < vec.size(); index++)
		if (vec[index] == element)
			return index;
	return -1;
}

//
// String encoding
//
inline std::string encodeBytes(const unsigned char *data, const size_t byteCount)
{
	std::ostringstream os;

	char hexDigits[16] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};

	for (size_t byteIndex = 0; byteIndex < byteCount; byteIndex++)
	{
		unsigned char byte = data[byteIndex];
		os << hexDigits[byte & 0x0f];
		os << hexDigits[(byte & 0xf0) >> 4];
	}

	return os.str();
}

template <class T>
inline std::string encodeBytes(const T &data)
{
	return encodeBytes((unsigned char *)&data, sizeof(data));
}

inline void decodeBytes(const std::string &str, unsigned char *data)
{
	auto digitToValue = [](char c) {
		if (c >= '0' && c <= '9')
			return (int)c - (int)'0';
		else
			return (int)c - (int)'a' + 10;
	};

	for (size_t byteIndex = 0; byteIndex < str.size() / 2; byteIndex++)
	{
		unsigned char c0 = str[byteIndex * 2 + 0];
		unsigned char c1 = str[byteIndex * 2 + 1];

		data[byteIndex] = digitToValue(c0) + (digitToValue(c1) << 4);
	}
}

//Usage: auto mappedVector = map(v, [](int a) { return a * 2.0; });
template <class mapFunction, class T>
auto map(const std::vector<T> &v, mapFunction function) -> std::vector<decltype(function(std::declval<T>()))>
{
	size_t size = v.size();
	std::vector<decltype(function(std::declval<T>()))> result(size);
	for (size_t i = 0; i < size; i++)
		result[i] = function(v[i]);
	return result;
}

//Usage: float result = minValue(v, [](vec3f x) { return x.length(); });
template <class mapFunction, class T>
auto minValue(const std::vector<T> &collection, mapFunction function) -> decltype(function(std::declval<T>()))
{
	auto result = function(*(collection.begin()));
	for (const auto &element : collection)
	{
		auto value = function(element);
		if (value < result)
			result = value;
	}
	return result;
}

template <class T>
size_t maxIndex(const std::vector<T> &collection)
{
	size_t maxIndex = 0;
	for (size_t i = 1; i < collection.size(); i++)
	{
		if (collection[i] > collection[maxIndex])
			maxIndex = i;
	}
	return maxIndex;
}

template <class T>
const T &maxValue(const std::vector<T> &collection)
{
	return collection[maxIndex(collection)];
}

//Usage: float result = minValue(v, [](vec3f x) { return x.length(); });
template <class mapFunction, class T>
auto maxValue(const std::vector<T> &collection, mapFunction function) -> decltype(function(std::declval<T>()))
{
	auto result = function(*(collection.begin()));
	for (const auto &element : collection)
	{
		auto value = function(element);
		if (value > result)
			result = value;
	}
	return result;
}

template <class T>
void removeSwap(std::vector<T> &collection, size_t index)
{
	std::swap(collection[index], collection.back());
	collection.pop_back();
}

template <class T>
int findFirstIndex(const std::vector<T> &collection, const T &value)
{
	int index = 0;
	for (const auto &element : collection)
	{
		if (element == value)
			return index;
		index++;
	}
	return -1;
}

//Usage: size_t result = minValue(v, [](const vec3f &x) { return (x.length() == 0.0f); });
template <class T, class selectFunction>
int findFirstIndex(const std::vector<T> &collection, selectFunction function)
{
	size_t index = 0;
	for (const auto &element : collection)
	{
		if (function(element))
			return index;
	}
	return -1;
}

//Usage: float result = minValue(v, [](vec3f x) { return x.length(); });
template <class mapFunction, class T>
size_t minIndex(const std::vector<T> &collection, mapFunction function)
//float zz(std::vector<int> &collection)
{
	auto minValue = function(*(collection.begin()));
	size_t minIndex = 0, curIndex = 0;
	for (const auto &element : collection)
	{
		auto value = function(element);
		if (value < minValue)
		{
			minValue = value;
			minIndex = curIndex;
		}
		curIndex++;
	}
	return minIndex;
}

//Usage: auto filteredVector = filter(v, [](int a) { return a > 10; });
template <class filterFunction, class T>
auto filter(const std::vector<T> &v, filterFunction function) -> std::vector<T>
{
	std::vector<T> result;
	for (const T &e : v)
		if (function(e))
			result.push_back(e);
	return result;
}

//! uses the <, >  and = operator of the key type
template <typename Iterator, typename T>
inline Iterator binarySearch(Iterator begin, Iterator end, const T &key)
{
	while (begin < end)
	{
		Iterator middle = begin + (std::distance(begin, end) / 2);

		if (*middle == key)
		{ // in that case we exactly found the value
			return middle;
		}
		else if (*middle > key)
		{
			end = middle;
		}
		else
		{
			begin = middle + 1;
		}
	}

	// if the value is not found return the lower interval
	if (begin < end)
	{
		return begin;
	}
	else
	{
		return end;
	}
}

//! uses the <, >, = , and - operator of the key type, returns two iterators surrounding
template <typename Iterator, typename T>
inline std::pair<Iterator, Iterator> binarySearchClosestElement(Iterator begin, Iterator end, const T &key)
{
	Iterator invalid = end;

	while (begin + 1 < end)
	{
		Iterator middle = begin + (std::distance(begin, end) / 2);

		//std::cout << "range (" << *begin << ", " << *middle << ", ";
		//if (end != invalid) std::cout << *end << ")" << std::endl;
		//else std::cout << "END)" << std::endl;

		if (*middle == key)
		{ // in that case we exactly found the value
			return std::make_pair(middle, middle);
		}
		else if (*middle > key)
		{
			end = middle;
		}
		else
		{
			begin = middle;
		}
	}

	return std::make_pair(begin, end); // still possible that begin == key or end == key
}

template <class Matrix, class FloatType>
unsigned int rank(Matrix mat, unsigned int dimension, FloatType eps = (FloatType)0.00001)
{
	const unsigned int n = dimension;

	for (unsigned int k = 0; k < n; k++)
	{ //loop over columns
		for (unsigned int i = k + 1; i < n; i++)
		{ //loop over rows (to zero a specific column)

			if (std::abs(mat(k, k)) <= eps)
			{
				//search for a non-zero element
				for (unsigned int j = k + 1; j < n; j++)
				{
					if (std::abs(mat(j, k) > eps))
					{
						//swap the column
						for (unsigned int l = 0; l < n; l++)
						{
							std::swap(mat(k, l), mat(j, l));
						}
						break;
					}
				}
			}
			if (std::abs(mat(k, k)) > eps)
			{
				FloatType s = mat(i, k) / mat(k, k);
				for (unsigned int j = k; j < n; j++)
				{
					mat(i, j) = mat(i, j) - s * mat(k, j);
				}
			}
		}
	}
	unsigned int r = 0;
	for (unsigned int i = 0; i < n; i++)
	{
		for (unsigned int j = 0; j < n; j++)
		{
			if (std::abs(mat(i, j)) > eps)
			{
				r++;
				break;
			}
		}
	}
	return r;
}

template <class T>
std::vector<T *> toVecPtr(std::vector<T> &v)
{
	std::vector<T *> res;
	res.reserve(v.size());
	for (auto &e : v)
	{
		res.push_back(&e);
	}
	return res;
}

template <class T>
std::vector<const T *> toVecPtr(const std::vector<T> &v)
{
	std::vector<const T *> res;
	res.reserve(v.size());
	for (const auto &e : v)
	{
		res.push_back(&e);
	}
	return res;
}

template <class T>
std::vector<const T *> toVecConstPtr(const std::vector<T> &v)
{
	std::vector<const T *> res;
	res.reserve(v.size());
	for (const auto &e : v)
	{
		res.push_back(&e);
	}
	return res;
}

inline std::string getMLibDir()
{
	return util::directoryFromPath(__FILE__) + "/../../";
}

} // namespace util

// This iterator is used to iterate over a collection as a pair<size_t, T>, where .first is the index and .second is a reference to the element.
// It is mean to replace this:
//
// for(int i = 0; i < v.size(); i++)
// { const T &e = v[i]; ... }
//
// or this:
// int i = 0;
// for(auto &e : v)
/// { ... i++; }
//
// with this:
// for(auto &e : ml::iterate(v))
// { e.index; e.value; ... }
//

template <class T, class U>
struct IndexedIterator
{
	struct Entry
	{
		Entry(size_t _index, U &_value) : index(_index), value(_value) {}
		size_t index;
		U &value;
	};
	IndexedIterator(size_t _index, const T &_iter) : index(_index), iter(_iter) {}

	void operator++(int postfix)
	{
		index++;
		iter++;
	}
	void operator++()
	{
		index++;
		iter++;
	}
	bool operator!=(const IndexedIterator &other)
	{
		return iter != other.iter;
	}
	Entry operator*()
	{
		return Entry(index, *iter);
	}

private:
	size_t index;
	T iter;
};

template <class T, class U>
struct IndexedIteratorConst
{
	struct Entry
	{
		Entry(size_t _index, const U &_value) : index(_index), value(_value) {}
		size_t index;
		const U &value;
	};
	IndexedIteratorConst(size_t _index, const T &_iter) : index(_index), iter(_iter) {}

	void operator++(int postfix)
	{
		index++;
		iter++;
	}
	void operator++()
	{
		index++;
		iter++;
	}
	bool operator!=(const IndexedIteratorConst &other)
	{
		return iter != other.iter;
	}
	Entry operator*()
	{
		return Entry(index, *iter);
	}

private:
	size_t index;
	T iter;
};

template <class T>
struct IndexedIteratorContainer
{
	typedef typename T::value_type ValueType;
	typedef typename T::iterator IteratorType;

	IndexedIteratorContainer(T *_v) : v(_v) {}
	IndexedIterator<IteratorType, ValueType> begin()
	{
		return IndexedIterator<IteratorType, ValueType>(0, v->begin());
	}
	IndexedIterator<IteratorType, ValueType> end()
	{
		return IndexedIterator<IteratorType, ValueType>(0, v->end());
	}

	T *v;
};

template <class T>
struct IndexedIteratorContainerConst
{
	typedef typename T::value_type ValueType;
	typedef typename T::const_iterator IteratorType;

	IndexedIteratorContainerConst(const T *_v) : v(_v) {}
	IndexedIteratorConst<IteratorType, ValueType> begin()
	{
		return IndexedIteratorConst<IteratorType, ValueType>(0, v->begin());
	}
	IndexedIteratorConst<IteratorType, ValueType> end()
	{
		return IndexedIteratorConst<IteratorType, ValueType>(0, v->end());
	}

	const T *v;
};

template <class T>
IndexedIteratorContainer<T> iterate(T &v)
{
	return IndexedIteratorContainer<T>(&v);
};

template <class T>
IndexedIteratorContainerConst<T> iterate(const T &v)
{
	return IndexedIteratorContainerConst<T>(&v);
};

}; // namespace ml

#endif // CORE_UTIL_UTILITY_H_
