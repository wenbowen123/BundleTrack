
#ifndef CORE_UTIL_BINARYDATASTREAM_H_
#define CORE_UTIL_BINARYDATASTREAM_H_

#include "core-util/binaryDataCompressor.h"
#include "core-util/binaryDataBuffer.h"

namespace ml
{

template<class BinaryDataBuffer, class BinaryDataCompressor>
class BinaryDataStream {
public:
	BinaryDataStream() {}
	BinaryDataStream(const std::string& filename, bool clearStream) {
		open(filename, clearStream);
	}
	BinaryDataStream(const std::string& filename, typename BinaryDataBuffer::Mode mode) {
		open(filename, mode);
	}
	~BinaryDataStream() {
		close();
	}

	//! only required for file streams: clear means write-only and delete file; otherwise it's read only
	void open(const std::string& filename, bool clearStream) {
		typename BinaryDataBuffer::Mode mode = BinaryDataBuffer::Mode::no_flag;
		if (clearStream) {
			mode |= BinaryDataBuffer::Mode::clear_flag | BinaryDataBuffer::Mode::write_flag;
		}
		else {
			mode |= BinaryDataBuffer::Mode::read_flag;
		}
		m_dataBuffer.open(filename, mode);
	}
	void open(const std::string& filename, typename BinaryDataBuffer::Mode mode) {
		m_dataBuffer.open(filename, mode);
	}

	void close() {
		m_dataBuffer.close();
	}

	template <class T>
	void writeData(const T& t) {
		writeData((const BYTE*)&t, sizeof(T));
	}

	//start compression after that byte size (only if compression is enabled)
#define COMPRESSION_THRESHOLD_ 1024
	void writeData(const BYTE* t, size_t size) {
		const bool useCompression = !std::is_same<BinaryDataCompressorNone, BinaryDataCompressor>::value;
		if (useCompression && size > COMPRESSION_THRESHOLD_) {
			std::vector<BYTE> compressedT;
			//ZLibWrapper::CompressStreamToMemory(t, size, compressedT, false);
			m_dataCompressor.compressStreamToMemory(t, size, compressedT);
			UINT64 compressedSize = compressedT.size();
			m_dataBuffer.writeData((const BYTE*)&compressedSize, sizeof(compressedSize));
			m_dataBuffer.writeData(&compressedT[0], compressedSize);
		} else 	{
			m_dataBuffer.writeData(t, size);
		}

	}

	template <class T>
	void readData(T& result) {
		readData((BYTE*)&result, sizeof(T));
	}

	void readData(BYTE* result, size_t size) {
		const bool useCompression = !std::is_same<BinaryDataCompressorNone, BinaryDataCompressor>::value;
		if (useCompression && size > COMPRESSION_THRESHOLD_) {
			UINT64 compressedSize;
			m_dataBuffer.readData((BYTE*)&compressedSize, sizeof(UINT64));
			std::vector<BYTE> compressedT;	compressedT.resize(compressedSize);
			m_dataBuffer.readData(&compressedT[0], compressedSize);
			//ZLibWrapper::DecompressStreamFromMemory(&compressedT[0], compressedSize, result, size);
			m_dataCompressor.decompressStreamFromMemory(&compressedT[0], compressedSize, result, size);
		} else
		{
			m_dataBuffer.readData(result, size);
		}
	}

	//! clears the read offset: copies all data to the front of the data array and frees all unnecessary memory
	void clearReadOffset() {
		m_dataBuffer.clearReadOffset();
	}

	//! destroys all the data in the stream (DELETES ALL DATA!)
	void clear() {
		m_dataBuffer.clear();
	}

	void reserve(size_t size) {
		m_dataBuffer.reserve(size);
	}

	void flush() {
		m_dataBuffer.flushBufferStream();
	}

	//! saves the stream to file; does not affect current data
	void saveToFile(const std::string &filename) {
		m_dataBuffer.saveToFile(filename);
	}

	//! loads a binary stream from file; destorys all previous data in the stream
	void loadFromFile(const std::string& filename) {
		m_dataBuffer.loadFromFile(filename);
	}



    const std::vector<BYTE>& getData() const {
        return m_dataBuffer.getData();
    }

    void setData(std::vector<BYTE> &&data) {
        m_dataBuffer.setData(std::move(data));
    }

private:
	BinaryDataBuffer		m_dataBuffer;
	BinaryDataCompressor	m_dataCompressor;
};

typedef BinaryDataStream<BinaryDataBufferMemory, BinaryDataCompressorNone> BinaryDataStreamVector;
typedef BinaryDataStream<BinaryDataBufferFile, BinaryDataCompressorNone> BinaryDataStreamFile;
//typedef BinaryDataStream<BinaryDataBufferVector, BinaryDataCompressorDefault> BinaryDataStreamCompressedVector; (see zlib for instance)
//typedef BinaryDataStream<BinaryDataBufferFile, BinaryDataCompressorDefault> BinaryDataStreamCompressedFile; (see zlib for instance)

//////////////////////////////////////////////////////
/////////write stream operators for base types ///////
//////////////////////////////////////////////////////

//cannot overload via template since it is not supposed to work for complex types

template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, UINT64 i) {
	s.writeData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, bool i) {
    s.writeData(i);
    return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, int i) {
	s.writeData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, unsigned int i) {
	s.writeData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, short i) {
	s.writeData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, unsigned short i) {
	s.writeData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, char i) {
	s.writeData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, unsigned char i) {
	s.writeData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, float i) {
	s.writeData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, double i) {
	s.writeData(i);
	return s;
}

template<class BinaryDataBuffer, class BinaryDataCompressor, class T>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const std::vector<T>& v) {
	s << (UINT64)v.size();
	if (std::is_pod<T>::value) {
		s.writeData((const BYTE*)v.data(), sizeof(T)*v.size());
	}
	else {
		s.reserve(sizeof(T)*v.size());
		for (size_t i = 0; i < v.size(); i++) {
			s << v[i];
		}
	}
	return s;
}

template<class BinaryDataBuffer, class BinaryDataCompressor, class T>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const std::list<T>& l) {
	s << (UINT64)l.size();
	s.reserve(sizeof(T)*l.size());
	for (typename std::list<T>::const_iterator iter = l.begin(); iter != l.end(); iter++) {
		s << *l;
	}
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const std::string& str) {
	s << (UINT64)str.size();
	s.writeData((const BYTE*)str.data(), sizeof(char)*str.size());
	return s;
}

template<class BinaryDataBuffer, class BinaryDataCompressor, class K, class T>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const std::unordered_map<K,T>& m) {
	s << m.size();
	s << m.max_load_factor();
	for (auto iter = m.begin(); iter != m.end(); iter++) {
		s << iter->first << iter->second;
	}
	return s;
}





//////////////////////////////////////////////////////
/////////read stream operators for base types ///////
//////////////////////////////////////////////////////

//cannot overload via template since it is not supposed to work for complex types
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, UINT64& i) {
	s.readData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, int& i) {
	s.readData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, bool& i) {
    s.readData(i);
    return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, unsigned int& i) {
	s.readData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, short& i) {
	s.readData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, unsigned short& i) {
	s.readData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, char& i) {
	s.readData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, unsigned char& i) {
	s.readData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, float& i) {
	s.readData(i);
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, double& i) {
	s.readData(i);
	return s;
}

template<class BinaryDataBuffer, class BinaryDataCompressor, class T>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, std::vector<T>& v) {
	UINT64 size;
	s >> size;
	v.resize(size);
	if (std::is_pod<T>::value) {
		s.readData((BYTE*)v.data(), sizeof(T)*v.size());
	}
	else {
		for (size_t i = 0; i < v.size(); i++) {
			s >> v[i];
		}
	}
	return s;
}

template<class BinaryDataBuffer, class BinaryDataCompressor, class T>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, std::list<T>& l) {
	UINT64 size;
	s >> size;
	l.clear();
	for (size_t i = 0; i < size; i++) {
		T curr;
		s >> curr;
		l.push_back(l);
	}
	return s;
}
template<class BinaryDataBuffer, class BinaryDataCompressor>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, std::string& str) {
	UINT64 size;
	s >> size;
	str.resize(size);
	s.readData((BYTE*)str.data(), sizeof(char)*str.size());
	return s;
}

template<class BinaryDataBuffer, class BinaryDataCompressor, class K, class T>
inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, std::unordered_map<K,T>& m) {
	m.clear();
	size_t size;	float maxLoadFactor;
	s >> size >> maxLoadFactor;
	m.max_load_factor(maxLoadFactor);
	for (size_t i = 0; i < size; i++) {
		K first;
		s >> first;
		T &second = m[first];
		s >> second;
	}
	return s;
}

namespace util
{
    template<class T>
    void serializeToFile(const std::string &filename, const T &o)
    {
        BinaryDataStreamFile out(filename, true);
        out << o;
        out.close();
    }

    template<class T, class U>
    void serializeToFile(const std::string &filename, const T &o0, const U &o1)
    {
        BinaryDataStreamFile out(filename, true);
        out << o0 << o1;
        out.close();
    }

    template<class T>
    void deserializeFromFile(const std::string &filename, T &o)
    {
        BinaryDataStreamFile in(filename, false);
        in >> o;
        //in.closeStream();change by guan
         in.close();
    }

    template<class T, class U>
    void deserializeFromFile(const std::string &filename, T &o0, U &o1)
    {
        BinaryDataStreamFile in(filename, false);
        in >> o0 >> o1;
        //in.closeStream();change by guan
        in.close();
    }
	//
	// TODO: these cannot be forward declared without zlib, and should be moved into the mlib-zlib header.
	//
    /*template<class T>
    void serializeToFileCompressed(const std::string &filename, const T &o)
    {
        BinaryDataStreamZLibFile out(filename, true);
        out << o;
        out.closeStream();
    }

    template<class T, class U>
    void serializeToFileCompressed(const std::string &filename, const T &o0, const U &o1)
    {
        BinaryDataStreamZLibFile out(filename, true);
        out << o0 << o1;
        out.closeStream();
    }

    template<class T>
    void deserializeFromFileCompressed(const std::string &filename, T &o)
    {
        BinaryDataStreamZLibFile in(filename, false);
        in >> o;
        in.closeStream();
    }

    template<class T, class U>
    void deserializeFromFileCompressed(const std::string &filename, T &o0, U &o1)
    {
        BinaryDataStreamZLibFile in(filename, false);
        in >> o0 >> o1;
        in.closeStream();
    }*/
}

}  // namespace ml

#endif  // CORE_UTIL_BINARYDATASTREAM_H_
