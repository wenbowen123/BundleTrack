
#ifndef CORE_UTIL_BINARYDATABUFFER_H_
#define CORE_UTIL_BINARYDATABUFFER_H_

#include <vector>
#include <list>
#include <fstream>
#include <string>
#include "core-util/utility.h"



namespace ml
{

/////////////////////////////////////////////////////////////
// BinaryDataBuffers (class used by BINARY_DATA_STREAM)    //
/////////////////////////////////////////////////////////////

// abstract interface class
class BinaryDataBuffer {
public:

	enum Mode {
		no_flag = 0,
		read_flag = 1 << 0,
		write_flag = 1 << 1,
		clear_flag = 1 << 2
	};

	//! opens the buffer (default is read/write)
	virtual void open(const std::string& filename, Mode mode) = 0;
	//! closes the buffer
	virtual void close() = 0;
	//! writes data to the buffer
	virtual void writeData(const BYTE* t, size_t size) = 0;
	//! reads data from the buffer
	virtual void readData(BYTE* result, size_t size) = 0;
	//! clears the buffers (DELETES ALL DATA WITHIN IT!)
	virtual void clear() = 0;
	//! reserves memory for potential writes
	virtual void reserve(size_t size) = 0;
	//! flushes all operations
	virtual void flush() = 0;
	//! saves the buffer to a file
	virtual void loadFromFile(const std::string& filename) = 0;
	//! loads the content of a file to the buffer
	virtual void saveToFile(const std::string& filename) = 0;
protected:
	Mode m_mode;
};

inline BinaryDataBuffer::Mode operator~ (BinaryDataBuffer::Mode a) { return (BinaryDataBuffer::Mode)~(int)a; }
inline BinaryDataBuffer::Mode operator| (BinaryDataBuffer::Mode a, BinaryDataBuffer::Mode b) { return (BinaryDataBuffer::Mode)((int)a | (int)b); }
inline BinaryDataBuffer::Mode operator& (BinaryDataBuffer::Mode a, BinaryDataBuffer::Mode b) { return (BinaryDataBuffer::Mode)((int)a & (int)b); }
inline BinaryDataBuffer::Mode operator^ (BinaryDataBuffer::Mode a, BinaryDataBuffer::Mode b) { return (BinaryDataBuffer::Mode)((int)a ^ (int)b); }
inline BinaryDataBuffer::Mode& operator|= (BinaryDataBuffer::Mode& a, BinaryDataBuffer::Mode b) { return (BinaryDataBuffer::Mode&)((int&)a |= (int)b); }
inline BinaryDataBuffer::Mode& operator&= (BinaryDataBuffer::Mode& a, BinaryDataBuffer::Mode b) { return (BinaryDataBuffer::Mode&)((int&)a &= (int)b); }
inline BinaryDataBuffer::Mode& operator^= (BinaryDataBuffer::Mode& a, BinaryDataBuffer::Mode b) { return (BinaryDataBuffer::Mode&)((int&)a ^= (int)b); }


/////////////////////////////////////////////////////////////
// BinaryDataBuffers (one for file, one for system memory) //
/////////////////////////////////////////////////////////////

class BinaryDataBufferFile : public BinaryDataBuffer {
public:
	BinaryDataBufferFile() {
		m_readOffset = 0;
		m_fileSize = 0;
		m_mode = Mode::no_flag;
	}

	~BinaryDataBufferFile() {
		closeFileStream();
		if (m_readOffset == 0 && m_fileSize == 0) {
			util::deleteFile(m_filename);
		}
	}

	void open(const std::string& filename, Mode mode = Mode::read_flag | Mode::write_flag) {
		m_filename = filename;
		m_mode = mode;
		if (mode & Mode::clear_flag)	util::deleteFile(m_filename);
		closeFileStream();
		openFileStream();
	}

	void close() {
		closeFileStream();
	}


	void writeData(const BYTE* t, size_t size) {
		MLIB_ASSERT((m_mode & Mode::write_flag) > 0);
		//std::cout << "tellp() " << m_fileStream.tellp() << std::endl;
		m_fileStream.seekp(m_fileSize);	//always append at the end
		m_fileStream.write((char*)t, size);
		m_fileSize += size;
	}

	void readData(BYTE* result, size_t size) {
		MLIB_ASSERT((m_mode & Mode::read_flag) > 0);
		//std::cout << "tellg() " << m_fileStream.tellg() << std::endl;
		//assert(m_readOffset + size <= m_fileSize);
		if (m_readOffset + size > m_fileSize) throw MLIB_EXCEPTION("invalid read; probably wrong file name (" + m_filename + ")?");
		m_fileStream.seekg(m_readOffset);
		m_fileStream.read((char*)result, size);
		m_readOffset += size;
	}


	//! destroys all the data in the stream
	void clear() {
		closeFileStream();
		util::deleteFile(m_filename);
		openFileStream();
		MLIB_ASSERT_STR(m_fileSize == 0, "file buffer not cleared correctly");
		m_readOffset = 0;
	}

	void clearReadOffset() {
		size_t len = m_fileSize - m_readOffset;
		if (len == 0) {  //if there is no data left, clear the buffer
			clear();
		} else {
			std::vector<BYTE> oldData;
			copyDataToMemory(oldData);

			closeFileStream();
			util::deleteFile(m_filename);
			openFileStream();
			m_fileStream.write((char*)&oldData[0], oldData.size());
			m_readOffset = 0;
			MLIB_ASSERT_STR(m_fileSize == oldData.size(), "");
		}
	}

	void reserve(size_t size) {
		//TODO implement a reasonable reserver (it would make sense to create an empty file of that size so no further allocation needs to happen on the drive)
		return;
	}

	//! flushes the stream
	void flush() {
		m_fileStream.flush();
	}

	void saveToFile(const std::string& filename) {
		std::vector<BYTE> oldData;
		copyDataToMemory(oldData);

		std::ofstream output(filename, std::ios::binary);
		output.write((char*)&oldData[0], sizeof(BYTE)*oldData.size());
		if (!output.is_open())	throw MLIB_EXCEPTION(filename);
		output.close();
		return;
	}

	//! loads a binary stream from file; destorys all previous data in the stream
	void loadFromFile(const std::string& filename) {
		m_fileStream.close();

		size_t inputFileSize = util::getFileSize(filename);

		BYTE* data = new BYTE[inputFileSize];
		std::ifstream input(filename, std::ios::binary);
		if (!input.is_open())	throw MLIB_EXCEPTION(filename);
		input.read((char*)data, sizeof(BYTE)*inputFileSize);
		input.close();

		clear();	//clear the old values
		m_fileStream.write((char*)data, sizeof(BYTE)*inputFileSize);
		MLIB_ASSERT(m_fileSize == inputFileSize);
		m_readOffset = 0;
	}
private:

	//! reads all the 'active' file data to system memory
	void copyDataToMemory(std::vector<BYTE>& data) {
		size_t len = m_fileSize - m_readOffset;
		data.resize(len);
		m_fileStream.seekg(m_readOffset);
		m_fileStream.read((char*)&data[0], sizeof(BYTE)*len);
	}

	//! opens the file stream
	void openFileStream() {
		if (m_fileStream.is_open())	m_fileStream.close();

		if (!util::fileExists(m_filename))	m_fileSize = 0;	//in case there was no file before
		else m_fileSize = util::getFileSize(m_filename);

		//std::ios_base::open_mode m = std::ios::binary;//change by guan
		//if (m_mode & Mode::read_flag) m |= std::ios::in;
		//if (m_mode & Mode::write_flag) m |= std::ios::out;

		//m_fileStream.open(m_filename, m);	//if the file does not exist, it will fail with the ::in flag

                std::ios_base::open_mode m = std::ios::binary;
                if (m_mode & Mode::read_flag) m_fileStream.open(m_filename, std::ios::binary | std::ios::in);
                if (m_mode & Mode::write_flag) m_fileStream.open(m_filename, std::ios::binary | std::ios::out);

		if (!m_fileStream.is_open() || !m_fileStream.good()) throw MLIB_EXCEPTION("could not open file: " + m_filename);
	}

	//! closes the file stream; data is automatically saved...
	void closeFileStream() {
		if (m_fileStream.is_open())	m_fileStream.close();
	}

	std::string		m_filename;
	std::fstream	m_fileStream;
	size_t			m_readOffset;
	size_t			m_fileSize;
};






class BinaryDataBufferMemory : public BinaryDataBuffer {
public:
	BinaryDataBufferMemory() {
		m_readOffset = 0;
		m_mode = Mode::read_flag | Mode::write_flag;
	}

	//TODO: this is pretty clunky and it is confusing this constructor is statically-compilable for BinaryDataBufferMemory.
	void open(const std::string& filename, Mode mode = Mode::read_flag | Mode::write_flag) {
		MLIB_ASSERT(false);
		//dummy just needed for file stream
		return;
	}
	void close() {
		MLIB_ASSERT(false);
		//dummy just needed for file stream
		return;
	}

	void writeData(const BYTE* t, size_t size) {
		MLIB_ASSERT((m_mode & Mode::write_flag) > 0);
		size_t basePtr = m_data.size();
		m_data.resize(basePtr + size);
		memcpy(&m_data[0] + basePtr, t, size);
	}

	void readData(BYTE* result, size_t size) {
		MLIB_ASSERT((m_mode & Mode::read_flag) > 0);
		MLIB_ASSERT(m_readOffset + size <= m_data.size());

		memcpy(result, &m_data[0] + m_readOffset, size);
		m_readOffset += size;

		//free memory if we reached the end of the stream
		if (m_readOffset == m_data.size()) {
			m_data.resize(0);
			m_readOffset = 0;
		}
	}


	//! destroys all the data in the stream
	void clear() {
		m_data.clear();
		m_readOffset = 0;
	}

	void clearReadOffset() {
		size_t len = m_data.size() - m_readOffset;
		for (unsigned int i = 0; i < len; i++) {
			m_data[i] = m_data[i + m_readOffset];
		}
		m_data.resize(len);
		m_readOffset = 0;
	}

	void reserve(size_t size) {
		if (size > m_data.size())
			m_data.reserve(size);
	}

	//! since all writes are immediate, there is nothing to do
	void flush() {
		return;
	}

	void saveToFile(const std::string &filename) {
		std::ofstream output(filename, std::ios::binary);
		output.write((char*)&m_data[0], sizeof(BYTE)*m_data.size());
		if (!output.is_open())	throw MLIB_EXCEPTION(filename);
		output.close();
	}


	//! returns the file size; if the file cannot be opened returns -1 (e.g., the file does not exist)
	size_t getFileSizeInBytes(const std::string &filename) {
		std::ifstream file(filename, std::ios::binary | std::ios::ate);
		if (!file.is_open())	return (size_t)-1;
		size_t size = file.tellg();
		file.close();
		return size;
	}

	//! loads a binary stream from file; destorys all previous data in the stream
	void loadFromFile(const std::string &filename) {
		size_t inputFileSize = getFileSizeInBytes(filename);
		m_data.resize(inputFileSize);
		std::ifstream input(filename, std::ios::binary);
		if (!input.is_open())	throw MLIB_EXCEPTION(filename);
		input.read((char*)&m_data[0], sizeof(BYTE)*inputFileSize);
		input.close();
		m_readOffset = 0;
	}

	// are these functions supposed to exist?
    const std::vector<BYTE>& getData() const {
        return m_data;
    }
    void setData(std::vector<BYTE> &&data) {
        m_data = data;
        m_readOffset = 0;
    }

private:
	std::vector<BYTE>	m_data;
	size_t				m_readOffset;
};

}  // namespace ml

#endif  // CORE_UTIL_BINARYDATABUFFER_H_
