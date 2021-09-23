
#ifndef EXT_ZLIB_ZLIBWRAPPER_H_
#define EXT_ZLIB_ZLIBWRAPPER_H_

namespace ml
{

class ZLibWrapper
{
public:

    static std::vector<BYTE> CompressStreamToMemory(const std::vector<BYTE> &decompressedStream, bool writeHeader);

    static std::vector<BYTE> CompressStreamToMemory(const BYTE *decompressedStream, UINT64 decompressedStreamLength, bool writeHeader);

    static void CompressStreamToMemory(const std::vector<BYTE> &decompressedStream, std::vector<BYTE> &compressedStream, bool writeHeader);

    static void CompressStreamToMemory(const BYTE *decompressedStream, UINT64 decompressedStreamLength, std::vector<BYTE> &compressedStream, bool writeHeader);
    
    static std::vector<BYTE> DecompressStreamFromMemory(const std::vector<BYTE> &compressedStream);

    static void DecompressStreamFromMemory(const std::vector<BYTE> &compressedStream, std::vector<BYTE> &decompressedStream);

    static void DecompressStreamFromMemory(const BYTE *compressedStream, UINT64 compressedStreamLength, BYTE *decompressedStream, UINT64 decompressedStreamLength);

};


//! interface to compress data
class BinaryDataCompressorZLib : public BinaryDataCompressorInterface {
public:
	void compressStreamToMemory(const BYTE *decompressedStream, UINT64 decompressedStreamLength, std::vector<BYTE> &compressedStream) const {
		ZLibWrapper::CompressStreamToMemory(decompressedStream, decompressedStreamLength, compressedStream, false);
	}

	void decompressStreamFromMemory(const BYTE *compressedStream, UINT64 compressedStreamLength, BYTE *decompressedStream, UINT64 decompressedStreamLength) const {
		ZLibWrapper::DecompressStreamFromMemory(compressedStream, compressedStreamLength, decompressedStream, decompressedStreamLength);
	}

	std::string getTypename() const {
		return "zlib compression";
	}
};

typedef BinaryDataStream<BinaryDataBufferMemory, BinaryDataCompressorZLib> BinaryDataStreamZLibVector;
typedef BinaryDataStream<BinaryDataBufferFile, BinaryDataCompressorZLib> BinaryDataStreamZLibFile;

}  // namespace ml

#endif  // EXT_ZLIB_ZLIBWRAPPER_H_
