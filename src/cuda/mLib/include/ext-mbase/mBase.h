
#include "mLibZLib.h"

namespace ml
{

namespace mBase
{

struct RecordOffset : public BinaryDataSerialize< RecordOffset >
{
    UINT64 recordIndex;
    UINT64 fileIndex;
    UINT64 recordSize;
    UINT64 byteOffset;
};

template<class RecordType>
class Writer
{
public:
    Writer(const std::string &directory)
    {
        finalized = false;
        init(directory);
    }

    ~Writer()
    {
        finalize();
    }

    void init(const std::string &_directory)
    {
        util::makeDirectory(_directory);
        directory = _directory;
        activeFile = nullptr;
        activeFileIndex = 0;
        activeFileSize = 0;
    }

    void updateActiveFile()
    {
        const int maxFileSize = 1024 * 1024 * 256;
        if (activeFileSize >= maxFileSize && activeFile)
        {
            fclose(activeFile);
            activeFile = nullptr;
            activeFileSize = 0;
            activeFileIndex++;
        }

        if (activeFile == nullptr)
        {
            activeFile = util::checkedFOpen(directory + util::zeroPad(activeFileIndex, 3), "wb");
        }
    }

    void addRecord(const RecordType &record)
    {
        updateActiveFile();

        BinaryDataStreamVector out;
        out << record;
        const std::vector<BYTE> compressedData = ZLibWrapper::CompressStreamToMemory(out.getData(), true);

        RecordOffset newRecord;
        newRecord.recordIndex = records.size();
        newRecord.byteOffset = activeFileSize;
        newRecord.fileIndex = activeFileIndex;
        newRecord.recordSize = compressedData.size();
        records.push_back(newRecord);

        util::checkedFWrite(compressedData.data(), sizeof(BYTE), compressedData.size(), activeFile);
        activeFileSize += compressedData.size();
    }

    void finalize()
    {
        if (finalized) return;
        finalized = true;
        if(activeFile) fclose(activeFile);
        util::serializeToFile(directory + "records.dat", records);
        cout << "Saved " << records.size() << " records" << endl;
    }

private:
    std::string directory;
    UINT64 activeFileSize;
    int activeFileIndex;
    FILE *activeFile;
    std::vector<RecordOffset> records;
    bool finalized;
};

template<class RecordType>
class Reader
{
public:
    Reader() {}
    Reader(const std::string &_directory, size_t _cacheSize, int maxRecords)
    {
        init(_directory, _cacheSize, maxRecords);
    }

    ~Reader()
    {
        terminateThread = true;
        if (decompThread.joinable()) {
            decompThread.join();
        }
    }

    void init(const std::string &_directory, size_t _cacheSize, int maxRecords)
    {
        epoch = 0;
        directory = _directory;
        cacheSize = _cacheSize;
        terminateThread = false;
        activeRecordIndex = 0;
        util::deserializeFromFile(directory + "records.dat", records);
        std::cout << "Loaded " << records.size() << " records" << std::endl;

        if (maxRecords >= 0 && records.size() >= maxRecords)
        {
            std::cout << "Truncated to " << maxRecords << " records" << std::endl;
            records.resize(maxRecords);
        }

        startDecompressBackgroundThread();
    }

    void readNextRecord(RecordType &result)
    {
        //std::cout << "RNR Epoch " << epoch << " cache: " << cache.size() << std::endl;
        while (1) {
            if (cache.size() > 0) {
                cacheMutex.lock();
                result = std::move(cache.front());
                cache.pop_front();
                cacheMutex.unlock();
                return;
            }
            else {
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        }
    }

private:

    void readRecordInternal(UINT64 recordIndex, RecordType &result)
    {
        const RecordOffset &offset = records[recordIndex];
        const std::string filename = directory + util::zeroPad(offset.fileIndex, 3);

        FILE* file = fopen(filename.c_str(), "rb");
        if (file == nullptr || ferror(file))
        {
            std::cout << "Failed to open file: " << filename << std::endl;
            return;
        }

        fseek(file, (long)offset.byteOffset, SEEK_SET);

        if (cacheStorage.size() < offset.recordSize)
            cacheStorage.resize(offset.recordSize);
        fread(cacheStorage.data(), offset.recordSize, 1, file);

        fclose(file);

        std::vector<BYTE> uncompressedData = ZLibWrapper::DecompressStreamFromMemory(cacheStorage);

        BinaryDataStreamVector out;
        out.setData(std::move(uncompressedData));
        out >> result;
    }

    void readNextRecordInternal(RecordType &result)
    {
        //std::cout << "RNRI activeRecordIndex " << activeRecordIndex << std::endl;
        readRecordInternal(activeRecordIndex, result);
        activeRecordIndex++;
        if (activeRecordIndex == records.size())
        {
            if(epoch <= 100) std::cout << "Epoch " << epoch++ << " finished" << std::endl;
            activeRecordIndex = 0;
        }
    }

    void startDecompressBackgroundThread() {
        decompThread = std::thread(decompressStub, this);
    }

    static void decompressStub(Reader<RecordType>* reader) {
        reader->decompressLoop();
    }

    void decompressLoop() {
        while (1) {
            if (terminateThread) break;
            
            if (cache.size() < cacheSize) {	//need to fill the cache
                RecordType newRecord;
                readNextRecordInternal(newRecord);

                cacheMutex.lock();
                cache.push_back(std::move(newRecord));
                //std::cout << "New cache size: " << cache.size() << std::endl;
                cacheMutex.unlock();
            }
            else {
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        }
    }

    std::string directory;
    std::vector<RecordOffset> records;
    UINT64 activeRecordIndex;

    size_t cacheSize;
    std::list<RecordType> cache;
    std::thread decompThread;
    std::mutex cacheMutex;
    bool terminateThread;
    std::vector<BYTE> cacheStorage;
    int epoch;
};

}

}