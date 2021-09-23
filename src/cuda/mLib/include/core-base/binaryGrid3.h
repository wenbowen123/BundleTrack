
#ifndef CORE_BASE_BINARY_GRID3D_H_
#define CORE_BASE_BINARY_GRID3D_H_

namespace ml {


	class BinaryGrid3 {
	public:
		BinaryGrid3() {
			m_dimZ = m_dimY = m_dimX = 0;
			m_data = nullptr;
		}

		BinaryGrid3(size_t dim) {
			m_data = nullptr;
			allocate(dim, dim, dim);
			clearVoxels();
		}

		BinaryGrid3(size_t width, size_t height, size_t depth) {
			m_data = nullptr;
			allocate(width, height, depth);
			clearVoxels();
		}

		BinaryGrid3(const vec3ui& dim) {
			m_data = nullptr;
			allocate(dim);
			clearVoxels();
		}

		BinaryGrid3(const BinaryGrid3& other) {
			m_data = nullptr;
			if (other.m_data != nullptr) {
				allocate(other.m_dimX, other.m_dimY, other.m_dimZ);
				memcpy(m_data, other.m_data, getNumUInts()*sizeof(unsigned int));
			}
			else {
				m_data = nullptr;
				m_dimX = other.m_dimX;
				m_dimY = other.m_dimY;
				m_dimZ = other.m_dimZ;
			}
		}

		BinaryGrid3(BinaryGrid3&& other) {
			m_dimX = m_dimY = m_dimZ = 0;
			m_data = nullptr;
			swap(*this, other);
		}

		~BinaryGrid3() {
			SAFE_DELETE_ARRAY(m_data);
		}

		//! adl swap
		friend void swap(BinaryGrid3& a, BinaryGrid3& b) {
			std::swap(a.m_dimX, b.m_dimX);
			std::swap(a.m_dimY, b.m_dimY);
			std::swap(a.m_dimZ, b.m_dimZ);
			std::swap(a.m_data, b.m_data);
		}


		inline void allocate(size_t width, size_t height, size_t depth) {
			if (width == 0 || height == 0 || depth == 0) {
				SAFE_DELETE_ARRAY(m_data);
				return;
			}
			else {
				SAFE_DELETE_ARRAY(m_data);
				m_dimX = width;
				m_dimY = height;
				m_dimZ = depth;

				size_t dataSize = getNumUInts();
				m_data = new unsigned int[dataSize];
			}
		}

		inline void allocate(const vec3ul& dim) {
			allocate(dim.x, dim.y, dim.z);
		}

		inline BinaryGrid3& operator=(const BinaryGrid3& other) {
			if (this != &other) {
				if (other.m_data != nullptr) {
					allocate(other.m_dimX, other.m_dimY, other.m_dimZ);
					memcpy(m_data, other.m_data, getNumUInts()*sizeof(unsigned int));
				}
				else {
					SAFE_DELETE_ARRAY(m_data);
					m_data = nullptr;
					m_dimX = other.m_dimX;
					m_dimY = other.m_dimY;
					m_dimZ = other.m_dimZ;
				}
			}
			return *this;
		}

		inline BinaryGrid3& operator=(BinaryGrid3&& other) {
			swap(*this, other);
			return *this;
		}

		inline bool operator==(const BinaryGrid3& other) const {
			if (m_dimX != other.m_dimX ||
				m_dimY != other.m_dimY ||
				m_dimZ != other.m_dimZ)	return false;

			size_t numUInts = getNumUInts();
			for (size_t i = 0; i < numUInts; i++) {
				if (m_data[i] != other.m_data[i])	return false;
			}

			return true;
		}

		inline bool operator!=(const BinaryGrid3& other) const {
			return !(*this == other);
		}

		//! clears all voxels
		inline void clearVoxels() {
			size_t numUInts = getNumUInts();
			for (size_t i = 0; i < numUInts; i++) {
				m_data[i] = 0;
			}
		}

		inline bool isVoxelSet(size_t x, size_t y, size_t z) const {
			size_t linIdx = m_dimX*m_dimY*z + m_dimX*y + x;
			size_t baseIdx = linIdx / bitsPerUInt;
			size_t localIdx = linIdx % bitsPerUInt;
			return (m_data[baseIdx] & (1 << localIdx)) != 0;
		}

		inline bool isVoxelSet(const vec3ul& v) const {
			return isVoxelSet(v.x, v.y, v.z);
		}

		inline void setVoxel(size_t x, size_t y, size_t z) {
			size_t linIdx = m_dimX*m_dimY*z + m_dimX*y + x;
			size_t baseIdx = linIdx / bitsPerUInt;
			size_t localIdx = linIdx % bitsPerUInt;
			m_data[baseIdx] |= (1 << localIdx);
		}

		inline void setVoxel(const vec3ul& v) {
			setVoxel(v.x, v.y, v.z);
		}

		inline void clearVoxel(size_t x, size_t y, size_t z) {
			size_t linIdx = m_dimX*m_dimY*z + m_dimX*y + x;
			size_t baseIdx = linIdx / bitsPerUInt;
			size_t localIdx = linIdx % bitsPerUInt;
			m_data[baseIdx] &= ~(1 << localIdx);
		}

		inline void clearVoxel(const vec3ul& v) {
			clearVoxel(v.x, v.y, v.z);
		}

		inline void toggleVoxel(size_t x, size_t y, size_t z) {
			size_t linIdx = m_dimX*m_dimY*z + m_dimX*y + x;
			size_t baseIdx = linIdx / bitsPerUInt;
			size_t localIdx = linIdx % bitsPerUInt;
			m_data[baseIdx] ^= (1 << localIdx);
		}

		inline void toggleVoxel(const vec3ul& v) {
			toggleVoxel(v.x, v.y, v.z);
		}

		inline void toggleVoxelAndBehindRow(size_t x, size_t y, size_t z) {
			for (size_t i = x; i < m_dimX; i++) {
				toggleVoxel(i, y, z);
			}
		}
		inline void toggleVoxelAndBehindRow(const vec3ul& v) {
			toggleVoxelAndBehindRow(v.x, v.y, v.z);
		}

		inline void toggleVoxelAndBehindSlice(size_t x, size_t y, size_t z) {
			for (size_t i = z; i < m_dimZ; i++) {
				toggleVoxel(x, y, i);
			}
		}
		inline void toggleVoxelAndBehindSlice(const vec3ul& v) {
			toggleVoxelAndBehindSlice(v.x, v.y, v.z);
		}

		inline size_t getDimX() const {
			return m_dimX;
		}
		inline size_t getDimY() const {
			return m_dimY;
		}
		inline size_t getDimZ() const {
			return m_dimZ;
		}

		inline vec3ul getDimensions() const {
			return vec3ul(getDimX(), getDimY(), getDimZ());
		}

		inline size_t getNumElements() const {
			return m_dimX*m_dimY*m_dimZ;
		}

		inline bool isValidCoordinate(size_t x, size_t y, size_t z) const
		{
			return (x < m_dimX && y < m_dimY && z < m_dimZ);
		}

		inline bool isValidCoordinate(const vec3ul& v) const
		{
			return isValidCoordinate(v.x, v.y, v.z);
		}

		inline size_t getNumOccupiedEntries() const {
			size_t numOccupiedEntries = 0;
			size_t numUInts = getNumUInts();
			for (size_t i = 0; i < numUInts; i++) {
				numOccupiedEntries += math::numberOfSetBits(m_data[i]);
			}
			return numOccupiedEntries;
		}

		inline const unsigned int* getData() const {
			return m_data;
		}

		inline unsigned int* getData() {
			return m_data;
		}

		std::string toString(bool verbose = true) const {
			std::stringstream ss;

			ss << "grid dim: " << getDimensions() << "\n";

			if (verbose) {
				for (size_t z = 0; z < m_dimZ; z++) {
					ss << "slice " << z << std::endl;
					for (size_t y = 0; y < m_dimY; y++) {
						ss << "\t";
						for (size_t x = 0; x < m_dimX; x++) {
							if (isVoxelSet(x, y, z)) {
								ss << "1";
							}
							else {
								ss << "0";
							}

						}
						ss << "\n";
					}
				}
			}
			return ss.str();
		}

		template<class BinaryDataBuffer, class BinaryDataCompressor> friend
			BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& 
			operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const BinaryGrid3& g);

		template<class BinaryDataBuffer, class BinaryDataCompressor> friend
			BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>&
			operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, BinaryGrid3& g);
	private:

#ifdef _WIN32
        // boost archive serialization functions
		friend class boost::serialization::access;
		template <class Archive>
		void save(Archive& ar, const unsigned int version) const {
			ar << m_dimX << m_dimY << m_dimZ << boost::serialization::make_array(m_data, getNumUInts());
		}
		template<class Archive>
		void load(Archive& ar, const unsigned int version) {
			ar >> m_dimX >> m_dimY >> m_dimZ;
			allocate(m_dimX, m_dimY, m_dimZ);
			ar >> boost::serialization::make_array(m_data, getNumUInts());
		}
		template<class Archive>
		void serialize(Archive &ar, const unsigned int version) {
			boost::serialization::split_member(ar, *this, version);
		}
#endif

		inline size_t getNumUInts() const {
			size_t numEntries = getNumElements();
			return (numEntries + bitsPerUInt - 1) / bitsPerUInt;
		}

		static const unsigned int bitsPerUInt = sizeof(unsigned int) * 8;
		size_t			m_dimX, m_dimY, m_dimZ;
		unsigned int*	m_data;
	};

	//! writes to a stream
	inline std::ostream& operator<<(std::ostream& s, const BinaryGrid3& g)
	{
		s << g.toString();
		return s;
	}
	 
	//! serialization (output)
	template<class BinaryDataBuffer, class BinaryDataCompressor>
	inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const BinaryGrid3& g) {
		s << (UINT64)g.getDimX() << (UINT64)g.getDimY() << (UINT64)g.getDimZ();
		size_t dataSize = g.getNumUInts();
		s.writeData((const BYTE*)g.getData(), dataSize*sizeof(unsigned int));
		return s;
	}

	//! serialization (input)
	template<class BinaryDataBuffer, class BinaryDataCompressor>
	inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, BinaryGrid3& g) {
		UINT64 dimX, dimY, dimZ;
		s >> dimX >> dimY >> dimZ;
		g.allocate(dimX, dimY, dimZ);
		size_t dataSize = g.getNumUInts();
		s.readData((BYTE*)g.getData(), dataSize*sizeof(unsigned int));
		return s;
	}

}

#endif