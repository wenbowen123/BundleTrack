
#ifndef CORE_BASE_GRID3D_H_
#define CORE_BASE_GRID3D_H_

namespace ml
{

	template <class T> class Grid3
	{
	public:
		Grid3();
		Grid3(size_t dimX, size_t dimY, size_t dimZ);
		Grid3(size_t dimX, size_t dimY, size_t dimZ, const T &value);
		Grid3(const vec3ul& dim) : Grid3(dim.x, dim.y, dim.z) {}
		Grid3(const vec3ul& dim, const T& value) : Grid3(dim.x, dim.y, dim.z, value) {}

		Grid3(const Grid3<T> &grid);
		Grid3(Grid3<T> &&grid);
		Grid3(size_t dimX, size_t dimY, size_t dimZ, const std::function< T(size_t x, size_t y, size_t z) > &fillFunction);

		~Grid3();

		//! adl swap
		friend void swap(Grid3& a, Grid3& b) {
			std::swap(a.m_dimX, b.m_dimX);
			std::swap(a.m_dimY, b.m_dimY);
			std::swap(a.m_dimZ, b.m_dimZ);
			std::swap(a.m_data, b.m_data);
		}

		Grid3<T>& operator=(const Grid3<T>& grid);
		Grid3<T>& operator=(Grid3<T>&& grid);

		void allocate(size_t dimX, size_t dimY, size_t dimZ);
		void allocate(size_t dimX, size_t dimY, size_t dimZ, const T &value);
		void allocate(const vec3ul& dim) {						allocate(dim.x, dim.y, dim.z);				}
		void allocate(const vec3ul& dim, const T& value) {		allocate(dim.x, dim.y, dim.z, value);		}

		//
		// Accessors
		//
		inline T& operator() (size_t x, size_t y, size_t z)	{
			MLIB_ASSERT(x < getDimX() && y < getDimY() && z < getDimZ());
			return m_data[getDimX()*getDimY()*z + getDimX()*y + x];
		}

		inline const T& operator() (size_t x, size_t y, size_t z) const	{
			MLIB_ASSERT(x < getDimX() && y < getDimY() && z < getDimZ());
			return m_data[getDimX()*getDimY()*z + getDimX()*y + x];
		}

		inline T& operator() (const vec3ul& coord)	{
			return (*this)(coord.x, coord.y, coord.z);
		}

		inline const T& operator() (const vec3ul& coord) const	{
			return (*this)(coord.x, coord.y, coord.z);
		}

		inline size_t getDimX() const	{
			return m_dimX;
		}
		inline size_t getDimY() const	{
			return m_dimY;
		}
		inline size_t getDimZ() const	{
			return m_dimZ;
		}

		inline vec3ul getDimensions() const {
			return vec3ul(getDimX(), getDimY(), getDimZ());
		}

		size_t getNumElements() const {
			return m_dimX * m_dimY * m_dimZ;
		}

		inline bool isSquare() const	{
			return (m_dimX == m_dimY && m_dimY == m_dimZ);
		}
		inline T* getData()	{
			return m_data;
		}
		inline const T* getData() const	{
			return m_data;
		}

		inline Grid3<T>& operator += (const Grid3<T>& right)
		{
			MLIB_ASSERT(getDimensions() == right.getDimensions());
			const size_t numElements = getNumElements();
			for (size_t i = 0; i < numElements; i++) {
				m_data[i] += right.m_data[i];
			}
			return *this;
		}
		inline Grid3<T>& operator += (T value)
		{
			const size_t numElements = getNumElements();
			for (size_t i = 0; i < numElements; i++) {
				m_data[i] += value;
			}
			return *this;
		}
		inline Grid3<T>& operator *= (T value)
		{
			const size_t numElements = getNumElements();
			for (size_t i = 0; i < numElements; i++) {
				m_data[i] *= value;
			}
			return *this;
		}

		inline Grid3<T> operator * (T value)
		{
			Grid3<T> result(m_dimX, m_dimY, m_dimZ);
			const size_t numElements = getNumElements();
			for (size_t i = 0; i < numElements; i++) {
				result.m_data[i] = m_data[i] * value;
			}
			return result;
		}

		//
		// Modifiers
		//
		void setValues(const T &value);

		void fill(const std::function<T(size_t x, size_t y, size_t z)> &fillFunction);

		//
		// Query
		//
		inline bool isValidCoordinate(size_t x, size_t y, size_t z) const
		{
			return (x < m_dimX && y < m_dimY && z < m_dimZ);
		}
        inline bool isValidCoordinate(const vec3ul& coord) const
        {
            return (coord.x < m_dimX && coord.y < m_dimY && coord.z < m_dimZ);
        }

		vec3ul getMaxIndex() const;
		const T& getMaxValue() const;
		vec3ul getMinIndex() const;
		const T& getMinValue() const;

		std::string toString(bool verbose = true) const {
			std::stringstream ss;
			ss << "grid dim: " << getDimensions() << "\n";
			if (verbose) {
				for (size_t z = 0; z < m_dimZ; z++) {
					ss << "slice " << z << std::endl;
					for (size_t y = 0; y < m_dimY; y++) {
						ss << "\t";
						for (size_t x = 0; x < m_dimX; x++) {
							ss << (*this)(x, y, z) << " ";
						}
						ss << "\n";
					}
				}
			}
			return ss.str();
		}

		//
		// Grid3 iterators
		//
		struct iteratorEntry
		{
			iteratorEntry(size_t _x, size_t _y, size_t _z, T &_value)
				: x(_x), y(_y), z(_z), value(_value)
			{

			}
			size_t x;
			size_t y;
			size_t z;
			T &value;
		};

		struct constIteratorEntry
		{
			constIteratorEntry(size_t _x, size_t _y, size_t _z, const T &_value)
				: x(_x), y(_y), z(_z), value(_value)
			{

			}
			size_t x;
			size_t y;
			size_t z;
			const T &value;
		};


		struct iterator
		{
			iterator(Grid3<T> *_grid)
			{
				x = 0;
				y = 0;
				z = 0;
				grid = _grid;
			}
			iterator(const iterator &i)
			{
				x = i.x;
				y = i.y;
				z = i.z;
				grid = i.grid;
			}
			~iterator() {}
			iterator& operator=(const iterator &i)
			{
				x = i.x;
				y = i.y;
				z = i.z;
				grid = i.grid;
				return *this;
			}
			iterator& operator++()
			{
				x++;
				if (x == grid->getDimX())
				{
					x = 0;
					y++;
					if (y == grid->getDimY())
					{
						y = 0;
						z++;
						if (z == grid->getDimZ())
						{
							grid = NULL;
						}
					}
				}
				return *this;
			}
			iteratorEntry operator* () const
			{
				return iteratorEntry(x, y, z, (*grid)(x, y, z));
			}

			bool operator != (const iterator &i) const
			{
				return i.grid != grid;
			}

			template<class U>
			friend void swap(iterator &a, iterator &b);

			size_t x, y, z;

		private:
			Grid3<T> *grid;
		};

		struct constIterator
		{
			constIterator(const Grid3<T> *_grid)
			{
				x = 0;
				y = 0;
				z = 0;
				grid = _grid;
			}
			constIterator(const constIterator &i)
			{
				x = i.x;
				y = i.y;
				z = i.z;
				grid = i.grid;
			}
			~constIterator() {}
			constIterator& operator=(const constIterator &i)
			{
				x = i.x;
				y = i.y;
				z = i.z;
				grid = i.grid;
				return *this;
			}
			constIterator& operator++()
			{
				x++;
				if (x == grid->getDimX())
				{
					x = 0;
					y++;
					if (y == grid->getDimY())
					{
						y = 0;
						z++;
						if (z == grid->getDimZ())
						{
							grid = NULL;
						}
					}
				}
				return *this;
			}
			constIteratorEntry operator* () const
			{
				return constIteratorEntry(x, y, z, (*grid)(x, y, z));
			}

			bool operator != (const constIterator &i) const
			{
				return i.grid != grid;
			}

			template<class U>
			friend void swap(const constIterator &a, const constIterator &b);

			size_t x, y, z;

		private:
			const Grid3<T> *grid;
		};


        iterator begin()
        {
            return iterator(this);
        }

        iterator end()
        {
            return iterator(NULL);
        }

        constIterator begin() const
        {
            return constIterator(this);
        }

        constIterator end() const
        {
            return constIterator(NULL);
        }

	protected:
		T* m_data;
		size_t m_dimX, m_dimY, m_dimZ;
	};

	template <class T> inline bool operator == (const Grid3<T> &a, const Grid3<T> &b)
	{
		if (a.getDimensions() != b.getDimensions()) return false;
		const size_t totalEntries = a.getNumElements();
		for (size_t i = 0; i < totalEntries; i++) {
			if (a.getData()[i] != b.getData()[i])	return false;
		}
		return true;
	}

	template <class T> inline bool operator != (const Grid3<T> &a, const Grid3<T> &b)
	{
		return !(a == b);
	}

	//! writes to a stream
	template <class T>
	inline std::ostream& operator<<(std::ostream& s, const Grid3<T>& g)
	{
		s << g.toString();
		return s;
	}

	//! serialization (output)
	template<class BinaryDataBuffer, class BinaryDataCompressor, class T>
	inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const Grid3<T>& g) {
		
		s << (UINT64)g.getDimX() << (UINT64)g.getDimY() << (UINT64)g.getDimZ();		

		if (std::is_pod<T>::value) {
			s.writeData((const BYTE*)g.getData(), sizeof(T)*g.getNumElements());
		}
		else {
			const size_t numElements = g.getNumElements();
			s.reserve(sizeof(T) * numElements);
			for (size_t i = 0; i < numElements; i++) {
				s << g.getData()[i];
			}
		}
		return s;
	}

	//! serialization (input)
	template<class BinaryDataBuffer, class BinaryDataCompressor, class T>
	inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, Grid3<T>& g) {
		
		UINT64 dimX, dimY, dimZ;
		s >> dimX >> dimY >> dimZ;
		g.allocate(dimX, dimY, dimZ);

		if (std::is_pod<T>::value) {
			s.readData((BYTE*)g.getData(), sizeof(T)*g.getNumElements());
		}
		else { 
			const size_t numElements = g.getNumElements();
			for (size_t i = 0; i < numElements; i++) {
				s >> g.getData()[i];
			}
		}

		return s;
	}

	typedef Grid3<float> Grid3f;
	typedef Grid3<double> Grid3d;
	typedef Grid3<int> Grid3i;
	typedef Grid3<unsigned int> Grid3ui;
	typedef Grid3<unsigned char> Grid3uc;

}  // namespace ml

#include "grid3.cpp"

#endif  // CORE_BASE_GRID3D_H_
