
#ifndef CORE_BASE_GRID2D_H_
#define CORE_BASE_GRID2D_H_

namespace ml
{
	template <class T> class Grid2
	{
	public:
		Grid2();
		Grid2(size_t dimX, size_t dimY);
		Grid2(size_t dimX, size_t dimY, const T &value);
		Grid2(const vec2ul& dim) : Grid2(dim.x, dim.y) {}
		Grid2(const vec2ul& dim, const T& value) : Grid2(dim.x, dim.y, value) {}

		Grid2(const Grid2<T> &grid);
		Grid2(Grid2<T> &&grid);
		Grid2(size_t dimX, size_t dimY, const std::function< T(size_t x, size_t y) > &fillFunction);

		~Grid2();

		//! adl swap
		friend void swap(Grid2& a, Grid2& b) {
			std::swap(a.m_dimX, b.m_dimX);
			std::swap(a.m_dimY, b.m_dimY);
			std::swap(a.m_data, b.m_data);
		}

		Grid2<T>& operator=(const Grid2<T> &grid);
		Grid2<T>& operator=(Grid2<T> &&grid);

		void allocate(size_t dimX, size_t dimY);
		void allocate(size_t dimX, size_t dimY, const T& value);
		void allocate(const vec2ul& dim) { allocate(dim.x, dim.y); }
		void allocate(const vec2ul& dim, const T& value) { allocate(dim.x, dim.y, value); }

		//
		// Accessors
		//
		inline T& operator() (size_t x, size_t y)	{
			MLIB_ASSERT(x < m_dimX && y < m_dimY);
			return m_data[getDimX()*y + x];
		}
		inline const T& operator() (size_t x, size_t y) const	{
			MLIB_ASSERT(x < m_dimX && y < m_dimY);
			return m_data[getDimX()*y + x];
		}

		inline T& operator() (const vec2ul& coord)	{
			return (*this)(coord.x, coord.y);
		}

		inline const T& operator() (const vec2ul& coord) const	{
			return (*this)(coord.x, coord.y);
		}

		inline size_t getDimX() const	{
			return m_dimX;
		}

		inline size_t getDimY() const	{
			return m_dimY;
		}

		inline vec2ul getDimensions() const {
			return vec2ul(m_dimX, m_dimY);
		}

		inline size_t getNumElements() const	{
			return m_dimX * m_dimY;
		}

		inline bool isSquare() const	{
			return (m_dimX == m_dimY);
		}

		inline T* getData()	{
			return m_data;
		}

		inline const T* getData() const	{
			return m_data;
		}


		inline Grid2<T>& operator += (const Grid2<T>& right) {
			MLIB_ASSERT(getDimensions() == right.getDimensions());
			const size_t numElements = getNumElements();
			for (size_t i = 0; i < numElements; i++) {
				m_data[i] += right.m_data[i];
			}
			return *this;
		}

		inline Grid2<T>& operator += (T value) {
			const size_t numElements = getNumElements();
			for (size_t i = 0; i < numElements; i++)
				m_data[i] += value;
			return *this;
		}

		inline Grid2<T>& operator *= (T value)
		{
			const size_t numElements = getNumElements();
			for (size_t i = 0; i < numElements; i++) {
				m_data[i] *= value;
			}
			return *this;
		}

		inline Grid2<T> operator * (T value)
		{
			Grid2<T> result(m_dimX, m_dimY);
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

		void fill(const std::function< T(size_t x, size_t y) > &fillFunction);

		std::vector<T> toStdVector() const
		{
			std::vector<T> result;
			for (size_t i = 0; i < m_dimX * m_dimY; i++)
				result.push_back(i);
			return result;
		}

		//
		// Query
		//
		inline bool isValidCoordinate(size_t x, size_t y) const
		{
			return (x < m_dimX && y < m_dimY);
		}
        inline bool isValidCoordinate(vec2ul& coord) const
        {
			return (coord.x < m_dimX && coord.y < m_dimY);
        }

		vec2ul getMaxIndex() const;
		const T& getMaxValue() const;
		vec2ul getMinIndex() const;
		const T& getMinValue() const;

		std::string toString(bool verbose = true) const {
			std::stringstream ss;
			ss << "grid dim: " << getDimensions() << "\n";
			if (verbose) {
				for (size_t y = 0; y < m_dimY; y++) {
					ss << "\t";
					for (size_t x = 0; x < m_dimX; x++) {
						ss << (*this)(x, y) << " ";
					}
					ss << "\n";
				}
			}
			return ss.str();
		}

		//
		// TODO: rename
		//
		void setRow(size_t row, const std::vector<T>& values)
		{
			for (size_t col = 0; col < m_dimY; col++) m_data[row * m_dimX + col] = values[col];
		}

		void setCol(size_t col, const std::vector<T>& values)
		{
			for (size_t row = 0; row < m_dimX; row++) m_data[row * m_dimX + col] = values[row];
		}

		std::vector<T> getRow(size_t y) const
		{
			std::vector<T> result(m_dimX);
			const T *CPtr = m_data;
			for (size_t x = 0; x < m_dimX; x++)
			{
				result[x] = CPtr[y * m_dimX + x];
			}
			return result;
		}

		std::vector<T> getCol(size_t x) const
		{
			std::vector<T> result(m_dimY);
			const T *CPtr = m_data;
			for (size_t y = 0; y < m_dimY; y++)
			{
				result[y] = CPtr[y * m_dimX + x];
			}
			return result;
		}


		//
		// Grid2 iterators
		//
		struct iteratorEntry
		{
			iteratorEntry(size_t _x, size_t _y, T &_value)
				: x(_x), y(_y), value(_value)
			{

			}
			size_t x;
			size_t y;
			T &value;
		};

		struct constIteratorEntry
		{
			constIteratorEntry(size_t _x, size_t _y, const T &_value)
				: x(_x), y(_y), value(_value)
			{

			}
			size_t x;
			size_t y;
			const T &value;
		};


		struct iterator
		{
			iterator(Grid2<T> *_grid)
			{
				x = 0;
				y = 0;
				grid = _grid;
			}
			iterator(const iterator &i)
			{
				x = i.x;
				y = i.y;
				grid = i.grid;
			}
			~iterator() {}
			iterator& operator=(const iterator &i)
			{
				x = i.x;
				y = i.y;
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
						grid = nullptr;
					}
				}
				return *this;
			}
			iteratorEntry operator* () const
			{
				return iteratorEntry(x, y, (*grid)(x, y));
			}

			bool operator != (const iterator &i) const
			{
				return i.grid != grid;
			}

			template<class U>
			friend void swap(iterator &a, iterator &b);

			size_t x, y;

		private:
			Grid2<T> *grid;
		};

		struct constIterator
		{
			constIterator(const Grid2<T> *_grid)
			{
				x = 0;
				y = 0;
				grid = _grid;
			}
			constIterator(const constIterator &i)
			{
				x = i.x;
				y = i.y;
				grid = i.grid;
			}
			~constIterator() {}
			constIterator& operator=(const constIterator &i)
			{
				x = i.x;
				y = i.y;
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
						grid = nullptr;
					}
				}
				return *this;
			}
			constIteratorEntry operator* () const
			{
				return constIteratorEntry(x, y, (*grid)(x, y));
			}

			bool operator != (const constIterator &i) const
			{
				return i.grid != grid;
			}

			template<class U>
			friend void swap(const constIterator &a, const constIterator &b);

			size_t x, y;

		private:
			const Grid2<T> *grid;
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
		size_t m_dimX, m_dimY;
	};

	template <class T> inline bool operator == (const Grid2<T> &a, const Grid2<T> &b)
	{
		if (a.getDimensions() != b.getDimensions()) return false;
		const size_t totalEntries = a.getNumElements();
		for (size_t i = 0; i < totalEntries; i++) {
			if (a.getData()[i] != b.getData()[i])	return false;
		}
		return true;
	}

	template <class T> inline bool operator != (const Grid2<T> &a, const Grid2<T> &b)
	{
		return !(a == b);
	}

	//! writes to a stream
	template <class T>
	inline std::ostream& operator<<(std::ostream& s, const Grid2<T>& g)
	{
		s << g.toString();
		return s;
	}

	//! serialization (output)
	template<class BinaryDataBuffer, class BinaryDataCompressor, class T>
	inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator<<(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, const Grid2<T>& g) {
		
		s << (UINT64)g.getDimX() << (UINT64)g.getDimY();
		
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
	inline BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& operator>>(BinaryDataStream<BinaryDataBuffer, BinaryDataCompressor>& s, Grid2<T>& g) {
		
		UINT64 dimX, dimY;
		s >> dimX >> dimY;
		g.allocate(dimX, dimY);

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

	typedef Grid2<float> Grid2f;
    typedef Grid2<int> Grid2i;
	typedef Grid2<double> Grid2d;
    typedef Grid2<unsigned char> Grid2uc;

}

#include "grid2.cpp"

#endif  // CORE_BASE_GRID2D_H_
