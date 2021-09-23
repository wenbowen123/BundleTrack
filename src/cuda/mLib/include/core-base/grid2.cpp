
#ifndef CORE_BASE_GRID2D_H_INL_
#define CORE_BASE_GRID2D_H_INL_

namespace ml
{

	template <class T> Grid2<T>::Grid2()
	{
		m_dimX = m_dimY = 0;
		m_data = nullptr;
	}

	template <class T> Grid2<T>::Grid2(size_t dimX, size_t dimY)
	{
		m_dimX = dimX;
		m_dimY = dimY;
		m_data = new T[dimX * dimY];
	}

	template <class T> Grid2<T>::Grid2(size_t dimX, size_t dimY, const T& value)
	{
		m_dimX = dimX;
		m_dimY = dimY;
		m_data = new T[dimX * dimY];
		setValues(value);
	}

	template <class T> Grid2<T>::Grid2(const Grid2<T>& grid)
	{
		m_dimX = grid.m_dimX;
		m_dimY = grid.m_dimY;
		
		const size_t totalEntries = getNumElements();
		m_data = new T[totalEntries];
		for (size_t i = 0; i < totalEntries; i++)
			m_data[i] = grid.m_data[i];
	}

	template <class T> Grid2<T>::Grid2(Grid2<T> &&grid)
	{
		m_dimX = m_dimY = 0;
		m_data = nullptr;
		swap(*this, grid);
	}

	template <class T> Grid2<T>::Grid2(size_t dimX, size_t dimY, const std::function< T(size_t, size_t) > &fillFunction)
	{
		m_dimX = dimX;
		m_dimY = dimY;
		m_data = new T[dimX * dimY];
		fill(fillFunction);
	}

	template <class T> Grid2<T>::~Grid2()
	{
		SAFE_DELETE_ARRAY(m_data);
	}

	template <class T> Grid2<T>& Grid2<T>::operator=(const Grid2<T> &grid)
	{
		SAFE_DELETE_ARRAY(m_data)
		m_dimX = grid.m_dimX;
		m_dimY = grid.m_dimY;

		const size_t totalEntries = m_dimX * m_dimY;
		m_data = new T[totalEntries];
		for (size_t i = 0; i < totalEntries; i++)
			m_data[i] = grid.m_data[i];

		return *this;
	}

	template <class T> Grid2<T>& Grid2<T>::operator=(Grid2<T> &&grid)
	{
		swap(*this, grid);
		return *this;
	}


	template <class T> void Grid2<T>::allocate(size_t dimX, size_t dimY)
	{
		if (dimX == 0 || dimY == 0) {
			SAFE_DELETE_ARRAY(m_data);
		}
		else if (getDimX() != dimX || getDimY() != dimY) {
			m_dimX = dimX;
			m_dimY = dimY;
			SAFE_DELETE_ARRAY(m_data);
			m_data = new T[dimX * dimY];
		}
	}

	template <class T> void Grid2<T>::allocate(size_t dimX, size_t dimY, const T& value)
	{
		allocate(dimX, dimY);
		setValues(value);
	}

	template <class T> void Grid2<T>::setValues(const T &value)
	{
		const size_t totalEntries = getNumElements();
		for (size_t i = 0; i < totalEntries; i++)	m_data[i] = value;
	}

	template <class T> void Grid2<T>::fill(const std::function< T(size_t, size_t) > &fillFunction)
	{
		for (size_t y = 0; y < m_dimY; y++)
			for (size_t x = 0; x < m_dimX; x++)
				(*this)(x, y) = fillFunction(x, y);
	}

	template <class T> vec2ul Grid2<T>::getMaxIndex() const
	{
		vec2ul maxIndex(0, 0);
		const T *maxValue = m_data;
		for (size_t y = 0; y < m_dimY; y++)
			for (size_t x = 0; x < m_dimX; x++)
			{				
				const T* curValue = &(*this)(x, y);
				if (*curValue > *maxValue)
				{
					maxIndex = vec2ul(x, y);
					maxValue = curValue;
				}
			}
		return maxIndex;
	}

	template <class T> const T& Grid2<T>::getMaxValue() const
	{
		vec2ul index = getMaxIndex();
		return (*this)(index);
	}

	template <class T> vec2ul Grid2<T>::getMinIndex() const
	{
		vec2ul minIndex(0, 0);
		const T *minValue = &m_data[0];
		for (size_t y = 0; y < m_dimY; y++)
			for (size_t x = 0; x < m_dimX; x++)
			{
				const T* curValue = &(*this)(x, y);
				if (*curValue < *minValue)
				{
					minIndex = vec2ul(x, y);
					minValue = curValue;
				}
			}
		return minIndex;
	}

	template <class T> const T& Grid2<T>::getMinValue() const
	{
		vec2ul index = getMinIndex();
		return (*this)(index);
	}

}  // namespace ml

#endif  // CORE_BASE_GRID2D_H_INL_
