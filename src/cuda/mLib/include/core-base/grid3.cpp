
#ifndef CORE_BASE_GRID3D_INL_H_
#define CORE_BASE_GRID3D_INL_H_

namespace ml
{

	template <class T> Grid3<T>::Grid3()
	{
		m_dimX = m_dimY = m_dimZ = 0;
		m_data = nullptr;
	}

	template <class T> Grid3<T>::Grid3(size_t dimX, size_t dimY, size_t dimZ)
	{
		m_dimX = dimX;
		m_dimY = dimY;
		m_dimZ = dimZ;
		m_data = new T[dimX * dimY * dimZ];
	}

	template <class T> Grid3<T>::Grid3(size_t dimX, size_t dimY, size_t dimZ, const T& value)
	{
		m_dimX = dimX;
		m_dimY = dimY;
		m_dimZ = dimZ;
		m_data = new T[dimX * dimY * dimZ];
		setValues(value);
	}

	template <class T> Grid3<T>::Grid3(const Grid3<T>& grid)
	{
		m_dimX = grid.m_dimX;
		m_dimY = grid.m_dimY;
		m_dimZ = grid.m_dimZ;

		const size_t totalEntries = getNumElements();
		m_data = new T[totalEntries];
		for (size_t i = 0; i < totalEntries; i++) {
			m_data[i] = grid.m_data[i];
		}
	}

	template <class T> Grid3<T>::Grid3(Grid3<T> &&grid)
	{
		m_dimX = m_dimY = m_dimZ = 0;
		m_data = nullptr;
		swap(*this, grid);
	}

	template <class T> Grid3<T>::Grid3(size_t dimX, size_t dimY, size_t dimZ, const std::function< T(size_t, size_t, size_t) > &fillFunction)
	{
		m_dimX = dimX;
		m_dimY = dimY;
		m_dimY = dimZ;
		m_data = new T[dimX * dimY * dimZ];
		fill(fillFunction);
	}

	template <class T> Grid3<T>::~Grid3()
	{
		SAFE_DELETE_ARRAY(m_data);
	}


	template <class T> Grid3<T>& Grid3<T>::operator=(const Grid3<T> &grid)
	{
		SAFE_DELETE_ARRAY(m_data);
		m_dimX = grid.m_dimX;
		m_dimY = grid.m_dimY;
		m_dimZ = grid.m_dimZ;

		const size_t totalEntries = getNumElements();
		m_data = new T[totalEntries];
		for (size_t i = 0; i < totalEntries; i++) {
			m_data[i] = grid.m_data[i];
		}

		return *this;
	}

	template <class T> Grid3<T>& Grid3<T>::operator=(Grid3<T> &&grid)
	{
		swap(*this, grid);
		return *this;
	}

	template <class T> void Grid3<T>::allocate(size_t dimX, size_t dimY, size_t dimZ)
	{
		if (dimX == 0 || dimY == 0 || dimZ == 0) {
			SAFE_DELETE_ARRAY(m_data);
		}
		else if (getDimX() != dimX || getDimY() != dimY || getDimZ() != dimZ) {
			m_dimX = dimX;
			m_dimY = dimY;
			m_dimZ = dimZ;
			SAFE_DELETE_ARRAY(m_data);
			m_data = new T[dimX * dimY * dimZ];
		}
	}

	template <class T> void Grid3<T>::allocate(size_t dimX, size_t dimY, size_t dimZ, const T& value)
	{
		allocate(dimX, dimY, dimZ);
		setValues(value);
	}

	template <class T> void Grid3<T>::setValues(const T &value)
	{
		const size_t totalEntries = getNumElements();
		for (size_t i = 0; i < totalEntries; i++) m_data[i] = value;
	}

	template <class T> void Grid3<T>::fill(const std::function<T(size_t x, size_t y, size_t z)> &fillFunction)
	{
		for (size_t z = 0; z < m_dimZ; z++)
			for (size_t y = 0; y < m_dimY; y++)
				for (size_t x = 0; x < m_dimX; x++)
					(*this)(x, y, z) = fillFunction(x, y, z);
	}



	template <class T> vec3ul Grid3<T>::getMaxIndex() const
	{
		vec3ul maxIndex(0, 0, 0);
		const T *maxValue = m_data;
		for (size_t z = 0; z < m_dimZ; z++)
			for (size_t y = 0; y < m_dimY; y++)
				for (size_t x = 0; x < m_dimX; x++)
				{
					const T* curValue = &(*this)(x, y, z);
					if (*curValue > *maxValue)
					{
						maxIndex = vec3ul(x, y, z);
						maxValue = curValue;
					}
				}
		return maxIndex;
	}

	template <class T> const T& Grid3<T>::getMaxValue() const
	{
		vec3ul index = getMaxIndex();
		return (*this)(index);
	}

	template <class T> vec3ul Grid3<T>::getMinIndex() const
	{
		vec3ul minIndex(0, 0, 0);
		const T *minValue = &m_data[0];
		for (size_t z = 0; z < m_dimZ; z++)
			for (size_t y = 0; y < m_dimY; y++)
				for (size_t x = 0; x < m_dimX; x++)
				{
					const T* curValue = &(*this)(x, y, z);
					if (*curValue < *minValue)
					{
						minIndex = vec3ul(x, y, z);
						minValue = curValue;
					}
				}
			return minIndex;
	}

	template <class T> const T& Grid3<T>::getMinValue() const
	{
		vec3ul index = getMinIndex();
		return (*this)(index);
	}

}  // namespace ml

#endif  // CORE_BASE_GRID3D_INL_H_