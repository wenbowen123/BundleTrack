#ifndef CORE_BASE_DISTANCE_FIELD3_H_
#define CORE_BASE_DISTANCE_FIELD3_H_

#include <core-base/grid3.h>

namespace ml {

	template<class FloatType>
	class DistanceField3 : public Grid3 < FloatType > {
	public:

		DistanceField3() {}
		DistanceField3(size_t dimX, size_t dimY, size_t dimZ) : Grid3<FloatType>(dimX, dimY, dimZ) {}
		DistanceField3(const vec3ul& dim) : Grid3<FloatType>(dim.x, dim.y, dim.z) {}
		DistanceField3(const BinaryGrid3& grid, FloatType trunc = std::numeric_limits<FloatType>::infinity()) : Grid3<FloatType>(grid.getDimX(), grid.getDimY(), grid.getDimZ())
		{
			generateFromBinaryGrid(grid, trunc);
		}

		FloatType getTruncation() const
		{
			return m_truncation;
		}

		BinaryGrid3 computeBinaryGrid(float distThres = 0.0001f) const {

			BinaryGrid3 grid(this->getDimX(), this->getDimY(), this->getDimZ());
			for (size_t z = 0; z < this->getDimZ(); z++) {
				for (size_t y = 0; y < this->getDimY(); y++) {
					for (size_t x = 0; x < this->getDimX(); x++) {
						if (std::abs((*this)(x, y, z)) <= distThres) {
							grid.setVoxel(x, y, z);
						}
					}
				}
			}
			return grid;
		}

		void generateFromBinaryGrid(const BinaryGrid3& grid, FloatType trunc = std::numeric_limits<FloatType>::infinity()) {
			this->allocate(grid.getDimX(), grid.getDimY(), grid.getDimZ());

			m_truncation = trunc;

			//the simple variant appears to be much faster
			generateFromBinaryGridSimple(grid, trunc);
			//generateFromBinaryGridQueue(grid);
		}

		//! computes the distance when projecting all grid points into the distance field (returns distance and valid comparisons)
		std::pair<FloatType, size_t> evalDist(const BinaryGrid3& grid, const Matrix4x4<FloatType>& gridToDF, bool squaredSum = false) const {

			FloatType dist = (FloatType)0;
			size_t numComparisons = 0;

			Matrix4x4<FloatType> DFToGrid = gridToDF.getInverse();


			BoundingBox3<int> bbBox;
			bbBox.include(DFToGrid*vec3<FloatType>((FloatType)0, (FloatType)0, (FloatType)0));
			bbBox.include(DFToGrid*vec3<FloatType>((FloatType)grid.getDimX(), (FloatType)0, (FloatType)0));
			bbBox.include(DFToGrid*vec3<FloatType>((FloatType)grid.getDimX(), (FloatType)grid.getDimY(), (FloatType)0));
			bbBox.include(DFToGrid*vec3<FloatType>((FloatType)0, (FloatType)grid.getDimY(), (FloatType)0));
			bbBox.include(DFToGrid*vec3<FloatType>((FloatType)0, (FloatType)0, (FloatType)grid.getDimZ()));
			bbBox.include(DFToGrid*vec3<FloatType>((FloatType)grid.getDimX(), (FloatType)0, (FloatType)grid.getDimZ()));
			bbBox.include(DFToGrid*vec3<FloatType>((FloatType)grid.getDimX(), (FloatType)grid.getDimY(), (FloatType)grid.getDimZ()));
			bbBox.include(DFToGrid*vec3<FloatType>((FloatType)0, (FloatType)grid.getDimY(), (FloatType)grid.getDimZ()));

			bbBox.setMin(math::max(bbBox.getMin() - 1, 0));
			bbBox.setMax(math::min(bbBox.getMax() + 1, vec3i(grid.getDimensions())));


			for (size_t z = bbBox.getMinZ(); z < bbBox.getMaxZ(); z++) {
				for (size_t y = bbBox.getMinY(); y < bbBox.getMaxY(); y++) {
					for (size_t x = bbBox.getMinZ(); x < bbBox.getMaxX(); x++) {
						vec3<FloatType> p = gridToDF * vec3<FloatType>((FloatType)x, (FloatType)y, (FloatType)z);
						vec3ul pi(math::round(p));
						if (this->isValidCoordinate(pi.x, pi.y, pi.z)) {
							const FloatType& d = (*this)(pi.x, pi.y, pi.z);
							if (d < m_truncation) {
								if (squaredSum) {
									dist += d*d;
								}
								else {
									dist += d;
								}
								numComparisons++;
							}

						}
					}
				}
			}
			return std::make_pair(dist, numComparisons);
		}

		size_t getNumZeroVoxels() const {
			return m_numZeroVoxels;
		}

		void normalize(float factor)  {
			BinaryGrid3 res(this->getDimX(), this->getDimY(), this->getDimZ());
			for (size_t k = 0; k < this->getDimZ(); k++) {
				for (size_t j = 0; j < this->getDimY(); j++) {
					for (size_t i = 0; i < this->getDimX(); i++) {

						if ((*this)(i, j, k) != 0.0f) {
							(*this)(i, j, k) /= factor;
						}

					}
				}
			}
		}

		void improveDF(unsigned int numIter, bool respectTruncation = false) {
			for (unsigned int iter = 0; iter < numIter; iter++) {
				bool hasUpdate = false;
				for (size_t k = 0; k < this->getDimZ(); k++) {
					for (size_t j = 0; j < this->getDimY(); j++) {
						for (size_t i = 0; i < this->getDimX(); i++) {
							if (checkDistToNeighborAndUpdate(i, j, k, respectTruncation)) {
								hasUpdate = true;
							}
						}
					}
				}

				if (!hasUpdate) break;
			}
		}


		FloatType trilinearInterpolationSimpleFastFast(const vec3<FloatType>& pos) const {
			//const FloatType oSet = m_voxelSize;
			const FloatType oSet = 1.0f;

			const vec3<FloatType> posDual = pos - vec3<FloatType>(oSet / 2.0f, oSet / 2.0f, oSet / 2.0f);
			vec3<FloatType> weight = math::frac(pos);

			//if (!isValidCoordinate(math::round(posDual)) || !isValidCoordinate(math::round(posDual + oSet))) throw MLIB_EXCEPTION("out of bounds " + pos.toString());

			FloatType dist = 0.0f;
			vec3<FloatType> maxBound((FloatType)this->getDimX() - 1, (FloatType)this->getDimY() - 1, (FloatType)this->getDimZ() - 1);

			FloatType v;
			v = (*this)(math::round(math::min(math::max(posDual + vec3<FloatType>(0.0f, 0.0f, 0.0f), vec3<FloatType>::origin), vec3<FloatType>(maxBound))));	    dist += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*v;
			v = (*this)(math::round(math::min(math::max(posDual + vec3<FloatType>(oSet, 0.0f, 0.0f), vec3<FloatType>::origin), vec3<FloatType>(maxBound))));	    dist += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*v;
			v = (*this)(math::round(math::min(math::max(posDual + vec3<FloatType>(0.0f, oSet, 0.0f), vec3<FloatType>::origin), vec3<FloatType>(maxBound))));	    dist += (1.0f - weight.x)*weight.y *(1.0f - weight.z)*v;
			v = (*this)(math::round(math::min(math::max(posDual + vec3<FloatType>(0.0f, 0.0f, oSet), vec3<FloatType>::origin), vec3<FloatType>(maxBound))));	    dist += (1.0f - weight.x)*(1.0f - weight.y)*weight.z *v;
			v = (*this)(math::round(math::min(math::max(posDual + vec3<FloatType>(oSet, oSet, 0.0f), vec3<FloatType>::origin), vec3<FloatType>(maxBound))));	    dist += weight.x*weight.y *(1.0f - weight.z)*v;
			v = (*this)(math::round(math::min(math::max(posDual + vec3<FloatType>(0.0f, oSet, oSet), vec3<FloatType>::origin), vec3<FloatType>(maxBound))));	    dist += (1.0f - weight.x)*weight.y*weight.z*v;
			v = (*this)(math::round(math::min(math::max(posDual + vec3<FloatType>(oSet, 0.0f, oSet), vec3<FloatType>::origin), vec3<FloatType>(maxBound))));	    dist += weight.x *(1.0f - weight.y)*weight.z*v;
			v = (*this)(math::round(math::min(math::max(posDual + vec3<FloatType>(oSet, oSet, oSet), vec3<FloatType>::origin), vec3<FloatType>(maxBound))));	    dist += weight.x*weight.y*weight.z *v;

			return dist;
		}

		DistanceField3 upsample(const vec3ul& newDim) const {
			DistanceField3 res(newDim.x, newDim.y, newDim.z);

			vec3<FloatType> factor(
				(FloatType)res.getDimX() / (FloatType)this->getDimX(), 
				(FloatType)res.getDimY() / (FloatType)this->getDimY(), 
				(FloatType)res.getDimZ() / (FloatType)this->getDimZ());

			for (size_t z = 0; z < res.getDimZ(); z++) {
				for (size_t y = 0; y < res.getDimY(); y++) {
					for (size_t x = 0; x < res.getDimX(); x++) {
						vec3<FloatType> c((FloatType)x, (FloatType)y, (FloatType)z);
						//c /= 2.0f;	//TODO check factor based on newDim / oldDim
						c.x /= factor.x;
						c.y /= factor.y;
						c.z /= factor.z;
						res(x, y, z) = trilinearInterpolationSimpleFastFast(c);
					}
				}
			}
			return res;
		}

		DistanceField3 upsample() const {
			return upsample(vec3ul(this->getDimX() * 2, this->getDimY() * 2, this->getDimZ() * 2));
		}

		void setTruncation(float truncation, bool updateValues = true) {
			m_truncation = truncation;
			if (updateValues) {
				for (unsigned int z = 0; z < this->m_dimZ; z++) {                     // add this-> by guan
					for (unsigned int y = 0; y < this->m_dimY; y++) {              // add this-> by guan
						for (unsigned int x = 0; x < this->m_dimX; x++) {        // add this-> by guan
							float v = (*this)(x, y, z);
							if (v > truncation) (*this)(x, y, z) = truncation;
						} //x
					} //y
				} //z
			}
		}

	private:

		void generateFromBinaryGridSimple(const BinaryGrid3& grid, FloatType trunc) {

			FloatType kernel[3][3][3];
			for (int k = -1; k <= 1; k++) {
				for (int j = -1; j <= 1; j++) {
					for (int i = -1; i <= 1; i++) {
						FloatType d = vec3<FloatType>((FloatType)k, (FloatType)j, (FloatType)i).length();
						kernel[k + 1][j + 1][i + 1] = d;
					}
				}
			}

			//initialize with grid distances
			m_numZeroVoxels = 0;
			for (size_t z = 0; z < grid.getDimZ(); z++) {
				for (size_t y = 0; y < grid.getDimY(); y++) {
					for (size_t x = 0; x < grid.getDimX(); x++) {
						if (grid.isVoxelSet(x, y, z)) {
							(*this)(x, y, z) = (FloatType)0;
							m_numZeroVoxels++;
						}
						else {
							(*this)(x, y, z) = std::numeric_limits<FloatType>::infinity();
						}
					}
				}
			}

			bool found = true;
			while (found) {
				found = false;
				for (size_t z = 0; z < this->getDimZ(); z++) {
					for (size_t y = 0; y < this->getDimY(); y++) {
						for (size_t x = 0; x < this->getDimX(); x++) {

							FloatType dMin = (*this)(x, y, z);
							for (int k = -1; k <= 1; k++) {
								for (int j = -1; j <= 1; j++) {
									for (int i = -1; i <= 1; i++) {
										vec3ul n(x + i, y + j, z + k);
										if (this->isValidCoordinate(n.x, n.y, n.z)) {
											FloatType dCurr = (*this)(n.x, n.y, n.z) + kernel[i + 1][j + 1][k + 1];
											if (dCurr < dMin && dCurr <= trunc) {
												dMin = dCurr;
												found = true;
											}
										}
									}
								}
							}
							(*this)(x, y, z) = dMin;
						}
					}
				}
			}

		}

		void generateFromBinaryGridQueue(const BinaryGrid3& grid) {

			BinaryGrid3 visited(grid.getDimensions());

			//initialize with grid distances
			m_numZeroVoxels = 0;
			for (size_t z = 0; z < grid.getDimZ(); z++) {
				for (size_t y = 0; y < grid.getDimY(); y++) {
					for (size_t x = 0; x < grid.getDimX(); x++) {
						if (grid.isVoxelSet(x, y, z)) {
							(*this)(x, y, z) = (FloatType)0;
							visited.setVoxel(x, y, z);
							m_numZeroVoxels++;
						}
						else {
							(*this)(x, y, z) = std::numeric_limits<FloatType>::infinity();
						}
					}
				}
			}

			//initialize priority queue
			struct Voxel {
				Voxel(size_t _x, size_t _y, size_t _z, FloatType _d) : x(_x), y(_y), z(_z), dist(_d) {}

				bool operator<(const Voxel& other) const {
					return dist < other.dist;	//need to have the smallest one at 'top'; so it's inverted
				}

				size_t x, y, z;
				FloatType dist;
			};
			std::priority_queue<Voxel> queue;
			for (size_t z = 0; z < this->getDimZ(); z++) {
				for (size_t y = 0; y < this->getDimY(); y++) {
					for (size_t x = 0; x < this->getDimX(); x++) {
						if (!grid.isVoxelSet(x, y, z)) {
							FloatType d;
							if (isNeighborSet(grid, x, y, z, d)) {
								queue.push(Voxel(x, y, z, d));
							}
						}
					}
				}
			}

			while (!queue.empty()) {
				//first, check if the current voxel needs to be updated (and update if necessary)
				Voxel top = queue.top();
				queue.pop();

				if (!visited.isVoxelSet(top.x, top.y, top.z)) {

					visited.setVoxel(top.x, top.y, top.z);

					if (checkDistToNeighborAndUpdate(top.x, top.y, top.z)) {

						//second, check if neighbors need to be inserted into the queue
						for (size_t k = 0; k < 3; k++) {
							for (size_t j = 0; j < 3; j++) {
								for (size_t i = 0; i < 3; i++) {
									if (k == 1 && j == 1 && i == 1) continue;	//don't consider itself
									vec3ul n(top.x - 1 + i, top.y - 1 + j, top.z - 1 + k);
									if (this->isValidCoordinate(n.x, n.y, n.z) && !visited.isVoxelSet(n)) {
										FloatType d = (vec3<FloatType>((FloatType)top.x, (FloatType)top.y, (FloatType)top.z) -
											vec3<FloatType>((FloatType)n.x, (FloatType)n.y, (FloatType)n.z)).length();
										FloatType dToN = (*this)(top.x, top.y, top.z) + d;
										if (dToN < (*this)(n.x, n.y, n.z)) {
											queue.push(Voxel(n.x, n.y, n.z, dToN));
										}
									}
								}
							}
						}

					}
				}
			}

		}



		//! bools checks if there is a neighbor with a smaller distance (+ the dist to the current voxel); if then it updates the distances and returns true
		bool checkDistToNeighborAndUpdate(size_t x, size_t y, size_t z, bool respectTruncation = false) {
			bool foundBetter = false;
			for (size_t k = 0; k < 3; k++) {
				for (size_t j = 0; j < 3; j++) {
					for (size_t i = 0; i < 3; i++) {
						if (k == 1 && j == 1 && i == 1) continue;	//don't consider itself
						vec3ul n(x - 1 + i, y - 1 + j, z - 1 + k);
						if (this->isValidCoordinate(n.x, n.y, n.z)) {
							FloatType d = (vec3<FloatType>((FloatType)x, (FloatType)y, (FloatType)z) - vec3<FloatType>((FloatType)n.x, (FloatType)n.y, (FloatType)n.z)).length();
							FloatType dToN = (*this)(n.x, n.y, n.z) + d;

							if (respectTruncation) throw MLIB_EXCEPTION("not implemented");
							//if (respectTruncation) dToN = std::min(dToN, m_truncation);
							
							if (dToN < (*this)(x, y, z)) {
								(*this)(x, y, z) = dToN;
								foundBetter = true;
							}
						}
					}
				}
			}
			return foundBetter;

		}

		//! checks if a neighbor of (x,y,z) in the grid is set
		bool isNeighborSet(const BinaryGrid3& grid, size_t x, size_t y, size_t z, FloatType& d) const {
			for (size_t k = 0; k < 3; k++) {
				for (size_t j = 0; j < 3; j++) {
					for (size_t i = 0; i < 3; i++) {
						if (k == 1 && j == 1 && i == 1) continue;	//don't consider itself
						vec3ul v(x - 1 + i, y - 1 + j, z - 1 + k);
						if (grid.isValidCoordinate(v)) {
							if (grid.isVoxelSet(v)) {
								d = (vec3<FloatType>((FloatType)x, (FloatType)y, (FloatType)z) -
									vec3<FloatType>((FloatType)v.x, (FloatType)v.y, (FloatType)v.z)).length();
								//TODO avoid the costly sqrt computation and just check for 1, sqrt(2), or, sqrt(3)
								return true;
							}
						}
					}
				}
			}
			return false;
		}

		size_t m_numZeroVoxels;
		FloatType m_truncation;
	};

	typedef DistanceField3<float> DistanceField3f;
	typedef DistanceField3<double> DistanceField3d;
}



#endif // CORE_BASE_DISTANCE_FIELD3_H_
