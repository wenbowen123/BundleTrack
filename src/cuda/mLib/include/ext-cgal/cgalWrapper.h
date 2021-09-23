#ifndef _CGAL_WRAPPER_H_
#define _CGAL_WRAPPER_H_


namespace ml {

template <class FloatType>
class CGALWrapper {
public:
	static std::vector<vec3<FloatType>> convexHull3(typename std::vector<vec3<FloatType>>::const_iterator pBegin, typename std::vector<vec3<FloatType>>::const_iterator pEnd) {

		typedef CGAL::Exact_predicates_inexact_constructions_kernel  K;
		typedef CGAL::Polyhedron_3<K>                                Polyhedron_3;
		typedef K::Point_3                                           Point_3;

		const size_t nPoints = pEnd - pBegin;
		std::vector<Point_3> cgalPoints;
		cgalPoints.reserve(nPoints);
		for (auto& it = pBegin; it != pEnd; ++it) {
			const auto& p = *it;
			cgalPoints.push_back(Point_3(p[0], p[1], p[2]));
		}

		Polyhedron_3 poly;
		CGAL::convex_hull_3(std::begin(cgalPoints), std::end(cgalPoints), poly);
		poly.vertices_begin();
		std::vector<vec3<FloatType>> out(poly.size_of_vertices());

		size_t i = 0;
		for (auto it = poly.vertices_begin(); it != poly.vertices_end(); ++it, ++i) {
			vec3<FloatType>& v = out[i];
			const auto& p = it->point();
			v[0] = static_cast<FloatType>(p[0]);
			v[1] = static_cast<FloatType>(p[1]);
			v[2] = static_cast<FloatType>(p[2]);
		}

		return out;
	}

	static std::vector<vec3<FloatType>> convexHull3(const std::vector<vec3<FloatType>>& points) {
		return convexHull3(points.begin(), points.end());
	}

	static std::vector<vec2<FloatType>> convexHull2(typename std::vector<vec2<FloatType>>::const_iterator pBegin, typename std::vector<vec2<FloatType>>::const_iterator pEnd) {
		typedef CGAL::Simple_cartesian<FloatType>        K;
		typedef K::Point_2                           Point_2;
		typedef CGAL::Polygon_2<K>                   Polygon_2;

		const size_t nPoints = pEnd - pBegin;
		std::vector<Point_2> cgalPoints;
		cgalPoints.reserve(nPoints);
		for (auto& it = pBegin; it != pEnd; ++it) {
			const auto& p = *it;
			cgalPoints.push_back(Point_2(p[0], p[1]));
		}

		Polygon_2 p;
		CGAL::convex_hull_2(begin(cgalPoints), end(cgalPoints), std::back_inserter(p));
		std::vector<vec2<FloatType>> out(p.size()); 

		size_t i = 0;
		for (auto it = p.vertices_begin(); it != p.vertices_end(); ++it, ++i) {
			vec2<FloatType>& v = out[i];
			v[0] = static_cast<FloatType>(it->x());
			v[1] = static_cast<FloatType>(it->y());
		}
		return out;
	}
	static std::vector<vec2<FloatType>> convexHull2(const std::vector<vec2<FloatType>>& points) {
		return convexHull2(points.begin(), points.end());
	}

	static FloatType polygonArea(typename std::vector<vec2<FloatType>>::const_iterator pBegin, typename std::vector<vec2<FloatType>>::const_iterator pEnd) {
		typedef CGAL::Simple_cartesian<FloatType>        K;
		typedef K::Point_2                           Point_2;
		typedef CGAL::Polygon_2<K>                   Polygon_2;

		const size_t nPoints = pEnd - pBegin;
		std::vector<Point_2> cgalPoints;
		cgalPoints.reserve(nPoints);
		for (auto& it = pBegin; it != pEnd; ++it) {
			const auto& p = *it;
			cgalPoints.push_back(Point_2(p[0], p[1]));
		}
		Polygon_2 p(begin(cgalPoints), end(cgalPoints));
		return (FloatType)p.area();
	}
	static FloatType polygonArea(const std::vector<vec2<FloatType>>& points) {
		return polygonArea(points.begin(), points.end());
	}

	static std::vector<vec2<FloatType>> minRectangle2D(typename std::vector<vec2<FloatType>>::const_iterator pBegin, typename std::vector<vec2<FloatType>>::const_iterator pEnd) {
		typedef CGAL::Simple_cartesian<FloatType>        K;
		typedef K::Point_2                           Point_2;
		typedef K::Line_2                            Line_2;
		typedef CGAL::Polygon_2<K>                   Polygon_2;

		const size_t nPoints = pEnd - pBegin;
		std::vector<Point_2> cgalPoints;
		cgalPoints.reserve(nPoints);
		for (auto& it = pBegin; it != pEnd; ++it) {
			const auto& p = *it;
			cgalPoints.push_back(Point_2(p[0], p[1]));
		}

		Polygon_2 p;
		CGAL::convex_hull_2(begin(cgalPoints), end(cgalPoints),
			std::back_inserter(p));

		Polygon_2 p_m;
		CGAL::min_rectangle_2(p.vertices_begin(), p.vertices_end(),
			std::back_inserter(p_m));

		std::vector<vec2<FloatType>> out(p_m.size());
		size_t i = 0;
		for (auto it = p_m.vertices_begin(); it != p_m.vertices_end(); ++it, ++i) {
			vec2<FloatType>& v = out[i];
			v[0] = static_cast<FloatType>(it->x());
			v[1] = static_cast<FloatType>(it->y());
		}
		return out;
	}

	static std::vector<vec2<FloatType> > minRectangle2D(const std::vector < vec2<FloatType> >& points) { 
		return minRectangle2D(points.begin(), points.end());  
	}

	enum FitOpts {
		PCA = 1 << 0,  // OBB axes determined through PCA decomposition of point set
		MIN_PCA = 1 << 1,  // Use minimum PCA axis and determine other axes through 2D project + min rectangle fit
		AABB_FALLBACK = 1 << 2,  // Fall back to AABB if volume of AABB is within 10% of OBB
		CONSTRAIN_Z = 1 << 3,  // Constrain OBB z axis to be canonical z axis
		CONSTRAIN_AXIS = 1 << 4, // Constrain OBB to have an arbitrary canonical z axis (needs an additional parameter)
		DEFAULT_OPTS = PCA
	};
	typedef FlagSet<FitOpts> FitOptFlags;

	static OrientedBoundingBox3<FloatType> computeOrientedBoundingBox(
		const std::vector<vec3<FloatType>>& points, 
		const FitOptFlags opts = DEFAULT_OPTS, 
		const vec3<FloatType>& axisConstrain = vec3<FloatType>(0,0,1)) 
	{

		if (opts[PCA]) {
			//auto pca = math::pointSetPCA(points);
			//return OrientedBoundingBox3<FloatType>(points, pca[0].first, pca[1].first, pca[2].first);
			return OrientedBoundingBox3<FloatType>(points);
		}
		else if (opts[CONSTRAIN_Z]) {
			// Get centroid, z range and x-y points for 2D rect fitting
			std::vector<vec2<FloatType>> projPs(points.size());
			size_t i = 0;
			FloatType big = std::numeric_limits<FloatType>::max();
			vec3<FloatType> pMin(big, big, big), pMax(-big, -big, -big);
			for (auto it = points.begin(); it != points.end(); it++, i++) {
				const FloatType x = (*it)[0], y = (*it)[1], z = (*it)[2];
				if (x < pMin[0]) { pMin[0] = x; }
				else if (x > pMax[0]) { pMax[0] = x; }
				if (y < pMin[1]) { pMin[1] = y; }
				else if (y > pMax[1]) { pMax[1] = y; }
				if (z < pMin[2]) { pMin[2] = z; }
				else if (z > pMax[2]) { pMax[2] = z; }
				projPs[i][0] = x;  projPs[i][1] = y;
			}

			// Find minimum rectangle in x-y plane
			const std::vector<vec2<FloatType>>& rectPts = minRectangle2D(projPs);

			// Set x and y bbox axes from 2D rectangle axes
			const vec2<FloatType>& v0 = rectPts[1] - rectPts[0], v1 = rectPts[2] - rectPts[1];
			const FloatType v0norm2 = v0.lengthSq(), v1norm2 = v1.lengthSq();
			size_t v0idx = (v0norm2 > v1norm2) ? 0 : 1;
			size_t v1idx = (v0idx + 1) % 2;
			const vec2<FloatType>& v0n = v0.getNormalized(), v1n = v1.getNormalized();
			//R_.col(v0idx) = vec3<FloatType>(v0n[0], v0n[1], 0);  r_[v0idx] = sqrt(v0norm2) * (FloatType)0.5;
			//R_.col(v1idx) = vec3<FloatType>(v1n[0], v1n[1], 0);  r_[v1idx] = sqrt(v1norm2) * (FloatType)0.5;
			//R_.col(2) = vec3<FloatType>(0, 0, 1);                r_[2] = (pMax[2] - pMin[2]) * (FloatType)0.5;
			//c_ = (pMin + pMax) * (FloatType)0.5;

			return OrientedBoundingBox3<FloatType>(points, vec3<FloatType>(v0n[0], v0n[1], 0), vec3<FloatType>(v1n[0], v1n[1], 0), vec3<FloatType>(0, 0, 1));
		}
		else if (opts[MIN_PCA]) {
			// Project points into 2D plane formed by the first two eigenvector
			// in R's columns. The plane normal is the last eigenvector
			vector< std::pair<vec3<FloatType>, FloatType> > pca = math::pointSetPCA(points);
			Matrix3x3<FloatType> proj3x3(pca[0].first, pca[1].first, pca[2].first);		proj3x3.transpose();
			std::vector<vec2<FloatType>> projPs(points.size());
			size_t i = 0;
			for (auto it = points.begin(); it != points.end(); it++, i++) {
				const vec3<FloatType>& p = proj3x3 * *it;
				projPs[i] = vec2<FloatType>(p.x, p.y);
			}

			// Find minimum rectangle in that plane
			const std::vector<vec2<FloatType>>& rectPts = minRectangle2D(projPs);

			// Set new bbox axes v0 and v1 from 2D rectangle's axes by first getting
			// back their 3D world space coordinates and then ordering by length so
			// that v0 remains largest OBB dimension, followed by v1
			//const vec2<FloatType> pV0 = rectPts[1] - rectPts[0], pV1 = rectPts[2] - rectPts[1];
			//const vec3<FloatType> bv0 = Mproj.transpose() * pV0, bv1 = Mproj.transpose() * pV1;
			//const float bv0norm = bv0.squaredNorm(), bv1norm = bv1.squaredNorm();
			//R_.col(0) = (bv0norm > bv1norm) ? bv0.normalized() : bv1.normalized();
			//R_.col(1) = (bv0norm > bv1norm) ? bv1.normalized() : bv0.normalized();
			//R_.col(2) = R_.col(0).cross(R_.col(1));

			const vec2<FloatType> pV0 = rectPts[1] - rectPts[0], pV1 = rectPts[2] - rectPts[1];
			const vec3<FloatType> bv0 = proj3x3.getTranspose() * vec3<FloatType>(pV0, (FloatType)0), bv1 = proj3x3.getTranspose() * vec3<FloatType>(pV1, (FloatType)0);

			return OrientedBoundingBox3<FloatType>(points, bv0, bv1, (bv0 ^ bv1).getNormalized());
		}
		else if (opts[CONSTRAIN_AXIS]) {
			vec3<FloatType> e(axisConstrain.z, -axisConstrain.x, axisConstrain.y);
			vec3<FloatType> a0 = axisConstrain^e;
			if (a0.lengthSq() < (FloatType)0.0001)	throw MLIB_EXCEPTION("invalid axis");
			vec3<FloatType> a1 = (a0 ^ axisConstrain).getNormalized();
			Matrix3x3<FloatType> proj3x3(a0, a1, axisConstrain.getNormalized());		proj3x3.transpose();
			std::vector<vec2<FloatType>> projPs(points.size());
			size_t i = 0;
			for (auto it = points.begin(); it != points.end(); it++, i++) {
				const vec3<FloatType>& p = proj3x3 * *it;
				projPs[i] = vec2<FloatType>(p.x, p.y);
			}

			// Find minimum rectangle in that plane
			const std::vector<vec2<FloatType>>& rectPts = minRectangle2D(projPs);

			const vec2<FloatType> pV0 = rectPts[1] - rectPts[0], pV1 = rectPts[2] - rectPts[1];
			const vec3<FloatType> bv0 = proj3x3.getTranspose() * vec3<FloatType>(pV0, (FloatType)0), bv1 = proj3x3.getTranspose() * vec3<FloatType>(pV1, (FloatType)0);

			return OrientedBoundingBox3<FloatType>(points, bv0, bv1, (bv0 ^ bv1).getNormalized());
		}

		// Can now decide if AABB is a better or almost as good fit and revert to it
		else if (opts[AABB_FALLBACK]) {
			return OrientedBoundingBox3<FloatType>(BoundingBox3<FloatType>(points));
		}
		else {
			throw MLIB_EXCEPTION("invalid flags");
			return OrientedBoundingBox3 < FloatType >();
		}
	}
private:

};

typedef CGALWrapper<float>	CGALWrapperf;
typedef CGALWrapper<double>	CGALWrapperd;

} // end namespace

#endif
