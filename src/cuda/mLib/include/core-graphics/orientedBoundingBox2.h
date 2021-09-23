#pragma once

#ifndef OBJECT_ORIENTED_BOUNDIG_BOX2_H_
#define OBJECT_ORIENTED_BOUNDIG_BOX2_H_

namespace ml {

template <class FloatType>
class OrientedBoundingBox2 {
public:

	OrientedBoundingBox2() {
		setInvalid();
	}

    OrientedBoundingBox2(const BoundingBox2<FloatType> &box)
    {
        m_Anchor = box.getMin();
        m_AxesScaled[0] = vec2f::eX * box.getExtentX();
        m_AxesScaled[1] = vec2f::eY * box.getExtentY();
    }

	//! constructs an oriented bounding box using PCA
	OrientedBoundingBox2(const std::vector<vec2<FloatType>>& points) {

		auto pca = math::pointSetPCA(points);
		m_AxesScaled[0] = pca[0].first.getNormalized();
		m_AxesScaled[1] = pca[1].first.getNormalized();

		computeAnchorAndExtentsForGivenNormalizedAxis(points);
	}

	//! creates an object oriented bounding for a given set of points with the same axes as the other OBB
	OrientedBoundingBox2(const std::vector<vec2<FloatType>>& points, const OrientedBoundingBox2& other) {
		m_AxesScaled[0] = other.m_AxesScaled[0].getNormalized();
		m_AxesScaled[1] = other.m_AxesScaled[1].getNormalized();

		computeAnchorAndExtentsForGivenNormalizedAxis(points);
	}

	//! creates an object oriented bounding box given a set of points and 2 axes
	OrientedBoundingBox2(const std::vector<vec2<FloatType>>& points, const vec2<FloatType>& xAxis, const vec2<FloatType>& yAxis) {
		m_AxesScaled[0] = xAxis.getNormalized();
		m_AxesScaled[1] = yAxis.getNormalized();

		computeAnchorAndExtentsForGivenNormalizedAxis(points);
	}

	//! creates an object oriented bounding box given an anchor and 3 axes
	OrientedBoundingBox2(const vec2<FloatType>& anchor, const vec2<FloatType>& xAxis, const vec2<FloatType>& yAxis) {
		m_AxesScaled[0] = xAxis;
		m_AxesScaled[1] = yAxis;

		m_Anchor = anchor;
	}

	bool isValid() const {
		if (m_Anchor.x == -std::numeric_limits<FloatType>::max() || m_Anchor.y == -std::numeric_limits<FloatType>::max())	
			return false;
		else return true;
	}

	void setInvalid() {
		m_Anchor.x = m_Anchor.y = -std::numeric_limits<FloatType>::max();
	}

	std::vector< vec2<FloatType> > getVertices() const
	{
		std::vector< vec2<FloatType> > result(4);

		result[0] = (m_Anchor + m_AxesScaled[0] * (FloatType)0.0 + m_AxesScaled[1] * (FloatType)0.0);
		result[1] = (m_Anchor + m_AxesScaled[0] * (FloatType)1.0 + m_AxesScaled[1] * (FloatType)0.0);
		result[2] = (m_Anchor + m_AxesScaled[0] * (FloatType)1.0 + m_AxesScaled[1] * (FloatType)1.0);
		result[3] = (m_Anchor + m_AxesScaled[0] * (FloatType)0.0 + m_AxesScaled[1] * (FloatType)1.0);

		return result;
	}


	//! returns the transformation matrix that transforms points into the space of the OBB
	inline Matrix3x3<FloatType> getOBBToWorld() const 
	{
		return Matrix3x3<FloatType>(
			m_AxesScaled[0].x, m_AxesScaled[1].x, m_Anchor.x,
			m_AxesScaled[0].y, m_AxesScaled[1].y, m_Anchor.y,
			0, 0, 1);
	}

	//! returns a matrix that transforms to OBB space [0,1]x[0,1]
	inline Matrix3x3<FloatType> getWorldToOBB() const {
		//return getOOBBToWorld().getInverse();

		FloatType scaleValues[2] = { m_AxesScaled[0].length(), m_AxesScaled[1].length() };
		Matrix2x2<FloatType> worldToOBB2x2(m_AxesScaled[0] / scaleValues[0], m_AxesScaled[1] / scaleValues[1]);

		worldToOBB2x2(0, 0) /= scaleValues[0];	worldToOBB2x2(0, 1) /= scaleValues[0];
		worldToOBB2x2(1, 0) /= scaleValues[1];	worldToOBB2x2(1, 1) /= scaleValues[1];

		vec2<FloatType> trans = worldToOBB2x2 * (-m_Anchor);
		Matrix3x3<FloatType> worldToOBB = Matrix3x3<FloatType>(
			worldToOBB2x2(0,0), worldToOBB2x2(0,1), 0.0f,
			worldToOBB2x2(1,0), worldToOBB2x2(1,1), 0.0f,
			0.0f, 0.0f, 1.0f);
		worldToOBB.at(0, 2) = trans.x;
		worldToOBB.at(1, 2) = trans.y;

		return worldToOBB;
	}

	//! returns the center of the OBB
	vec2<FloatType> getCenter() const {
		return m_Anchor + (m_AxesScaled[0] + m_AxesScaled[1]) * (FloatType)0.5;
	}

	//! returns the n'th axis of the OBB
	const vec2<FloatType>& getAxis(unsigned int n) const {
		return m_AxesScaled[n];
	}

	//! returns the first axis of the OBB
	const vec2<FloatType>& getAxisX() const {
		return m_AxesScaled[0];
	}

	//! returns the second axis of the OBB
	const vec2<FloatType>& getAxisY() const {
		return m_AxesScaled[1];
	}

	vec2<FloatType> getExtent() const {
		return vec2<FloatType>(m_AxesScaled[0].length(), m_AxesScaled[1].length());
	}
	float getExtentX() const {
		return m_AxesScaled[0].length();
	}
	float getExtentY() const {
		return m_AxesScaled[1].length();
	}

	vec2<FloatType> getAnchor() const {
		return m_Anchor;
	}

	FloatType getArea() const {
		vec2<FloatType> extent = getExtent();
		return extent.x * extent.y;
	}

	//! returns the diagonal extent of the OBB
	FloatType getDiagonalLength() const {
		return (m_AxesScaled[0] + m_AxesScaled[1]).length();
	}

	std::vector< LineSegment2<FloatType> > getEdges() const
	{
		std::vector< LineSegment2<FloatType> > result;	result.reserve(4);
		auto v = getVertices();

		result.push_back(LineSegment2<FloatType>(v[0], v[1]));
		result.push_back(LineSegment2<FloatType>(v[1], v[2]));
		result.push_back(LineSegment2<FloatType>(v[2], v[3]));
		result.push_back(LineSegment2<FloatType>(v[3], v[0]));

		return result;
	}

	//! scales the OBB
	void operator*=(const FloatType& scale) {
		vec2<FloatType> center = getCenter();
		m_AxesScaled[0] *= scale;
		m_AxesScaled[1] *= scale;
		m_Anchor = center - (m_AxesScaled[0] + m_AxesScaled[1]) * (FloatType)0.5;
	}
	//! returns a scaled OBB
	OrientedBoundingBox2<FloatType> operator*(const FloatType& scale) const {
		OrientedBoundingBox2<FloatType> res = *this;
		res *= scale;
		return res;
	}

	//! extends the OBB
	void operator+=(const FloatType& ext) {
		FloatType scaleValues[2] = { m_AxesScaled[0].length(), m_AxesScaled[1].length() };
		vec2<FloatType> center = getCenter();
		m_AxesScaled[0] *= (scaleValues[0] + ext) / scaleValues[0];
		m_AxesScaled[1] *= (scaleValues[1] + ext) / scaleValues[1];
		m_Anchor = center - (m_AxesScaled[0] + m_AxesScaled[1]) * (FloatType)0.5;
	}
	//! returns an extended OBB
	OrientedBoundingBox2<FloatType> operator+(const FloatType& ext) const {
		OrientedBoundingBox2<FloatType> res = *this;
		res += ext;
		return res;
	}

	//! returns a transformed OBB
	void operator*=(const Matrix3x3<FloatType>& mat) {
		assert(mat.isAffine());
		m_Anchor = mat * m_Anchor;
		Matrix2x2<FloatType> rot = mat.getMatrix2x2();
		m_AxesScaled[0] = rot * m_AxesScaled[0];
		m_AxesScaled[1] = rot * m_AxesScaled[1];
	}

	//TODO
	//bool intersects(const OrientedBoundingBox2<FloatType>& other) const {
	//	return intersection::intersectOBBOBB<FloatType>(m_Anchor, &m_AxesScaled[0], other.m_Anchor, &other.m_AxesScaled[0]);
	//}


	//! scales the bounding box by the factor t (for t=1 the bb remains unchanged)
	void scale(FloatType x, FloatType y) {
		vec2<FloatType> extent = getExtent();
		FloatType scale[] = { x, y };
		vec2<FloatType> center(0, 0);
		for (unsigned int i = 0; i < 2; i++) {
			center += (FloatType)0.5 * m_AxesScaled[i];
			m_AxesScaled[i] *= scale[i];
		}
		m_Anchor = center - (FloatType)0.5 * (m_AxesScaled[0] + m_AxesScaled[1]);
	}

	//! scales the bounding box by the factor t (for t=1 the bb remains unchanged)
	void scale(FloatType t) {
		scale(t, t, t);
	}

	//! warning: not tested
	bool intersects(const vec2<FloatType>& point) const {
		vec3f pp = getWorldToOBB() * vec3f(point.x, point.y, 1.0f);
		vec2f p(pp.x / pp.z, pp.y / pp.z);
		return (p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1);
	}
private:

	void computeAnchorAndExtentsForGivenNormalizedAxis(const std::vector<vec2<FloatType>>& points)
	{
		assert((m_AxesScaled[0] | m_AxesScaled[1]) < (FloatType)0.001);
		assert((m_AxesScaled[1] | m_AxesScaled[2]) < (FloatType)0.001);

		Matrix2x2<FloatType> worldToOBBSpace(m_AxesScaled[0], m_AxesScaled[1]);
		Matrix2x2<FloatType> OBBSpaceToWorld = worldToOBBSpace.getTranspose();	//is an orthogonal matrix

		vec2<FloatType> minValues(std::numeric_limits<FloatType>::max(), std::numeric_limits<FloatType>::max());
		vec2<FloatType> maxValues(-std::numeric_limits<FloatType>::max(), -std::numeric_limits<FloatType>::max());

		for (size_t i = 0; i < points.size(); i++) {
			vec2<FloatType> curr = worldToOBBSpace * points[i];
			if (curr.x < minValues.x)	minValues.x = curr.x;
			if (curr.y < minValues.y)	minValues.y = curr.y;

			if (curr.x > maxValues.x)	maxValues.x = curr.x;
			if (curr.y > maxValues.y)	maxValues.y = curr.y;
		}

		m_Anchor = OBBSpaceToWorld * minValues;

		FloatType extent[2];

		extent[0] = maxValues.x - minValues.x;
		extent[1] = maxValues.y - minValues.y;

		//if bounding box has no extent; set invalid and return
		if (extent[0] < (FloatType)0.00001 ||
			extent[1] < (FloatType)0.00001) {
			setInvalid();
			return;
		}

		m_AxesScaled[0] *= extent[0];
		m_AxesScaled[1] *= extent[1];
	}

	vec2<FloatType>	m_Anchor;
	vec2<FloatType>	m_AxesScaled[2];	//these axes are not normalized; they are scaled according to the extent

};

template<class FloatType>
OrientedBoundingBox2<FloatType> operator*(const Matrix3x3<FloatType> &mat, const OrientedBoundingBox2<FloatType>& oobb) {
	OrientedBoundingBox2<FloatType> res = oobb;
	res *= mat;
	return res;
}

template<class FloatType>
std::ostream& operator<<(std::ostream& os, const OrientedBoundingBox2<FloatType>& obb)  {
	os << obb.getAxisX() << std::endl << obb.getAxisY() << std::endl << std::endl;
	os << "Extent: " << obb.getExtent() << std::endl;
	os << "Anchor: " << obb.getAnchor() << std::endl;
	os << "Volume: " << obb.getVolume() << std::endl;
	return os;
}

typedef OrientedBoundingBox2<float> OrientedBoundingBox2f;
typedef OrientedBoundingBox2<double> OrientedBoundingBox2d;

typedef OrientedBoundingBox2<float> OBB2f;
typedef OrientedBoundingBox2<double> OBB2d;

} //namespace ml

#endif
