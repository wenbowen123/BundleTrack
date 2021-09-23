
#ifndef CORE_GRAPHICS_RAY_H_
#define CORE_GRAPHICS_RAY_H_

namespace ml {


template<class FloatType>
class Ray
{
public:

    Ray()
    {

    }

	inline Ray(const vec3<FloatType> &o, const vec3<FloatType> &d) {
		m_origin = o;
		m_direction = d.getNormalized();
        m_inverseDirection = vec3<FloatType>((FloatType)1.0 / m_direction.x, (FloatType)1.0 / m_direction.y, (FloatType)1.0 / m_direction.z);

		m_sign.x = (m_inverseDirection.x < (FloatType)0);
		m_sign.y = (m_inverseDirection.y < (FloatType)0);
		m_sign.z = (m_inverseDirection.z < (FloatType)0);
	}

	inline vec3<FloatType> getHitPoint(FloatType t) const {
		return m_origin + t * m_direction;
	}

	inline const vec3<FloatType>& getOrigin() const {
		return m_origin;
	}

	inline const vec3<FloatType>& getDirection() const {
		return m_direction;
	}

	inline const vec3<FloatType>& getInverseDirection() const {
		return m_inverseDirection;
	}

	inline const vec3i& getSign() const {
		return m_sign;
	}

	inline void transform(const Matrix4x4<FloatType>& m) {
		*this = Ray(m * m_origin,  m.transformNormalAffine(m_direction));
	}

	inline void rotate(const Matrix3x3<FloatType>& m) {
		*this = Ray(m_origin, m * m_direction);
	}

	inline void translate(const vec3<FloatType>& p) {
		*this = Ray(m_origin + p, m_direction);
	}
private:
	vec3<FloatType> m_direction;
	vec3<FloatType> m_inverseDirection;
	vec3<FloatType> m_origin;

	vec3i m_sign;
};

template<class FloatType>
Ray<FloatType> operator*(const Matrix4x4<FloatType>& m, const Ray<FloatType>& r) {
	Ray<FloatType> res = r; 
	res.transform(m);
	return res;
}

template<class FloatType>
std::ostream& operator<<(std::ostream& os, const Ray<FloatType>& r) {
	os << r.getOrigin() << " | " << r.getDirection();
	return os;
}

typedef Ray<float> Rayf;
typedef Ray<double> Rayd;

}  // namespace ml

#endif  // CORE_GRAPHICS_RAY_H_
