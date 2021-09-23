
#ifndef CORE_GRAPHICS_LINESEGMENT2_H_
#define CORE_GRAPHICS_LINESEGMENT2_H_

namespace ml {

template<class T>
class LineSegment2
{
public:

    LineSegment2(const vec2<T> &p0, const vec2<T> &p1)
    {
        m_p0 = p0;
        m_p1 = p1;
        m_delta = m_p1 - m_p0;
	}

	const vec2<T>& p0() const
    {
        return m_p0;
	}

	const vec2<T>& p1() const
    {
        return m_p1;
	}

	const vec2<T>& delta() const
    {
        return m_delta;
	}

private:
	vec2<T> m_p0;
	vec2<T> m_p1;
	vec2<T> m_delta;  //p1 - p0
};

typedef LineSegment2<float> LineSegment2f;
typedef LineSegment2<double> LineSegment2d;

}  // namespace ml

#endif  // CORE_GRAPHICS_LINESEGMENT2D_H_
