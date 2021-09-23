
#ifndef CORE_GRAPHICS_DIST_H_
#define CORE_GRAPHICS_DIST_H_

namespace ml {


////
//// TODO: this should be moved into an intersect class
////
//namespace math {
//bool triangleIntersectTriangle(const ml::vec3f &t0v0, const ml::vec3f &t0v1, const ml::vec3f &t0v2, const ml::vec3f &t1v0, const ml::vec3f &t1v1, const ml::vec3f &t1v2);
//
//bool triangleIntersectTriangle(const ml::vec3f t0[3], const ml::vec3f t1[3]);
//}


template <class T>
T distSq(const vec2<T> &ptA, const vec2<T> &ptB)
{
    return vec2<T>::distSq(ptA, ptB);
}

template <class T>
T distSq(const vec3<T> &ptA, const vec3<T> &ptB)
{
    return vec3<T>::distSq(ptA, ptB);
}

template <class T>
T distSq(const LineSegment2<T> &seg, const vec2<T> &p)
{
    const vec2<T> &v = seg.p0();
    const vec2<T> &w = seg.p1();
    
    //
    // http://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
    // Return minimum distance between line segment vw and point p
    //
    const T l2 = distSq(v, w);  // i.e. |w-v|^2 -  avoid a sqrt
    if (l2 == (T)0.0) return distSq(p, v);   // v == w case

    //
    // Consider the line extending the segment, parameterized as v + t (w - v).
    // We find projection of point p onto the line. 
    // It falls where t = [(p-v) . (w-v)] / |w-v|^2
    //
    const T t = ((p - v) | (w - v)) / l2;
    if (t < (T)0.0) return distSq(p, v);      // Beyond the 'v' end of the segment
    else if (t > (T)1.0) return distSq(p, w); // Beyond the 'w' end of the segment
    const vec2<T> projection = v + t * (w - v);  // Projection falls on the segment
    return distSq(p, projection);
}

template <class T>
T distSq(const Line2<T> &line, const vec2<T> &pt)
{
    const vec2<T> diff = line.dir();
    const T d = diff.lengthSq();
    const vec2<T> p0 = line.p0();
    const vec2<T> p1 = line.p0() + line.dir();
    T n = fabs(diff.y * pt.x - diff.x * pt.y + p1.x * p0.y - p1.y * p0.x);
    return n / d;
}

template <class T>
T distSq(const OrientedBoundingBox3<T> &box, const vec3<T> &pt)
{
    //
    // This is wrong, this file is just meant as an example of the dist interface
    //
    return vec3<T>::distSq(box.getCenter(), pt);
}

template <class T>
T distSq(const vec3<T> &pt, const OrientedBoundingBox3<T> &box)
{
    return distSq(box, pt);
}

//
// code adapted from http://geomalgorithms.com/a07-_distance.html#dist3D_Segment_to_Segment
//
template <class T>
double distSq(const LineSegment2<T> &s0, const LineSegment2<T> &s1)
{
    const vec2<T> u = s0.delta();
    const vec2<T> v = s1.delta();
    const vec2<T> w = s0.p0() - s1.p0();
    double a = vec2<T>::dot(u, u);         // always >= 0
    double b = vec2<T>::dot(u, v);
    double c = vec2<T>::dot(v, v);         // always >= 0
    double d = vec2<T>::dot(u, w);
    double e = vec2<T>::dot(v, w);
    double D = a * c - b * b;        // always >= 0
    double sc, sN, sD = D;       // sc = sN / sD, default sD = D >= 0
    double tc, tN, tD = D;       // tc = tN / tD, default tD = D >= 0

    // compute the line parameters of the two closest points
    if (D < 1e-6) { // the lines are almost parallel
        sN = 0.0;         // force using point P0 on segment S1
        sD = 1.0;         // to prevent possible division by 0.0 later
        tN = e;
        tD = c;
    }
    else {                 // get the closest points on the infinite lines
        sN = (b*e - c*d);
        tN = (a*e - b*d);
        if (sN < 0.0) {        // sc < 0 => the s=0 edge is visible
            sN = 0.0;
            tN = e;
            tD = c;
        }
        else if (sN > sD) {  // sc > 1  => the s=1 edge is visible
            sN = sD;
            tN = e + b;
            tD = c;
        }
    }

    if (tN < 0.0) {            // tc < 0 => the t=0 edge is visible
        tN = 0.0;
        // recompute sc for this edge
        if (-d < 0.0)
            sN = 0.0;
        else if (-d > a)
            sN = sD;
        else {
            sN = -d;
            sD = a;
        }
    }
    else if (tN > tD) {      // tc > 1  => the t=1 edge is visible
        tN = tD;
        // recompute sc for this edge
        if ((-d + b) < 0.0)
            sN = 0;
        else if ((-d + b) > a)
            sN = sD;
        else {
            sN = (-d + b);
            sD = a;
        }
    }
    // finally do the division to get sc and tc
    sc = (std::abs(sN) < 1e-6 ? 0.0 : sN / sD);
    tc = (std::abs(tN) < 1e-6 ? 0.0 : tN / tD);

    // get the difference of the two closest points
    const vec2<T> dP = w + ((float)sc * u) - ((float)tc * v);  // =  S1(sc) - S2(tc)

    return dP.lengthSq();   // return the closest distance
}

template <class T, class U>
double distSq(const std::vector<T> &collectionA, const std::vector<U> &collectionB)
{
    double minDistSq = std::numeric_limits<double>::max();
    for (const T &a : collectionA)
    {
        for (const U &b : collectionB)
        {
            double curDistSq = distSq(a, b);
            if (curDistSq < minDistSq)
            {
                minDistSq = curDistSq;
            }
        }
    }
    return minDistSq;
}

template <class A, class B>
double dist(const A &a, const B &b)
{
    return sqrt(distSq(a, b));
}

}  // namespace ml

#endif  // CORE_GRAPHICS_DIST_H_
