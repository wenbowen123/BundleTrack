
namespace ml
{

    struct UniformAccelerator
    {
        struct Entry
        {
            std::vector< std::pair<int, vec3f> > points;
        };

        bbox3f bbox;
        float cubeSize;
        Grid3<Entry> data;

        UniformAccelerator() {}
        UniformAccelerator(const std::vector<vec3f> &points, float _cubeSize)
        {
            bbox3f _bbox;
            for (auto &v : points)
                _bbox.include(v);
            init(_bbox, _cubeSize);
            for (int i = 0; i < points.size(); i++)
                addPoint(points[i], i);
        }

        void init(const bbox3f &_bbox, float _cubeSize)
        {
            bbox = _bbox;
            cubeSize = _cubeSize;
            int dimX = math::ceil(bbox.getExtentX() / cubeSize) + 1;
            int dimY = math::ceil(bbox.getExtentY() / cubeSize) + 1;
            int dimZ = math::ceil(bbox.getExtentZ() / cubeSize) + 1;
            data.allocate(dimX, dimY, dimZ);
        }

        vec3i getCoord(const vec3f &pos) const
        {
            int x = math::clamp((int)math::linearMap(bbox.getMinX(), bbox.getMaxX(), 0.0f, (float)data.getDimX(), pos.x), 0, (int)data.getDimX() - 1);
            int y = math::clamp((int)math::linearMap(bbox.getMinY(), bbox.getMaxY(), 0.0f, (float)data.getDimY(), pos.y), 0, (int)data.getDimY() - 1);
            int z = math::clamp((int)math::linearMap(bbox.getMinZ(), bbox.getMaxZ(), 0.0f, (float)data.getDimZ(), pos.z), 0, (int)data.getDimZ() - 1);
            return vec3i(x, y, z);
        }

        void addPoint(const vec3f &pos, int pointIndex)
        {
            const vec3i coord = getCoord(pos);
            data(coord.x, coord.y, coord.z).points.push_back(std::make_pair(pointIndex, pos));
        }

        //! returns the closest point index and corresponding vec3. If no point is found in the adjacent boxes, returns <-1, infinity>.
        std::pair<int, vec3f> findClosestPoint(const vec3f &pos) const
        {
            const vec3i baseCoord = getCoord(pos);

            float bestDistSq = std::numeric_limits<float>::max();
            int bestPointIndex = -1;
            vec3f bestPoint = vec3f(bestDistSq, bestDistSq, bestDistSq);

            for (int xOffset = -1; xOffset <= 1; xOffset++)
                for (int yOffset = -1; yOffset <= 1; yOffset++)
                    for (int zOffset = -1; zOffset <= 1; zOffset++)
                    {
                        const vec3i coord = baseCoord + vec3i(xOffset, yOffset, zOffset);
                        if (data.isValidCoordinate(coord.x, coord.y, coord.z))
                        {
                            for (auto &p : data(coord.x, coord.y, coord.z).points)
                            {
                                const float distSq = vec3f::distSq(pos, p.second);
                                if (distSq < bestDistSq)
                                {
                                    bestPointIndex = p.first;
                                    bestPoint = p.second;
                                    bestDistSq = distSq;
                                }
                            }
                        }
                    }

            return std::make_pair(bestPointIndex, bestPoint);
        }
    };

}
