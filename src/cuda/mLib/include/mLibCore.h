#ifndef MLIBCORE_H_
#define MLIBCORE_H_

#ifndef _WIN32
#define LINUX
#endif

//
// core-base headers
//
#include "core-base/common.h"

//
// core-util headers (these are required by matrices)
//
#include "core-util/utility.h"
#include "core-util/stringUtil.h"
#include "core-util/windowsUtil.h"
#include "core-util/flagSet.h"
#include "core-util/binaryDataCompressor.h"
#include "core-util/binaryDataBuffer.h"
#include "core-util/binaryDataSerialize.h"
#include "core-util/binaryDataStream.h"

//
// core-math headers
//
#include "core-math/numericalRecipesTemplates.h"
#include "core-math/vec1.h"
#include "core-math/vec2.h"
#include "core-math/vec3.h"
#include "core-math/vec4.h"
#include "core-math/vec6.h"
#include "core-math/matrix2x2.h"
#include "core-math/matrix3x3.h"
#include "core-math/matrix4x4.h"
#include "core-math/quaternion.h"
#include "core-math/mathVector.h"
#include "core-math/sparseMatrix.h"
#include "core-math/denseMatrix.h"
#include "core-math/linearSolver.h"
#include "core-math/eigenSolver.h"
#include "core-math/rng.h"
#include "core-math/kMeansClustering.h"
#include "core-math/sampling.h"
#include "core-math/mathUtil.h"
#include "core-math/PCA.h"
#include "core-math/blockedPCA.h"

namespace ml
{

//
// These should be moved back into vec1 -> vec6...
//
#ifdef LINUX
    template<>  const vec3f vec3f::origin;
    template<>  const vec3f vec3f::eX;
    template<>  const vec3f vec3f::eY;
    template<>  const vec3f vec3f::eZ;

    template<>  const vec3d vec3d::origin;
    template<>  const vec3d vec3d::eX;
    template<>  const vec3d vec3d::eY;
    template<>  const vec3d vec3d::eZ;
    template<>  const vec6d vec6d::origin;
    template<>  const vec6f vec6f::origin;

    template<>  const vec4f vec4f::origin;
    template<>  const vec4f vec4f::eX;
    template<>  const vec4f vec4f::eY;
    template<>  const vec4f vec4f::eZ;
    template<>  const vec4f vec4f::eW;

    template<>  const vec4d vec4d::origin;
    template<>  const vec4d vec4d::eX;
    template<>  const vec4d vec4d::eY;
    template<>  const vec4d vec4d::eZ;
    template<>  const vec4d vec4d::eW;

    template<>  const vec2f vec2f::origin;
    template<>  const vec2f vec2f::eX;
    template<>  const vec2f vec2f::eY;

    template<>  const vec2d vec2d::origin;
    template<>  const vec2d vec2d::eX;
    template<>  const vec2d vec2d::eY;

    template<>  const vec1f vec1f::origin;
    template<>  const vec1f vec1f::eX;

    template<>  const vec1d vec1d::origin;
    template<>  const vec1d vec1d::eX;
#else
    template<> const vec3f vec3f::origin(0.0f, 0.0f, 0.0f);
    template<> const vec3f vec3f::eX(1.0f, 0.0f, 0.0f);
    template<> const vec3f vec3f::eY(0.0f, 1.0f, 0.0f);
    template<> const vec3f vec3f::eZ(0.0f, 0.0f, 1.0f);

    template<> const vec3d vec3d::origin(0.0, 0.0, 0.0);
    template<> const vec3d vec3d::eX(1.0, 0.0, 0.0);
    template<> const vec3d vec3d::eY(0.0, 1.0, 0.0);
    template<> const vec3d vec3d::eZ(0.0, 0.0, 1.0);
    template<> const vec6d vec6d::origin(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    template<> const vec6f vec6f::origin(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);

    template<> const vec4f vec4f::origin(0.0f, 0.0f, 0.0f, 0.0f);
    template<> const vec4f vec4f::eX(1.0f, 0.0f, 0.0f, 0.0f);
    template<> const vec4f vec4f::eY(0.0f, 1.0f, 0.0f, 0.0f);
    template<> const vec4f vec4f::eZ(0.0f, 0.0f, 1.0f, 0.0f);
    template<> const vec4f vec4f::eW(0.0f, 0.0f, 0.0f, 1.0f);

    template<> const vec4d vec4d::origin(0.0, 0.0, 0.0, 0.0);
    template<> const vec4d vec4d::eX(1.0, 0.0, 0.0, 0.0);
    template<> const vec4d vec4d::eY(0.0, 1.0, 0.0, 0.0);
    template<> const vec4d vec4d::eZ(0.0, 0.0, 1.0, 0.0);
    template<> const vec4d vec4d::eW(0.0, 0.0, 0.0, 1.0);

    template<> const vec2f vec2f::origin(0.0f, 0.0f);
    template<> const vec2f vec2f::eX(1.0f, 0.0f);
    template<> const vec2f vec2f::eY(0.0f, 1.0f);

    template<> const vec2d vec2d::origin(0.0, 0.0);
    template<> const vec2d vec2d::eX(1.0, 0.0);
    template<> const vec2d vec2d::eY(0.0, 1.0);

    template<> const vec1f vec1f::origin(0.0f);
    template<> const vec1f vec1f::eX(1.0f);

    template<> const vec1d vec1d::origin(0.0);
    template<> const vec1d vec1d::eX(1.0);
#endif
}

//
// core-base headers
//
#include "core-base/grid2.h"
#include "core-base/grid3.h"

//
// core-util headers
//
#include "core-util/stringUtilConvert.h"
#include "core-util/directory.h"
#include "core-util/timer.h"
#include "core-util/nearestNeighborSearch.h"
#include "core-util/commandLineReader.h"
#include "core-util/parameterFile.h"
#include "core-util/keycodes.h"
#include "core-util/pipe.h"
#include "core-util/UIConnection.h"
#include "core-util/eventMap.h"
#include "core-util/sparseGrid3.h"
#include "core-base/binaryGrid3.h"

//
// core-multithreading headers
//
#include "core-multithreading/taskList.h"
#include "core-multithreading/workerThread.h"
#include "core-multithreading/threadPool.h"

//
// core-graphics headers
//
#include "core-graphics/RGBColor.h"
#include "core-graphics/ray.h"
#include "core-graphics/camera.h"
#include "core-graphics/cameraTrackball.h"
#include "core-graphics/lineSegment2.h"
#include "core-graphics/lineSegment3.h"
#include "core-graphics/line2.h"
#include "core-graphics/plane.h"
#include "core-graphics/triangle.h"
#include "core-graphics/intersection.h"
#include "core-graphics/polygon.h"
#include "core-graphics/boundingBox2.h"
#include "core-graphics/boundingBox3.h"
#include "core-graphics/orientedBoundingBox2.h"
#include "core-graphics/orientedBoundingBox3.h"
#include "core-graphics/dist.h"
#include "core-base/distanceField3.h"
#include "core-util/uniformAccelerator.h"
#include "core-base/baseImage.h"
#include "core-util/colorGradient.h"
#include "core-util/textWriter.h"
#include "core-graphics/colorUtils.h"

//
// core-mesh headers
//
// #include "core-mesh/material.h"
// #include "core-mesh/meshData.h"
// #include "core-mesh/plyHeader.h"
// #include "core-mesh/meshIO.h"
// #include "core-mesh/pointCloud.h"
// #include "core-mesh/pointCloudIO.h"

// #include "core-mesh/triMesh.h"
// #include "core-mesh/triMeshSampler.h"

// #include "core-mesh/triMeshAccelerator.h"
// #include "core-mesh/triMeshRayAccelerator.h"
// #include "core-mesh/triMeshCollisionAccelerator.h"
// #include "core-mesh/triMeshAcceleratorBruteForce.h"
// #include "core-mesh/triMeshAcceleratorBVH.h"

// #include "core-mesh/meshUtil.h"
// #include "core-mesh/meshShapes.h"

//
// core-network headers
//
#include "core-network/networkClient.h"
#include "core-network/networkServer.h"

#endif  // MLIBCORE_H_
