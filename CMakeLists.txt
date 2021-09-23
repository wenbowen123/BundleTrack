cmake_minimum_required(VERSION 3.10)
project(BundleTrack)


set(CMAKE_BUILD_TYPE Release)
add_compile_options(-std=c++14 -fpermissive -fPIC -g)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler;-fPIC;-gencode arch=compute_30,code=sm_30;-gencode arch=compute_61,code=sm_61;-gencode arch=compute_75,code=sm_75;-O3;-std=c++11;-use_fast_math;--default-stream per-thread)

if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
  set(COMPILE_OPTIONS -fopenmp -march=native -msse -msse2 -msse3 -msse4 -msse4.2 -mavx2 -ftree-vectorize)
  add_compile_options(${COMPILE_OPTIONS})
  message("Compile with " ${COMPILE_OPTIONS})
endif (${CMAKE_BUILD_TYPE} STREQUAL "Release")


# add_definitions(-DPRINT_RESIDUALS_DENSE=1)
# add_definitions(-DPRINT_RESIDUALS_SPARSE=1)

add_definitions(-DNO_OPENCV_CUDA=0)

find_package(Boost REQUIRED COMPONENTS system program_options)
find_package(PCL 1.9 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(BLAS REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)
find_package(GLUT REQUIRED)
find_package(LAPACK)


include_directories(
  src
  ${GLUT_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OPENGL_INCLUDE_DIR}
  ${CSPARSE_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/src/cuda/
  ${PROJECT_SOURCE_DIR}/src/cuda/Solver/
  ${PROJECT_SOURCE_DIR}/src/cuda/mLib/include
  ${PROJECT_SOURCE_DIR}/src/Thirdparty/cppzmq
  ${PROJECT_SOURCE_DIR}/src/Thirdparty
)

file(GLOB MY_SRC ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/cuda/*.cpp ${PROJECT_SOURCE_DIR}/src/cuda/Solver/*.cpp)

file(GLOB CUDA_FILES
  "${PROJECT_SOURCE_DIR}/src/*.cu"
  "${PROJECT_SOURCE_DIR}/src/cuda/*.cu"
  "${PROJECT_SOURCE_DIR}/src/cuda/Solver/*.cu"
)
CUDA_COMPILE(my_cuda_obj ${CUDA_FILES})
list(REMOVE_ITEM PCL_LIBRARIES pcl_simulation)
set(YAML_CPP_LIBRARIES "/usr/local/lib/libyaml-cpp.so")
add_library(${PROJECT_NAME} SHARED ${MY_SRC} ${my_cuda_obj})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${OpenMP_CXX_FLAGS} ${OPENGL_LIBRARY}  ${YAML_CPP_LIBRARIES} ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} zmq
)
add_executable(bundle_track_nocs  src/app/bundle_track_nocs.cpp)
target_link_libraries(bundle_track_nocs ${PROJECT_NAME} )

add_executable(bundle_track_ycbineoat  src/app/bundle_track_ycbineoat.cpp)
target_link_libraries(bundle_track_ycbineoat ${PROJECT_NAME} )
