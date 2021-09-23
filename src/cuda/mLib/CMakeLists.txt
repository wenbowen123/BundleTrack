cmake_minimum_required(VERSION 3.2)
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
add_compile_options(-std=c++11)
include_directories(/home/manifold/CUDA_SLAM/BundleFusion_Ubuntu/external/mLib/include)


 
             
                     

