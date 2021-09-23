/*
Authors: Bowen Wen
Contact: wenbowenxjtu@gmail.com
Created in 2021

Copyright (c) Rutgers University, 2021 All rights reserved.

Bowen Wen and Kostas Bekris. "BundleTrack: 6D Pose Tracking for Novel Objects
 without Instance or Category-Level 3D Models."
 In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). 2021.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Bowen Wen, Kostas Bekris, Rutgers University,
      nor the names of its contributors may be used to
      endorse or promote products derived from this software without
      specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS' AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/



#ifndef COMMON_IO__HH
#define COMMON_IO__HH

// STL
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <cmath>
#include <time.h>
#include <queue>
#include <climits>
#include <boost/assign.hpp>
#include <boost/algorithm/string.hpp>
#include <thread>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <string>
#include <limits.h>
#include <unistd.h>
#include <memory>
#include <math.h>
#include <boost/format.hpp>
#include <numeric>
#include <thread>
#include <omp.h>
#include <exception>
#include <deque>
#include <random>

// For IO
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>

// For OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/rgbd.hpp>
#include "opencv2/calib3d/calib3d.hpp"

// For Visualization
// #include <pcl/visualization/pcl_visualizer.h>
// #include <pcl/visualization/cloud_viewer.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/common/pca.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/registration/correspondence_rejection.h>
#include <pcl/registration/correspondence_rejection_surface_normal.h>
#include <pcl/registration/icp.h>
#include <pcl/recognition/ransac_based/auxiliary.h>
#include <pcl/recognition/ransac_based/trimmed_icp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/surface/mls.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/bilateral.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/fast_bilateral_omp.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/lccp_segmentation.h>
#include <pcl/filters/passthrough.h>
#include <pcl/octree/octree_pointcloud.h>
#include <pcl/registration/transformation_estimation_point_to_plane_lls.h>
#include <pcl/registration/correspondence_estimation_normal_shooting.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/correspondence_estimation_backprojection.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/transformation_estimation_point_to_plane_weighted.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/filters/voxel_grid_occlusion_estimation.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/impl/point_types.hpp>
#include <pcl/search/impl/search.hpp>
#include <pcl/features/ppf.h>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/xpressive/xpressive.hpp>
#include <regex>
#include <pcl/features/integral_image_normal.h>
#include <pcl/tracking/normal_coherence.h>
#include <pcl/registration/default_convergence_criteria.h>
#include <pcl/features/principal_curvatures.h>
#include <boost/serialization/array.hpp>
#include "yaml-cpp/yaml.h"
#include <unordered_map>
#include <unsupported/Eigen/NonLinearOptimization>
#include <boost/filesystem.hpp>


#define EIGEN_DENSEBASE_PLUGIN "EigenDenseBaseAddons.h"


typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudRGB;
typedef pcl::PointCloud<pcl::PointXYZRGBNormal> PointCloudRGBNormal;
typedef pcl::PointCloud<pcl::PointNormal> PointCloudNormal;
typedef pcl::PointCloud<pcl::PointSurfel> PointCloudSurfel;
typedef pcl::PointCloud<pcl::PrincipalCurvatures> PointCloudCurvatures;
typedef std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f> > PoseVector;
using uchar = unsigned char;



namespace Utils
{
float rotationGeodesicDistance(const Eigen::Matrix3f &R1, const Eigen::Matrix3f &R2);
template<class PointT>
void convert3dOrganizedRGB(cv::Mat &objDepth, cv::Mat &colImage, Eigen::Matrix3f &camIntrinsic, boost::shared_ptr<pcl::PointCloud<PointT>> objCloud);
void readDepthImage(cv::Mat &depthImg, std::string path);
void readDirectory(const std::string& name, std::vector<std::string>& v);


template<class PointType>
void downsamplePointCloud(boost::shared_ptr<pcl::PointCloud<PointType> > cloud_in, boost::shared_ptr<pcl::PointCloud<PointType> > cloud_out, float vox_size);

void parsePoseTxt(std::string filename, Eigen::Matrix4f &out);

void normalizeRotationMatrix(Eigen::Matrix3f &R);
void normalizeRotationMatrix(Eigen::Matrix4f &pose);
bool isPixelInsideImage(const int H, const int W, float u, float v);


void solveRigidTransformBetweenPoints(const Eigen::MatrixXf &points1, const Eigen::MatrixXf &points2, Eigen::Matrix4f &pose);
void drawProjectPoints(PointCloudRGBNormal::Ptr cloud, const Eigen::Matrix3f &K, cv::Mat &out);

template<int rows, int cols>
void parseMatrixTxt(std::string filename, Eigen::Matrix<float,rows,cols> &out)
{
  using namespace std;
  std::vector<float> data;
  string line;
  ifstream file(filename);
  if (file.is_open())
  {
    while (getline(file, line))
    {
      std::stringstream ss(line);
      while (getline(ss, line, ' '))
      {
        if (line.size()>0)
          data.push_back(stof(line));
      }
    }
  }
  else
  {
    std::cout<<"opening failed: \n"<<filename<<std::endl;
  }
  for (int i=0;i<rows*cols;i++)
  {
    out(i/cols,i%cols) = data[i];
  }
}

template<typename Derived>
inline bool isMatrixFinite(const Eigen::MatrixBase<Derived>& x)
{
	return (x.array().isFinite()).all();
};


}

namespace pcl
{
 template <typename PointT, typename Scalar>
 inline PointT transformPointWithNormal(const PointT &point, const Eigen::Matrix<Scalar,4,4> &transform)
 {
   PointT ret = point;
   pcl::detail::Transformer<Scalar> tf (transform);
   tf.se3 (point.data, ret.data);
   tf.so3 (point.data_n, ret.data_n);
   return (ret);
 }

};




#endif