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

#ifndef FRAME_HH_
#define FRAME_HH_
#include <cuda.h>
#include <cuda_runtime.h>
#include "Utils.h"
#include "FeatureManager.h"
#include <opencv2/core/cuda.hpp>

class MapPoint;

class Frame
{
public:
  enum Status
  {
    FAIL,
    NO_BA,
    OTHER,
  };

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  cv::Mat _color, _depth, _depth_raw, _depth_sim, _gray, _fg_mask, _vis;
  PointCloudRGBNormal::Ptr _cloud, _cloud_down, _real_model;
  Eigen::Matrix4f _pose_in_model;
  Eigen::Matrix3f _K;
  Eigen::Vector4f _roi;
  int _id;
  std::string _id_str;
  std::string _color_file;
  int _H, _W;
  std::shared_ptr<YAML::Node> yml;
  std::vector<cv::KeyPoint> _keypts;
  cv::Mat _feat_des;
  cv::cuda::GpuMat _feat_des_gpu;
  Status _status;
  bool _pose_inited;
  std::map<std::pair<float,float>, std::shared_ptr<MapPoint>> _map_points;

  float *_depth_gpu;
  uchar4 *_color_gpu;
  float4 *_normal_gpu;

  static zmq::context_t context;
  static zmq::socket_t socket;

  Frame();
  Frame(const cv::Mat &color, const cv::Mat &depth, const cv::Mat &depth_raw, const cv::Mat &depth_sim, const Eigen::Vector4f &roi, const Eigen::Matrix4f &pose_in_model, int id, std::string id_str, const Eigen::Matrix3f &K, std::shared_ptr<YAML::Node> yml1, PointCloudRGBNormal::Ptr cloud=NULL, PointCloudRGBNormal::Ptr real_model=NULL);
  ~Frame();
  void updateDepthCPU();
  void updateDepthGPU();
  void updateColorGPU();
  void updateNormalGPU();
  void processDepth();
  void depthToCloudAndNormals();
  void invalidatePixel(const int h, const int w);
  void invalidatePixelsByMask(const cv::Mat &fg_mask);
  void segmentationByGtPose();
  void Frame::segmentationByMaskFile();
  bool operator == (const Frame &other);
  bool operator < (const Frame &other);

};

class FramePtrComparator
{
public:
  bool operator () (const std::shared_ptr<Frame> &f1, const std::shared_ptr<Frame> &f2)
  {
    if (f1->_id < f2->_id) return true;
    return false;
  }
};

class FramePairComparator
{
public:
  typedef std::pair<std::shared_ptr<Frame>, std::shared_ptr<Frame>> FramePair;

  bool operator ()(const FramePair &p1, const FramePair &p2)
  {
    const int &id11 = p1.first->_id;
    const int &id12 = p1.second->_id;
    const int &id21 = p2.first->_id;
    const int &id22 = p2.second->_id;
    if (id11<id21) return true;
    if (id11>id21) return false;
    if (id12<id22) return true;
    return false;
  }
};



#endif