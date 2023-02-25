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

#ifndef DATA_LOADER_H
#define DATA_LOADER_H


#include "Utils.h"
#include "Frame.h"
#include "CUDAImageUtil.h"

class Frame;

class DataLoaderBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  std::shared_ptr<YAML::Node> yml;
  Eigen::Matrix3f _K;
  Eigen::Matrix4f _ob_in_cam0;
  PointCloudRGBNormal::Ptr _real_model;
  pcl::PolygonMeshPtr _mesh;
  std::vector<std::string> _color_files, _gt_files;
  int _id;
  std::string _gt_dir;
  int _scene_id;
  std::string _model_name;
  std::string _model_dir;

public:
  DataLoaderBase(std::shared_ptr<YAML::Node> yml1);
  ~DataLoaderBase();
  bool hasNext();
};

class DataLoaderNOCS : public DataLoaderBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

public:
  DataLoaderNOCS(std::shared_ptr<YAML::Node> yml1);
  ~DataLoaderNOCS();
  std::shared_ptr<Frame> next();
  std::shared_ptr<Frame> getFrameByIndex(std::string id_str);
};

class DataLoaderYcbineoat : public DataLoaderBase
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  int _start_digit;
  std::string _id_str_prefix;

public:
  DataLoaderYcbineoat(std::shared_ptr<YAML::Node> yml1);
  ~DataLoaderYcbineoat();
  std::shared_ptr<Frame> next();
  std::shared_ptr<Frame> nextCustom();
};


#endif