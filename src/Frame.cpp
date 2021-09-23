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

#include "Frame.h"

zmq::context_t Frame::context;
zmq::socket_t Frame::socket;

Frame::Frame()
{

}

Frame::Frame(const cv::Mat &color, const cv::Mat &depth, const cv::Mat &depth_raw, const cv::Mat &depth_sim, const Eigen::Vector4f &roi, const Eigen::Matrix4f &pose_in_model, int id, std::string id_str, const Eigen::Matrix3f &K, std::shared_ptr<YAML::Node> yml1, PointCloudRGBNormal::Ptr cloud, PointCloudRGBNormal::Ptr real_model)
{
  _status = OTHER;
  yml = yml1;
  _color = color;
  _vis = color.clone();
  _depth = depth;
  _depth_raw = depth_raw;
  _depth_sim = depth_sim;
  _H = color.rows;
  _W = color.cols;
  _id = id;
  _id_str = id_str;
  _pose_in_model = pose_in_model;
  Utils::normalizeRotationMatrix(_pose_in_model);
  _K = K;
  _cloud = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();
  _cloud_down = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();
  _real_model = real_model;
  _pose_inited = false;
  _roi = roi;

  const int n_pixels = _H*_W;
  cudaMalloc(&_depth_gpu, n_pixels*sizeof(float));
  cudaMalloc(&_normal_gpu, n_pixels*sizeof(float4));
  cudaMalloc(&_color_gpu, n_pixels*sizeof(uchar4));


  cv::cvtColor(_color, _gray, CV_BGR2GRAY);

  updateDepthGPU();
  processDepth();

  updateColorGPU();

  depthToCloudAndNormals();

  if (!cloud)
  {
  }
  else
  {
    _cloud_down = cloud;
  }
}


Frame::~Frame()
{
  cudaFree(_depth_gpu);
  cudaFree(_normal_gpu);
  cudaFree(_color_gpu);
}

void Frame::updateDepthCPU()
{
  const int n_pixels = _H*_W;
  _depth = cv::Mat::zeros(1, n_pixels, CV_32F);
  cudaMemcpy(_depth.data, _depth_gpu, n_pixels*sizeof(float), cudaMemcpyDeviceToHost);
  _depth = _depth.reshape(1,_H);
}

void Frame::updateDepthGPU()
{
  const int n_pixels = _H*_W;
  cv::Mat depth_flat = _depth.reshape(1,1);
  cudaMemcpy(_depth_gpu, depth_flat.data, n_pixels*sizeof(float), cudaMemcpyHostToDevice);
}

void Frame::updateColorGPU()
{
  const int n_pixels = _H*_W;
  std::vector<uchar4> color_array(n_pixels);
  for (int h=0;h<_H;h++)
  {
    for (int w=0;w<_W;w++)
    {
      const auto &bgr = _color.at<cv::Vec3b>(h,w);
      color_array[h*_W+w] = make_uchar4(bgr[0],bgr[1],bgr[2],0);
    }
  }
  cudaMemcpy(_color_gpu, color_array.data(), sizeof(uchar4)*color_array.size(), cudaMemcpyHostToDevice);
}

void Frame::updateNormalGPU()
{
  const int n_pixels = _H*_W;
  std::vector<float4> normal_array(n_pixels);
  for (int h=0;h<_H;h++)
  {
    for (int w=0;w<_W;w++)
    {
      const auto &pt = (*_cloud)(w,h);
      if (pt.z>0.1)
      {
        normal_array[h*_W+w] = make_float4(pt.normal_x, pt.normal_y, pt.normal_z, 0);
      }
      else
      {
        normal_array[h*_W+w] = make_float4(0,0,0,0);
      }
    }
  }
  cudaMemcpy(_normal_gpu, normal_array.data(), sizeof(float4)*normal_array.size(), cudaMemcpyHostToDevice);
}


void Frame::processDepth()
{
  const int n_pixels = _H*_W;

  float *depth_tmp_gpu;
  cudaMalloc(&depth_tmp_gpu, n_pixels*sizeof(float));

  const float sigma_D = (*yml)["depth_processing"]["bilateral_filter"]["sigma_D"].as<float>();
  const float sigma_R = (*yml)["depth_processing"]["bilateral_filter"]["sigma_R"].as<float>();
  const int bf_radius = (*yml)["depth_processing"]["bilateral_filter"]["radius"].as<int>();
  const float erode_ratio = (*yml)["depth_processing"]["erode"]["ratio"].as<float>();
  const float erode_radius = (*yml)["depth_processing"]["erode"]["radius"].as<float>();
  const float erode_diff = (*yml)["depth_processing"]["erode"]["diff"].as<float>();

  CUDAImageUtil::erodeDepthMap(depth_tmp_gpu, _depth_gpu, erode_radius, _W,_H, erode_diff, erode_ratio);
  CUDAImageUtil::gaussFilterDepthMap(_depth_gpu, depth_tmp_gpu, bf_radius, sigma_D, sigma_R, _W, _H);
  CUDAImageUtil::gaussFilterDepthMap(depth_tmp_gpu, _depth_gpu, bf_radius, sigma_D, sigma_R, _W, _H);

  {
    float *tmp = _depth_gpu;
    _depth_gpu = depth_tmp_gpu;
    depth_tmp_gpu = tmp;
  }

  updateDepthCPU();

  cudaFree(depth_tmp_gpu);

}

void Frame::depthToCloudAndNormals()
{
  const int n_pixels = _H*_W;
  float4 *xyz_map_gpu;
  cudaMalloc(&xyz_map_gpu, n_pixels*sizeof(float4));
  float4x4 K_inv_data;
  K_inv_data.setIdentity();
  Eigen::Matrix3f K_inv = _K.inverse();
  for (int row=0;row<3;row++)
  {
    for (int col=0;col<3;col++)
    {
      K_inv_data(row,col) = K_inv(row,col);
    }
  }
  CUDAImageUtil::convertDepthFloatToCameraSpaceFloat4(xyz_map_gpu, _depth_gpu, K_inv_data, _W, _H);

  CUDAImageUtil::computeNormals(_normal_gpu, xyz_map_gpu, _W, _H);

  std::vector<float4> xyz_map(n_pixels);
  cudaMemcpy(xyz_map.data(), xyz_map_gpu, sizeof(float4)*n_pixels, cudaMemcpyDeviceToHost);
  std::vector<float4> normals(n_pixels);
  cudaMemcpy(normals.data(), _normal_gpu, sizeof(float4)*n_pixels, cudaMemcpyDeviceToHost);

  _cloud->height = _H;
  _cloud->width = _W;
  _cloud->is_dense = false;
  _cloud->points.resize(_cloud->width * _cloud->height);
  for (int h=0;h<_H;h++)
  {
    for (int w=0;w<_W;w++)
    {
      const int id = h*_W + w;
      const auto &xyz = xyz_map[id];
      (*_cloud)(w,h).x = xyz.x;
      (*_cloud)(w,h).y = xyz.y;
      (*_cloud)(w,h).z = xyz.z;

      const auto &color = _color.at<cv::Vec3b>(h,w);
      (*_cloud)(w,h).b = color[0];
      (*_cloud)(w,h).g = color[1];
      (*_cloud)(w,h).r = color[2];

      const auto &normal = normals[id];
      (*_cloud)(w,h).normal_x = normal.x;
      (*_cloud)(w,h).normal_y = normal.y;
      (*_cloud)(w,h).normal_z = normal.z;
    }
  }

  cudaFree(xyz_map_gpu);
}


void Frame::segmentationByMaskFile()
{
  const std::string data_dir = (*yml)["data_dir"].as<std::string>();
  int scene_id =-1;
  {
    std::regex pattern("scene_[0-9]");
    std::smatch what;
    if (std::regex_search(data_dir, what, pattern)) {
      std::string result = what[0];
      boost::replace_all(result, "scene_", "");
      scene_id = std::stoi(result);
    }
  }

  std::string mask_file;
  if (data_dir.find("NOCS")!=-1)
  {
    const std::string mask_dir = (*yml)["mask_dir"].as<std::string>();
    mask_file = mask_dir+"/"+_id_str+".png";
  }
  else
  {
    mask_file = data_dir+"/masks/"+_id_str+".png";
  }
  _fg_mask = cv::imread(mask_file, cv::IMREAD_UNCHANGED);
  if (_fg_mask.rows==0)
  {
    printf("mask file open failed: %s\n",mask_file.c_str());
    exit(1);
  }

  if (data_dir.find("NOCS")!=-1)
  {
    cv::Mat label;
    cv::connectedComponents(_fg_mask,label,8);
    std::unordered_map<int,int> hist;
    for (int h=0;h<_H;h++)
    {
      for (int w=0;w<_W;w++)
      {
        if (_fg_mask.at<uchar>(h,w)==0) continue;
        hist[label.at<int>(h,w)]++;
      }
    }
    int max_num = 0;
    int max_id = 0;
    for (const auto &h:hist)
    {
      if (h.second>max_num)
      {
        max_num = h.second;
        max_id = h.first;
      }
    }

    if (max_num>0)
    {
      std::vector<cv::Point2i> pts;
      for (int h=0;h<_H;h++)
      {
        for (int w=0;w<_W;w++)
        {
          if (label.at<int>(h,w)==max_id && _fg_mask.at<uchar>(h,w)>0)
          {
            pts.push_back({w,h});
          }
        }
      }
      _fg_mask = cv::Mat::zeros(_H,_W,CV_8UC1);
      std::vector<cv::Point2i> hull;
      cv::convexHull(pts,hull);
      cv::fillConvexPoly(_fg_mask,hull,1);
    }
    else
    {
      _fg_mask = cv::Mat::zeros(_H,_W,CV_8UC1);
    }
  }
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, {5,5});
  cv::dilate(_fg_mask,_fg_mask,kernel);

  invalidatePixelsByMask(_fg_mask);

}




void Frame::invalidatePixel(const int h, const int w)
{
  _color.at<cv::Vec3b>(h,w) = {0,0,0};
  _depth.at<float>(h,w) = 0;
  _depth_sim.at<float>(h,w) = 0;
  _depth_raw.at<float>(h,w) = 0;
  _gray.at<uchar>(h,w) = 0;
  {
    auto &pt = (*_cloud)(w,h);
    pt.x = 0;
    pt.y = 0;
    pt.z = 0;
    pt.normal_x = 0;
    pt.normal_y = 0;
    pt.normal_z = 0;
  }
}

void Frame::invalidatePixelsByMask(const cv::Mat &fg_mask)
{
  assert(fg_mask.rows==_H && fg_mask.cols==_W);
  for (int h=0;h<_H;h++)
  {
    for (int w=0;w<_W;w++)
    {
      if (fg_mask.at<uchar>(h,w)==0)
      {
        invalidatePixel(h,w);
      }
    }
  }
  updateColorGPU();
  updateDepthGPU();
  updateNormalGPU();

  _roi<<9999,0,9999,0;
  for (int h=0;h<_H;h++)
  {
    for (int w=0;w<_W;w++)
    {
      if (fg_mask.at<uchar>(h,w)>0)
      {
        _roi(0) = std::min(_roi(0), float(w));
        _roi(1) = std::max(_roi(1), float(w));
        _roi(2) = std::min(_roi(2), float(h));
        _roi(3) = std::max(_roi(3), float(h));
      }
    }
  }
  _fg_mask = fg_mask.clone();
}

bool Frame::operator == (const Frame &other)
{
  if (std::stoi(_id_str)==std::stoi(other._id_str)) return true;
  return false;
}


bool Frame::operator < (const Frame &other)
{
  if (std::stoi(_id_str)<std::stoi(other._id_str)) return true;
  return false;
}

