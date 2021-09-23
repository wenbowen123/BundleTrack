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

#ifndef FEATURE_MANAGER_H_
#define FEATURE_MANAGER_H_
#include <zmq.hpp>
#include <zmq_addon.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core/types.hpp>
#include "Utils.h"
#include "Frame.h"
#include "Bundler.h"

class Bundler;
class Frame;
class FramePairComparator;

typedef std::pair<std::shared_ptr<Frame>, std::shared_ptr<Frame>> FramePair;
typedef std::pair<int,int> IndexPair;
typedef std::map<FramePair, std::vector<IndexPair>, FramePairComparator, Eigen::aligned_allocator<std::pair<const FramePair, std::vector<IndexPair> > > > MatchMap;
typedef std::map<FramePair, Eigen::Matrix4f, FramePairComparator, Eigen::aligned_allocator<std::pair<const FramePair, Eigen::Matrix4f > > > PoseMap;




class MapPoint
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  std::map<std::shared_ptr<Frame>, std::pair<float,float>> _img_pt;

public:
  MapPoint();
  MapPoint::MapPoint(std::shared_ptr<Frame> frame, float u, float v);
  ~MapPoint();
};


class Correspondence
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  float _uA,_vA,_uB,_vB;
  pcl::PointXYZRGBNormal _ptA_cam, _ptB_cam;
  bool _isinlier;
  bool _ispropogated;

public:
  Correspondence(float uA, float vA, float uB, float vB, pcl::PointXYZRGBNormal ptA_cam, pcl::PointXYZRGBNormal ptB_cam, bool isinlier);
  ~Correspondence();
  bool operator == (const Correspondence &other) const;

};


class SiftManager
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  std::shared_ptr<YAML::Node> yml;
  cv::Ptr<cv::Feature2D> _detector;
  std::mt19937 _rng;
  Bundler *_bundler;

  std::map<FramePair, std::vector<Correspondence>> _matches;
  std::map<FramePair, std::vector<Correspondence>> _gt_matches;
  std::map<FramePair, std::vector<std::shared_ptr<MapPoint>> > _covisible_mappoints;
  std::vector<std::shared_ptr<MapPoint>> _map_points_global;


public:
  SiftManager(std::shared_ptr<YAML::Node> yml1, Bundler *bundler);
  ~SiftManager();
  void detectFeature(std::shared_ptr<Frame> frame);
  int countInlierCorres(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB);
  void vizKeyPoints(std::shared_ptr<Frame> frame);
  void updateFramePairMapPoints(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB);
  void findCorres(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB);
  void findCorresByMapPoints(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB);
  virtual void findCorresbyNN(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB);
  void pruneMatches(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB, const std::vector< std::vector<cv::DMatch> > &knn_matchesAB, std::vector<cv::DMatch> &matches_AB);
  void collectMutualMatches(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB, const std::vector<cv::DMatch> &matches_AB, const std::vector<cv::DMatch> &matches_BA);
  void findCorresbyNNMultiPair(std::vector<FramePair> &pairs);
  Eigen::Matrix4f procrustesByCorrespondence(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB,  const std::vector<Correspondence> &matches);
  void vizCorresBetween(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB, const std::string &name);
  void runRansacBetween(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB);
  void runRansacMultiPairGPU(const std::vector<std::pair<std::shared_ptr<Frame>, std::shared_ptr<Frame>>> &pairs);
  void forgetFrame(std::shared_ptr<Frame> frame);

};


class Lfnet : public SiftManager
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  zmq::context_t _context;
  zmq::socket_t _socket;

public:
  Lfnet(std::shared_ptr<YAML::Node> yml1, Bundler *bundler);
  ~Lfnet();
  void detectFeature(std::shared_ptr<Frame> frame, const float rot_deg=0);

};


#endif