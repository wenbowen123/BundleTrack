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

#include "FeatureManager.h"
#include "cuda_ransac.h"
#include <opencv2/cudafeatures2d.hpp>


using namespace std;



MapPoint::MapPoint()
{

}

MapPoint::MapPoint(std::shared_ptr<Frame> frame, float u, float v)
{
  _img_pt[frame] = {u,v};
}

MapPoint::~MapPoint()
{

}


Correspondence::Correspondence(float uA, float vA, float uB, float vB, pcl::PointXYZRGBNormal ptA_cam, pcl::PointXYZRGBNormal ptB_cam, bool isinlier) : _uA(uA), _uB(uB), _vA(vA), _vB(vB), _ptA_cam(ptA_cam), _ptB_cam(ptB_cam), _isinlier(isinlier), _ispropogated(false)
{

}

Correspondence::~Correspondence()
{

}

bool Correspondence::operator == (const Correspondence &other) const
{
  if (_uA==other._uA && _uB==other._uB && _vA==other._vA && _vB==other._vB) return true;
  return false;
}


SiftManager::SiftManager(std::shared_ptr<YAML::Node> yml1, Bundler *bundler) : _rng(std::random_device{}())
{
  _bundler = bundler;
  yml = yml1;
  srand(0);
  _rng.seed(0);

  // _detector = cv::xfeatures2d::SIFT::create(0,
  //                                         (*yml)["sift"]["nOctaveLayers"].as<int>(),
  //                                         (*yml)["sift"]["contrastThreshold"].as<float>(),
  //                                         (*yml)["sift"]["edgeThreshold"].as<float>(),
  //                                         (*yml)["sift"]["sigma"].as<float>());

}

SiftManager::~SiftManager()
{

}


void SiftManager::detectFeature(std::shared_ptr<Frame> frame)
{
  // if (frame->_keypts.size()>0) return;
  // std::vector<float> scales = (*yml)["sift"]["scales"].as<std::vector<float>>();
  // for (int i=0;i<scales.size();i++)
  // {
  //   const auto &scale = scales[i];
  //   cv::Mat cur;
  //   cv::resize(frame->_gray, cur, {0,0}, scale,scale);
  //   std::vector<cv::KeyPoint> keypts;
  //   cv::Mat des;
  //   _detector->detectAndCompute(cur, cv::noArray(), keypts, des);
  //   for (int ii=0;ii<keypts.size();ii++)
  //   {
  //     keypts[ii].pt.x = keypts[ii].pt.x/scale;
  //     keypts[ii].pt.y = keypts[ii].pt.y/scale;
  //   }
  //   frame->_keypts.insert(frame->_keypts.end(),keypts.begin(),keypts.end());
  //   if (frame->_feat_des.rows>0)
  //     cv::vconcat(frame->_feat_des,des,frame->_feat_des);
  //   else
  //     frame->_feat_des = des;
  // }

}


void SiftManager::vizKeyPoints(std::shared_ptr<Frame> frame)
{
  const std::string out_dir = (*yml)["debug_dir"].as<std::string>()+"/"+frame->_id_str+"/";
  cv::Mat out = frame->_color.clone();
  cv::drawKeypoints(out, frame->_keypts, out);
  cv::imwrite(out_dir+"keypoints.jpg", out);

  const auto &kpts = frame->_keypts;
  std::ofstream ff(out_dir+"keypoints.txt");
  for (int i=0;i<kpts.size();i++)
  {
    ff<<kpts[i].pt.x<<" "<<kpts[i].pt.y<<std::endl;
  }
  ff.close();
}


void SiftManager::forgetFrame(std::shared_ptr<Frame> frame)
{
  auto _matches_tmp = _matches;
  for (const auto& h:_matches_tmp)
  {
    const auto& f_pair = h.first;
    if (f_pair.first->_id==frame->_id || f_pair.second->_id==frame->_id)
    {
      _matches.erase(f_pair);
      _gt_matches.erase(f_pair);
    }
  }
  auto _covisible_mappoints_tmp = _covisible_mappoints;
  for (const auto &h:_covisible_mappoints_tmp)
  {
    const auto& f_pair = h.first;
    if (f_pair.first->_id==frame->_id || f_pair.second->_id==frame->_id)
    {
      _covisible_mappoints.erase(f_pair);
    }
  }
  for (const auto &mpt:_map_points_global)
  {
    if (mpt->_img_pt.find(frame)!=mpt->_img_pt.end())
    {
      mpt->_img_pt.erase(frame);
    }
  }
}


void SiftManager::findCorres(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  assert(frameA->_id > frameB->_id);
  if (_matches.find({frameA,frameB})!=_matches.end()) return;

  bool is_neighbor = std::abs(frameA->_id-frameB->_id)==1;
  printf("finding corres between %s(id=%d) and %s(id=%d)\n", frameA->_id_str.c_str(), frameA->_id, frameB->_id_str.c_str(), frameB->_id);

  if (is_neighbor)
  {
    findCorresbyNN(frameA,frameB);
    vizCorresBetween(frameA, frameB, "before_ransac");

    if (frameA->_status==Frame::FAIL)
    {
      return;
    }

    runRansacBetween(frameA, frameB);
    vizCorresBetween(frameA, frameB, "after_ransac");

    if (frameA->_status==Frame::FAIL)
    {
      return;
    }

    updateFramePairMapPoints(frameA,frameB);
    vizCorresBetween(frameA, frameB, "after_mappoints");
  }
  else
  {
    findCorresbyNN(frameA,frameB);
    vizCorresBetween(frameA, frameB, "before_ransac");

    if (frameA->_status==Frame::FAIL)
    {
      return;
    }

    int num_matches = _matches[{frameA,frameB}].size();
    findCorresByMapPoints(frameA,frameB);
    vizCorresBetween(frameA, frameB, "after_mappoints");

    if (frameA->_status==Frame::FAIL)
    {
      return;
    }

    runRansacBetween(frameA, frameB);
    vizCorresBetween(frameA, frameB, "after_ransac");

    if (frameA->_status==Frame::FAIL)
    {
      return;
    }

    updateFramePairMapPoints(frameA,frameB);

  }

  if (_matches[{frameA,frameB}].size()<5)
  {
    _matches[{frameA,frameB}].clear();
    if (is_neighbor)
    {
      frameA->_status = Frame::FAIL;
      printf("frame %s is marked FAIL since matches between %s and %s is too few\n", frameA->_id_str.c_str(),frameA->_id_str.c_str(),frameB->_id_str.c_str());
    }
  }
}




void SiftManager::findCorresbyNN(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  assert(frameA->_id > frameB->_id);
  if (frameA->_keypts.size()==0 || frameB->_keypts.size()==0) return;

  const float max_dist_no_neighbor = (*yml)["feature_corres"]["max_dist_no_neighbor"].as<float>();
  const float cos_max_normal_no_neighbor = std::cos((*yml)["feature_corres"]["max_normal_no_neighbor"].as<float>()/180.0*M_PI);
  const float max_dist_neighbor = (*yml)["feature_corres"]["max_dist_neighbor"].as<float>();
  const float cos_max_normal_neighbor = std::cos((*yml)["feature_corres"]["max_normal_neighbor"].as<float>()/180.0*M_PI);

  const int H = frameA->_H;
  const int W = frameA->_W;

  bool is_neighbor = std::abs(frameA->_id-frameB->_id)==1;

  std::vector<cv::DMatch> matches_AB, matches_BA;
  std::vector< std::vector<cv::DMatch> > knn_matchesAB, knn_matchesBA;
  const int k_near = 5;

#if NO_OPENCV_CUDA
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
  matcher->knnMatch( frameA->_feat_des, frameB->_feat_des, knn_matchesAB, k_near);
  matcher->knnMatch( frameB->_feat_des, frameA->_feat_des, knn_matchesBA, k_near);
#else
  cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
  matcher->knnMatch( frameA->_feat_des_gpu, frameB->_feat_des_gpu, knn_matchesAB, k_near);
  matcher->knnMatch( frameB->_feat_des_gpu, frameA->_feat_des_gpu, knn_matchesBA, k_near);
#endif

  pruneMatches(frameA,frameB,knn_matchesAB,matches_AB);

  pruneMatches(frameB,frameA,knn_matchesBA,matches_BA);

  collectMutualMatches(frameA,frameB,matches_AB,matches_BA);

  if (_matches[{frameA,frameB}].size()<5 && is_neighbor)
  {
    frameA->_status = Frame::FAIL;
    printf("frame %s and %s findNN too few match, %s status marked as FAIL", frameA->_id_str.c_str(), frameB->_id_str.c_str(), frameA->_id_str.c_str());
  }

}

void SiftManager::pruneMatches(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB, const std::vector< std::vector<cv::DMatch> > &knn_matchesAB, std::vector<cv::DMatch> &matches_AB)
{
  const float max_dist_no_neighbor = (*yml)["feature_corres"]["max_dist_no_neighbor"].as<float>();
  const float cos_max_normal_no_neighbor = std::cos((*yml)["feature_corres"]["max_normal_no_neighbor"].as<float>()/180.0*M_PI);
  const float max_dist_neighbor = (*yml)["feature_corres"]["max_dist_neighbor"].as<float>();
  const float cos_max_normal_neighbor = std::cos((*yml)["feature_corres"]["max_normal_neighbor"].as<float>()/180.0*M_PI);

  const int H = frameA->_H;
  const int W = frameA->_W;

  bool is_neighbor = std::abs(frameA->_id-frameB->_id)==1;

  matches_AB.clear();
  matches_AB.reserve(knn_matchesAB.size());
  for (int i=0;i<knn_matchesAB.size();i++)
  {
    for (int k=0;k<knn_matchesAB[i].size();k++)
    {
      const auto &match = knn_matchesAB[i][k];
      auto pA = frameA->_keypts[match.queryIdx].pt;
      auto pB = frameB->_keypts[match.trainIdx].pt;
      int uA = std::round(pA.x);
      int vA = std::round(pA.y);
      int uB = std::round(pB.x);
      int vB = std::round(pB.y);
      if (!Utils::isPixelInsideImage(H, W, uA, vA) || !Utils::isPixelInsideImage(H, W, uB, vB)) continue;
      const auto &ptA = (*frameA->_cloud)(uA, vA);
      const auto &ptB = (*frameB->_cloud)(uB, vB);
      if (ptA.z<0.1 || ptB.z<0.1) continue;
      auto PA_world = pcl::transformPointWithNormal(ptA, frameA->_pose_in_model);
      auto PB_world = pcl::transformPointWithNormal(ptB, frameB->_pose_in_model);
      float dist = pcl::geometry::distance(PA_world, PB_world);
      Eigen::Vector3f n1(PA_world.normal_x,PA_world.normal_y,PA_world.normal_z);
      Eigen::Vector3f n2(PB_world.normal_x,PB_world.normal_y,PB_world.normal_z);
      if (!is_neighbor)
      {
        if (dist>max_dist_no_neighbor || n1.normalized().dot(n2.normalized())<cos_max_normal_no_neighbor) continue;
      }
      else
      {
        if (dist>max_dist_neighbor || n1.normalized().dot(n2.normalized())<cos_max_normal_neighbor) continue;
      }
      matches_AB.push_back(match);
      break;
    }
  }
}

void SiftManager::collectMutualMatches(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB, const std::vector<cv::DMatch> &matches_AB, const std::vector<cv::DMatch> &matches_BA)
{
  auto &matches = _matches[{frameA, frameB}];

  for (int i=0;i<matches_AB.size();i++)
  {
    int Aid = matches_AB[i].queryIdx;
    int Bid = matches_AB[i].trainIdx;
    float uA = frameA->_keypts[Aid].pt.x;
    float vA = frameA->_keypts[Aid].pt.y;
    float uB = frameB->_keypts[Bid].pt.x;
    float vB = frameB->_keypts[Bid].pt.y;
    const auto &ptA = frameA->_cloud->at(std::round(uA), std::round(vA));
    const auto &ptB = frameB->_cloud->at(std::round(uB), std::round(vB));
    Correspondence corr(uA,vA,uB,vB, ptA, ptB, true);
    matches.push_back(corr);
  }
  for (int i=0;i<matches_BA.size();i++)
  {
    int Aid = matches_BA[i].trainIdx;
    int Bid = matches_BA[i].queryIdx;
    float uA = frameA->_keypts[Aid].pt.x;
    float vA = frameA->_keypts[Aid].pt.y;
    float uB = frameB->_keypts[Bid].pt.x;
    float vB = frameB->_keypts[Bid].pt.y;
    const auto &ptA = frameA->_cloud->at(std::round(uA), std::round(vA));
    const auto &ptB = frameB->_cloud->at(std::round(uB), std::round(vB));
    Correspondence corr(uA,vA,uB,vB, ptA, ptB, true);
    matches.push_back(corr);
  }
}

void SiftManager::findCorresbyNNMultiPair(std::vector<FramePair> &pairs)
{
  const bool mutual = (*yml)["feature_corres"]["mutual"].as<bool>();
  const float max_dist_no_neighbor = (*yml)["feature_corres"]["max_dist_no_neighbor"].as<float>();
  const float cos_max_normal_no_neighbor = std::cos((*yml)["feature_corres"]["max_normal_no_neighbor"].as<float>()/180.0*M_PI);
  const float max_dist_neighbor = (*yml)["feature_corres"]["max_dist_neighbor"].as<float>();
  const float cos_max_normal_neighbor = std::cos((*yml)["feature_corres"]["max_normal_neighbor"].as<float>()/180.0*M_PI);
  const int k_near = 5;


#if NO_OPENCV_CUDA
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE);
  std::vector<std::vector<std::vector<cv::DMatch>>> matchesABs(pairs.size()), matchesBAs(pairs.size());
#else
  cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_L2);
  std::vector<cv::cuda::Stream> streams(pairs.size()*2);
  std::vector<cv::cuda::GpuMat> matchesAB_gpus(pairs.size()), matchesBA_gpus(pairs.size());
#endif

  for (int i=0;i<pairs.size();i++)
  {
    auto &fA = pairs[i].first;
    auto &fB = pairs[i].second;
#if NO_OPENCV_CUDA
    matcher->knnMatch(fA->_feat_des, fB->_feat_des, matchesABs[i], k_near);
    if (mutual)
    {
      matcher->knnMatch(fB->_feat_des, fA->_feat_des, matchesBAs[i], k_near);
    }
#else
    matcher->knnMatchAsync(fA->_feat_des_gpu, fB->_feat_des_gpu, matchesAB_gpus[i], k_near, cv::noArray(), streams[2*i]);
    if (mutual)
    {
      matcher->knnMatchAsync(fB->_feat_des_gpu, fA->_feat_des_gpu, matchesBA_gpus[i], k_near, cv::noArray(), streams[2*i+1]);
    }
#endif
  }

#if NO_OPENCV_CUDA==0
  for (int i=0;i<pairs.size();i++)
  {
    streams[2*i].waitForCompletion();
    streams[2*i+1].waitForCompletion();
  }
#endif

  for (int i=0;i<pairs.size();i++)
  {
    auto &fA = pairs[i].first;
    auto &fB = pairs[i].second;
    std::vector< std::vector<cv::DMatch> > knn_matchesAB, knn_matchesBA;

#if NO_OPENCV_CUDA
    knn_matchesAB = matchesABs[i];
    if (mutual)
    {
      knn_matchesBA = matchesBAs[i];
    }
#else
    matcher->knnMatchConvert(matchesAB_gpus[i], knn_matchesAB);
    if (mutual)
    {
      matcher->knnMatchConvert(matchesBA_gpus[i], knn_matchesBA);
    }
#endif
    std::vector<cv::DMatch> matchesAB, matchesBA;
    pruneMatches(fA,fB,knn_matchesAB,matchesAB);
    if (mutual)
    {
      pruneMatches(fB,fA,knn_matchesBA,matchesBA);
    }
    collectMutualMatches(fA,fB,matchesAB,matchesBA);
  }

}



void SiftManager::updateFramePairMapPoints(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  assert(frameA->_id > frameB->_id);
  const auto &matches = _matches[{frameA,frameB}];
  for (int i=0;i<matches.size();i++)
  {
    const auto &match = matches[i];
    if (match._isinlier==false) continue;
    const float uA = match._uA;
    const float vA = match._vA;
    const float uB = match._uB;
    const float vB = match._vB;

    if (frameA->_map_points.find({uA,vA})!=frameA->_map_points.end() && frameB->_map_points.find({uB,vB})!=frameB->_map_points.end()) continue;

    std::shared_ptr<MapPoint> mpt;
    bool existed = false;
    if (frameB->_map_points.find({uB,vB})==frameB->_map_points.end())
    {
      mpt = std::make_shared<MapPoint>(frameB,uB,vB);
      frameB->_map_points[{uB,vB}] = mpt;
      mpt->_img_pt[frameB] = {uB,vB};
      existed = false;
    }
    else
    {
      mpt = frameB->_map_points[{uB,vB}];
      existed = true;
    }
    mpt->_img_pt[frameA] = {uA,vA};
    frameA->_map_points[{uA,vA}] = mpt;

    if (!existed)
    {
      _map_points_global.push_back(mpt);
    }
  }
}



void SiftManager::findCorresByMapPoints(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  assert(frameA->_id > frameB->_id);
  auto &matches = _matches[{frameA,frameB}];
  for (const auto &h:frameA->_map_points)
  {
    const auto &uvA = h.first;
    const auto &mpt = h.second;
    if (mpt->_img_pt.find(frameB)==mpt->_img_pt.end()) continue;
    const auto &uvB = mpt->_img_pt[frameB];
    const auto &pA = frameA->_cloud->at(std::round(uvA.first), std::round(uvA.second));
    const auto &pB = frameB->_cloud->at(std::round(uvB.first), std::round(uvB.second));
    Correspondence match(uvA.first, uvA.second, uvB.first, uvB.second, pA, pB, true);
    bool existed = false;
    for (int i=0;i<matches.size();i++)
    {
      if (matches[i]._uA==match._uA && matches[i]._vA==match._vA)
      {
        existed = true;
        break;
      }
      if (matches[i]._uB==match._uB && matches[i]._vB==match._vB)
      {
        existed = true;
        break;
      }
    }
    if (existed) continue;
    match._ispropogated = true;
    matches.push_back(match);
  }
}


Eigen::Matrix4f SiftManager::procrustesByCorrespondence(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB, const std::vector<Correspondence> &matches)
{
  assert(frameA->_id > frameB->_id);
  Eigen::Matrix4f pose(Eigen::Matrix4f::Identity());
  if (countInlierCorres(frameA,frameB)<5) return pose;

  Eigen::MatrixXf src(Eigen::MatrixXf::Constant(matches.size(),3,1));
  Eigen::MatrixXf dst(Eigen::MatrixXf::Constant(matches.size(),3,1));
  for (int i=0;i<matches.size();i++)
  {
    const auto &match = matches[i];
    if (!match._isinlier) continue;
    auto pcl_pt1 = match._ptA_cam;
    auto pcl_pt2 = match._ptB_cam;
    pcl_pt1 = pcl::transformPointWithNormal<pcl::PointXYZRGBNormal>(pcl_pt1,frameA->_pose_in_model);
    pcl_pt2 = pcl::transformPointWithNormal<pcl::PointXYZRGBNormal>(pcl_pt2,frameB->_pose_in_model);
    src.row(i) << pcl_pt1.x, pcl_pt1.y, pcl_pt1.z;
    dst.row(i) << pcl_pt2.x, pcl_pt2.y, pcl_pt2.z;
  }

  Utils::solveRigidTransformBetweenPoints(src, dst, pose);
  if (pose==Eigen::Matrix4f::Identity()) return;
  Eigen::MatrixXf src_est = src;
  for (int i=0;i<src_est.rows();i++)
  {
    src_est.row(i) = pose.block(0,0,3,3)*src_est.row(i).transpose() + pose.block(0,3,3,1);
  }
  float err = (src_est - dst).norm()/src_est.rows();
  if (frameB->_id-frameA->_id==1 && err>1e-3)
  {
    printf("ERROR too big\n");
    std::abort();
  }
  return pose;
}



void SiftManager::runRansacBetween(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  assert(frameA->_id > frameB->_id);

  const int num_sample = (*yml)["ransac"]["num_sample"].as<int>();
  const int max_iter = (*yml)["ransac"]["max_iter"].as<int>();
  const float inlier_dist = (*yml)["ransac"]["inlier_dist"].as<float>();
  const float cos_normal_angle = std::cos((*yml)["ransac"]["inlier_normal_angle"].as<float>()/180.0*M_PI);
  const float max_rot_deg_neighbor = (*yml)["ransac"]["max_rot_deg_neighbor"].as<float>();
  const float max_trans_neighbor = (*yml)["ransac"]["max_trans_neighbor"].as<float>();
  const float max_trans_no_neighbor = (*yml)["ransac"]["max_trans_no_neighbor"].as<float>();
  const float max_rot_no_neighbor = (*yml)["ransac"]["max_rot_no_neighbor"].as<float>();


  if (countInlierCorres(frameA, frameB)<=5)
  {
    _matches[{frameA, frameB}].clear();
    return;
  }
  const auto matches = _matches[{frameA, frameB}];
  std::vector<int> indices(matches.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::vector<int> propogated_indices;
  propogated_indices.reserve(matches.size());
  for (int i=0;i<matches.size();i++)
  {
    const auto &m = matches[i];
    if (m._ispropogated)
    {
      propogated_indices.push_back(i);
    }
  }
  std::vector<std::vector<int>> propogated_samples;
  for (int i=0;i<propogated_indices.size() && propogated_samples.size()<max_iter;i++)
  {
    for (int j=i+1;j<propogated_indices.size() && propogated_samples.size()<max_iter;j++)
    {
      for (int k=j+1;k<propogated_indices.size() && propogated_samples.size()<max_iter;k++)
      {
        propogated_samples.push_back({propogated_indices[i],propogated_indices[j],propogated_indices[k]});
      }
    }
  }

  bool is_neighbor = std::abs(frameA->_id-frameB->_id)==1;

  std::map<std::vector<int>, bool> tried_sample;
  int max_num_comb = 1;
  for (int i=0;i<num_sample;i++)
  {
    max_num_comb *= matches.size()-i;
  }
  for (int i=0;i<num_sample-1;i++)
  {
    max_num_comb /= i+1;
  }

  std::vector<std::vector<int>> samples_to_try;
  samples_to_try.reserve(max_iter);
  for (int i=0;i<max_iter;i++)
  {
    if (tried_sample.size()==max_num_comb) break;
    std::vector<int> chosen_match_ids;
    if (i<propogated_samples.size())
    {
      chosen_match_ids = propogated_samples[i];
    }
    else
    {

      while (chosen_match_ids.size()<num_sample)
      {
        int id = rand()%matches.size();
        if (std::find(chosen_match_ids.begin(),chosen_match_ids.end(),id)==chosen_match_ids.end())
          chosen_match_ids.push_back(id);
      }

      std::sort(chosen_match_ids.begin(),chosen_match_ids.end());
    }

    if (tried_sample.find(chosen_match_ids)!=tried_sample.end())
    {
      continue;
    }
    tried_sample[chosen_match_ids] = true;
    samples_to_try.push_back(chosen_match_ids);
  }

  std::vector<Correspondence> inliers;

  runRansacMultiPairGPU({{frameA, frameB}});
}





void SiftManager::runRansacMultiPairGPU(const std::vector<std::pair<std::shared_ptr<Frame>, std::shared_ptr<Frame>>> &pairs)
{
  printf("start multi pair ransac GPU, pairs#=%d\n", pairs.size());

  const int num_sample = (*yml)["ransac"]["num_sample"].as<int>();
  const int max_iter = (*yml)["ransac"]["max_iter"].as<int>();
  const float inlier_dist = (*yml)["ransac"]["inlier_dist"].as<float>();
  const float cos_normal_angle = std::cos((*yml)["ransac"]["inlier_normal_angle"].as<float>()/180.0*M_PI);
  const float max_rot_deg_neighbor = (*yml)["ransac"]["max_rot_deg_neighbor"].as<float>();
  const float max_trans_neighbor = (*yml)["ransac"]["max_trans_neighbor"].as<float>();
  const float max_trans_no_neighbor = (*yml)["ransac"]["max_trans_no_neighbor"].as<float>();
  const float max_rot_no_neighbor = (*yml)["ransac"]["max_rot_no_neighbor"].as<float>();

  std::vector<float4*> ptsA_gpu(pairs.size()), ptsB_gpu(pairs.size());
  std::vector<std::vector<int>> ids;
  std::vector<int> n_pts;

  for (int i_pair=0;i_pair<pairs.size();i_pair++)
  {
    const auto pair = pairs[i_pair];
    std::vector<float4> ptsA_cur_pair, ptsB_cur_pair;

    const auto &frameA = pair.first;
    const auto &frameB = pair.second;
    const auto &matches = _matches[{frameA, frameB}];

    std::vector<int> ids_cur_pair;
    for (int i=0;i<matches.size();i++)
    {
      const auto &match = matches[i];
      if (match._isinlier==false) continue;
      const auto pA = pcl::transformPointWithNormal(match._ptA_cam, frameA->_pose_in_model);
      const auto pB = pcl::transformPointWithNormal(match._ptB_cam, frameB->_pose_in_model);
      ptsA_cur_pair.push_back(make_float4(pA.x, pA.y, pA.z, 1));
      ptsB_cur_pair.push_back(make_float4(pB.x, pB.y, pB.z, 1));
      ids_cur_pair.push_back(i);
    }


    n_pts.push_back(int(ptsA_cur_pair.size()));
    ids.push_back(ids_cur_pair);

    cudaMalloc(&ptsA_gpu[i_pair], sizeof(float4)*ptsA_cur_pair.size());
    cudaMalloc(&ptsB_gpu[i_pair], sizeof(float4)*ptsB_cur_pair.size());

    cutilSafeCall(cudaMemcpy(ptsA_gpu[i_pair], ptsA_cur_pair.data(), sizeof(float4)*ptsA_cur_pair.size(), cudaMemcpyHostToDevice));
    cutilSafeCall(cudaMemcpy(ptsB_gpu[i_pair], ptsB_cur_pair.data(), sizeof(float4)*ptsB_cur_pair.size(), cudaMemcpyHostToDevice));
  }


  Eigen::Matrix4f best_pose(Eigen::Matrix4f::Identity());
  std::vector<std::vector<int>> inlier_ids;


  ransacMultiPairGPU(ptsA_gpu, ptsB_gpu, n_pts, max_iter, inlier_dist, inlier_ids);

  for (int i=0;i<pairs.size();i++)
  {
    const auto &frameA = pairs[i].first;
    const auto &frameB = pairs[i].second;
    const auto &inlier_ids_cur_pair = inlier_ids[i];
    const auto &ids_cur_pair = ids[i];
    auto &matches_cur_pair = _matches[{frameA, frameB}];
    std::vector<Correspondence> matches_new;
    for (int i_inlier=0;i_inlier<inlier_ids_cur_pair.size();i_inlier++)
    {
      matches_new.push_back(matches_cur_pair[ids_cur_pair[inlier_ids_cur_pair[i_inlier]]]);
    }
    matches_cur_pair = matches_new;
    if (matches_cur_pair.size()<5)
    {
      matches_cur_pair.clear();
    }
  }

  for (int i=0;i<pairs.size();i++)
  {
    cutilSafeCall(cudaFree(ptsA_gpu[i]));
    cutilSafeCall(cudaFree(ptsB_gpu[i]));
  }


}




int SiftManager::countInlierCorres(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB)
{
  int cnt = 0;
  const auto &corres = _matches[{frameA, frameB}];
  for (int i=0;i<corres.size();i++)
  {
    if (corres[i]._isinlier)
    {
      cnt++;
    }
  }
  return cnt;
}

void SiftManager::vizCorresBetween(std::shared_ptr<Frame> frameA, std::shared_ptr<Frame> frameB, const std::string &name)
{
  if ((*yml)["LOG"].as<int>()<1) return;
  const std::string out_dir = (*yml)["debug_dir"].as<std::string>()+"/"+_bundler->_newframe->_id_str+"/";
  const std::string out_match_file = out_dir+frameA->_id_str+"_match_"+frameB->_id_str+"_"+name+".jpg";
  std::ifstream tmp(out_match_file);
  cv::Mat colorA = frameA->_color.clone();
  cv::Mat colorB = frameB->_color.clone();
  const auto &corres = _matches[{frameA,frameB}];
  const auto& corneri = frameA->_roi;
  const auto& cornerj = frameB->_roi;

  colorA = colorA(cv::Rect(corneri(0),corneri(2),corneri(1)-corneri(0),corneri(3)-corneri(2)));
  colorB = colorB(cv::Rect(cornerj(0),cornerj(2),cornerj(1)-cornerj(0),cornerj(3)-cornerj(2)));
  const float scale = std::min(900/colorA.rows, 900/colorB.rows);
  cv::resize(colorA, colorA,{0,0},scale,scale);
  cv::resize(colorB, colorB,{0,0},scale,scale);

  std::vector<cv::DMatch> cv_matches;
  std::vector<cv::KeyPoint> kptsA,kptsB;
  for (int i=0;i<corres.size();i++)
  {
    if (!corres[i]._isinlier) continue;
    float uA = (corres[i]._uA-corneri(0))*scale;
    float vA = (corres[i]._vA-corneri(2))*scale;
    float uB = (corres[i]._uB-cornerj(0))*scale;
    float vB = (corres[i]._vB-cornerj(2))*scale;
    kptsA.push_back(cv::KeyPoint(uA,vA, 1));
    kptsB.push_back(cv::KeyPoint(uB,vB, 1));
    cv_matches.push_back(cv::DMatch(kptsA.size()-1,kptsA.size()-1,0));
  }

  cv::Mat out;
  cv::drawMatches( colorA, kptsA, colorB, kptsB, cv_matches, out, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
  cv::imwrite(out_match_file, out, {CV_IMWRITE_JPEG_QUALITY,80});

}


Lfnet::Lfnet(std::shared_ptr<YAML::Node> yml1, Bundler *bundler) : SiftManager(yml1, bundler), _context(1), _socket(_context, ZMQ_REQ)
{
  const std::string port = (*yml)["port"].as<std::string>();
  _socket.connect("tcp://0.0.0.0:"+port);
  printf("Connected to port %s\n", port.c_str());
}

Lfnet::~Lfnet()
{

}

void Lfnet::detectFeature(std::shared_ptr<Frame> frame, const float rot_deg)
{
  const auto &roi = frame->_roi;
  const int W = roi(1)-roi(0);
  const int H = roi(3)-roi(2);

  Eigen::Matrix3f forward_transform(Eigen::Matrix3f::Identity());
  Eigen::Matrix3f new_transform(Eigen::Matrix3f::Identity());

  int side = std::max(H,W);
  cv::Mat img = cv::Mat::zeros(side,side,CV_8UC3);
  for (int h=0;h<H;h++)
  {
    for (int w=0;w<W;w++)
    {
      img.at<cv::Vec3b>(h,w) = frame->_color.at<cv::Vec3b>(h+roi(2), w+roi(0));
    }
  }
  new_transform.setIdentity();
  new_transform(0,2) = -roi(0);
  new_transform(1,2) = -roi(2);
  forward_transform = new_transform * forward_transform;

  if (rot_deg!=0)
  {
    cv::Mat M = cv::getRotationMatrix2D(cv::Point2f(side/2, side/2), rot_deg, 1);
    cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), img.size(), rot_deg).boundingRect2f();
    M.at<double>(0,2) += bbox.width/2.0 - img.cols/2.0;
    M.at<double>(1,2) += bbox.height/2.0 - img.rows/2.0;
    side = std::max(bbox.width, bbox.height);
    cv::warpAffine(img, img, M, {side,side});
    Eigen::Matrix<float,2,3> tmp;
    cv::cv2eigen(M, tmp);
    new_transform.setIdentity();
    new_transform.block(0,0,2,3) = tmp;
    forward_transform = new_transform * forward_transform;

  }


  const int H_input = 400;
  const int W_input = 400;

  cv::resize(img, img, {W_input, H_input});
  new_transform.setIdentity();
  new_transform(0,0) = W_input/float(side);
  new_transform(1,1) = H_input/float(side);
  forward_transform = new_transform * forward_transform;


  {
    zmq::message_t msg(2*sizeof(int));
    std::vector<int> wh = {img.cols, img.rows};
    std::memcpy(msg.data(), wh.data(), 2*sizeof(int));
    _socket.send(msg, ZMQ_SNDMORE);
  }

  {
    cv::Mat flat = img.reshape(1, img.total()*img.channels());
    std::vector<unsigned char> vec = img.isContinuous()? flat : flat.clone();
    zmq::message_t msg(vec.size()*sizeof(unsigned char));
    std::memcpy(msg.data(), vec.data(), vec.size()*sizeof(unsigned char));
    _socket.send(msg, 0);
  }

  printf("zmq start waiting for reply\n");
  std::vector<zmq::message_t> recv_msgs;
  zmq::recv_multipart(_socket, std::back_inserter(recv_msgs));
  printf("zmq got reply\n");

  std::vector<int> info(2);
  std::memcpy(info.data(), recv_msgs[0].data(), info.size()*sizeof(int));
  const int num_feat = info[0];
  const int feat_dim = info[1];

  std::vector<float> kpts_array(num_feat*2);
  std::memcpy(kpts_array.data(), recv_msgs[1].data(), kpts_array.size()*sizeof(float));

  std::vector<float> feat_array(num_feat*feat_dim);
  std::memcpy(feat_array.data(), recv_msgs[2].data(), feat_array.size()*sizeof(float));

  frame->_keypts.resize(num_feat);
  frame->_feat_des = cv::Mat::zeros(num_feat, feat_dim, CV_32F);
  Eigen::Matrix3f backward_transform = forward_transform.inverse();
  for (int i=0;i<num_feat;i++)
  {
    Eigen::Vector3f p(kpts_array[2*i], kpts_array[2*i+1], 1);
    p = backward_transform * p;
    cv::KeyPoint kpt({p(0), p(1)}, 1);
    frame->_keypts[i] = kpt;

    for (int j=0;j<feat_dim;j++)
    {
      frame->_feat_des.at<float>(i,j) = feat_array[i*feat_dim+j];
    }
  }
  frame->_feat_des_gpu.upload(frame->_feat_des);
}

