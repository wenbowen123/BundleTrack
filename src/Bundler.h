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

#ifndef BUNDLER_HH__
#define BUNDLER_HH__


#include "Utils.h"
#include "Frame.h"
#include "FeatureManager.h"
#include "DataLoader.h"

class Frame;
class PointToPlaneLoss;
class ReprojectionIntensityLoss;
class SiftManager;
class FeatureMatchLoss;
class Lfnet;
class DataLoaderBase;
class Shape;

class Bundler
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  std::shared_ptr<Frame> _newframe;
  std::deque<std::shared_ptr<Frame>> _keyframes;
  std::deque<std::shared_ptr<Frame>> _frames;
  std::vector<std::shared_ptr<PointToPlaneLoss>> _p2p_losses;
  std::vector<std::shared_ptr<ReprojectionIntensityLoss>> _rpi_losses;
  std::vector<std::shared_ptr<FeatureMatchLoss>> _fm_losses;
  std::shared_ptr<Shape> _shape;

  std::shared_ptr<Lfnet> _fm;

  float _max_dist, _max_normal_angle;
  int _max_iter;
  bool _need_reinit;
  std::shared_ptr<YAML::Node> yml;
  DataLoaderBase* _data_loader;
  std::vector<std::shared_ptr<Frame>> _local_frames;


public:
  Bundler(std::shared_ptr<YAML::Node> yml1, DataLoaderBase *data_loader);
  void processNewFrame(std::shared_ptr<Frame> frame);
  void checkAndAddKeyframe(std::shared_ptr<Frame> frame);
  void optimizeToPrev(std::shared_ptr<Frame> frame);
  void optimizeGPU();
  void selectKeyFramesForBA();
  void saveNewframeResult();

};

#endif