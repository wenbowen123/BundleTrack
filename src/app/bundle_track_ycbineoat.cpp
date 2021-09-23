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

#include "Utils.h"
#include <bits/stdc++.h>
#include <boost/algorithm/string.hpp>
#include "Bundler.h"
#include "DataLoader.h"


int main(int argc, char **argv)
{
  std::shared_ptr<YAML::Node> yml(new YAML::Node);
  if (argc<2)
  {
    printf("Please provide path to config file\n");
    exit(1);
  }

  std::string config_dir = std::string(argv[1]);
  *yml = YAML::LoadFile(config_dir);

  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
  DataLoaderYcbineoat data_loader(yml);

  const std::string base_dir = (*yml)["debug_dir"].as<std::string>();
  std::string cmd = "rm -rf "+base_dir+" && mkdir -p "+base_dir+" && mkdir -p "+base_dir+"/color_viz/";
  system(cmd.c_str());


  Eigen::Matrix4f ob_in_cam_last(Eigen::Matrix4f::Identity());

  Bundler bundler(yml,&data_loader);

  while (data_loader.hasNext())
  {
    std::shared_ptr<Frame> frame = data_loader.next();
    if (!frame) break;
    const std::string index_str = frame->_id_str;
    const std::string out_dir = (*yml)["debug_dir"].as<std::string>()+"/"+index_str+"/";
    cv::imwrite(out_dir+index_str+"_color.png",frame->_color);

    Eigen::Matrix4f cur_in_model(data_loader._ob_in_cam0.inverse());
    bundler.processNewFrame(frame);

    bundler.saveNewframeResult();

  }
}