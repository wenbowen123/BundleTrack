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

#include "DataLoader.h"

DataLoaderBase::DataLoaderBase(std::shared_ptr<YAML::Node> yml1)
{
  yml = yml1;
  _id = 0;
  _model_name = (*yml)["model_name"].as<std::string>();
  _real_model = boost::make_shared<pcl::PointCloud<pcl::PointXYZRGBNormal>>();

}

DataLoaderBase::~DataLoaderBase()
{

}



bool DataLoaderBase::hasNext()
{
  return _id>=_color_files.size() ? false : true;
}




DataLoaderNOCS::DataLoaderNOCS(std::shared_ptr<YAML::Node> yml1) : DataLoaderBase(yml1)
{
  const std::string data_dir = (*yml)["data_dir"].as<std::string>();
  _scene_id = std::stoi(data_dir.substr(data_dir.find("scene_") + 6, 1));

  {
    std::smatch what;
    std::regex_search(data_dir, what, std::regex("NOCS/.*"));
    _model_dir = data_dir;
    boost::replace_all(_model_dir, std::string(what[0]), "NOCS/obj_models/real_test/"+_model_name+".obj");
    _gt_dir = data_dir;
    boost::replace_all(_gt_dir, std::string(what[0]), "NOCS/gts/real_test_text/scene_" + std::to_string(_scene_id) + "/model_" + _model_name + "/");
  }

  _K<<591.0125, 0, 322.525,
      0, 590.16775, 244.11084,
      0, 0, 1;
  std::cout<<"K\n"<<_K<<"\n\n";

  {
    std::vector<std::string> gt_files;
    Utils::readDirectory(_gt_dir, gt_files);
    assert(gt_files.size()>0);
    Utils::parsePoseTxt(_gt_dir + gt_files[0], _ob_in_cam0);
    std::cout<<"ob_in_cam0\n"<<_ob_in_cam0<<"\n\n";
  };

  pcl::io::loadOBJFile(_model_dir,*_real_model);
  _mesh = boost::make_shared<pcl::PolygonMesh>();
  pcl::io::loadOBJFile(_model_dir,*_mesh);
  assert(_real_model->points.size()>0);
  Utils::downsamplePointCloud(_real_model,_real_model,0.015);

  const std::vector<std::string> synset_names = {
    "BG",
    "bottle",
    "bowl",
    "camera",
    "can",
    "laptop",
    "mug"
    };


  if (!(*yml)["use_6pack_datalist"])
  {
    std::vector<std::string> files;
    Utils::readDirectory(data_dir,files);
    printf("data has %d images\n",files.size());
    assert(files.size()>0);
    for (int i=0;i<files.size();i++)
    {
      auto f = files[i];
      if (f.find("color.png")==-1) continue;
      _color_files.push_back(data_dir+f);
    }
  }
  else
  {
    int class_id = 0;
    for (int i=1;i<synset_names.size();i++)
    {
      if (_model_name.find(synset_names[i])!=-1)
      {
        class_id = i;
        break;
      }
    }
    const std::string datalist_file = data_dir+"/../../NOCS-REAL275-additional/data_list/real_val/"+std::to_string(class_id)+"/"+_model_name+"/list.txt";
    printf("datalist_file %s\n", datalist_file.c_str());
    std::ifstream ff(datalist_file);
    std::string line;
    while (std::getline(ff, line))
    {
      boost::replace_all(line, "\n", "");
      if (line.find("scene_"+std::to_string(_scene_id))!=-1)
      {
        std::vector<std::string> tokens;
        boost::split(tokens, line, boost::is_any_of("/"));
        std::string color_file = data_dir+"/../../real_test/scene_"+std::to_string(_scene_id)+"/"+tokens.back()+"_color.png";
        _color_files.push_back(color_file);
      }
    }

  }

}

DataLoaderNOCS::~DataLoaderNOCS()
{

}

std::shared_ptr<Frame> DataLoaderNOCS::next()
{
  assert(_id<_color_files.size());
  const std::string data_dir = (*yml)["data_dir"].as<std::string>();
  const std::string model_name = (*yml)["model_name"].as<std::string>();

  std::string color_file = _color_files[_id];
  std::cout<<"color file: "<<color_file<<std::endl;
  cv::Mat color = cv::imread(color_file);
  std::string index_str = color_file.substr(color_file.find("color")-5,4);

  cv::Mat depth_raw;
  std::string depth_dir = data_dir+index_str+"_depth.png";
  Utils::readDepthImage(depth_raw, depth_dir);

  cv::Mat depth_sim = depth_raw.clone();

  cv::Mat depth;
  depth_dir = data_dir+index_str+"_depth.png";
  Utils::readDepthImage(depth, depth_dir);

  std::string mask_name = index_str+"_mask.png";
  cv::Mat mask = cv::imread(data_dir+mask_name, -1);
  int mask_id = -1;
  {
    std::string line;
    std::ifstream file(data_dir+index_str+"_meta.txt");
    if (file.is_open())
    {
      while (getline(file, line))
      {
        std::vector<std::string> words;
        boost::split(words, line, boost::is_any_of(" "));
        std::cout<<std::endl;
        int cur_mask_id = std::stoi(words[0]);
        int cur_class_id = std::stoi(words[1]);
        std::string cur_model_name = words[2];
        if (cur_model_name==model_name)
        {
          mask_id = cur_mask_id;
          break;
        }
      }
    }
    file.close();
  }

  Eigen::Vector4f roi;
  roi << 99999,0,99999,0;

  int cnt = 0;

  Eigen::Matrix4f pose(Eigen::Matrix4f::Identity());
  if (_id==0)
  {
    pose = _ob_in_cam0.inverse();
  }

  std::shared_ptr<Frame> frame(new Frame(color,depth,depth_raw, depth_sim, roi, pose, _id, index_str, _K, yml, NULL));
  _id++;

  return frame;
}


std::shared_ptr<Frame> DataLoaderNOCS::getFrameByIndex(std::string id_str)
{
  const std::string data_dir = (*yml)["data_dir"].as<std::string>();
  const std::string model_name = (*yml)["model_name"].as<std::string>();

  std::string color_file;
  int id = -1;
  for (int i=0;i<_color_files.size();i++)
  {
    if (_color_files[i].find(id_str+"color")!=-1)
    {
      color_file = _color_files[i];
      id = i;
      break;
    }
  }
  std::cout<<"color file: "<<color_file<<std::endl;
  cv::Mat color = cv::imread(color_file);
  std::string index_str = color_file.substr(color_file.find("color")-5,4);

  cv::Mat depth_raw;
  std::string depth_dir = data_dir+index_str+"_depth.png";
  Utils::readDepthImage(depth_raw, depth_dir);

  cv::Mat depth_sim;
  depth_dir = data_dir+index_str+"_depth_sim.png";
  Utils::readDepthImage(depth_sim, depth_dir);

  cv::Mat depth;
  depth_dir = data_dir+index_str+"_depth_filled.png";
  Utils::readDepthImage(depth, depth_dir);

  std::string mask_name = index_str+"_mask.png";
  cv::Mat mask = cv::imread(data_dir+mask_name, -1);
  int mask_id = -1;
  {
    std::string line;
    std::ifstream file(data_dir+index_str+"_meta.txt");
    if (file.is_open())
    {
      while (getline(file, line))
      {
        std::vector<std::string> words;
        boost::split(words, line, boost::is_any_of(" "));
        std::cout<<std::endl;
        int cur_mask_id = std::stoi(words[0]);
        int cur_class_id = std::stoi(words[1]);
        std::string cur_model_name = words[2];
        if (cur_model_name==model_name)
        {
          mask_id = cur_mask_id;
          break;
        }
      }
    }
    file.close();
  }

  Eigen::Vector4f roi;
  roi << 99999,0,99999,0;

  Eigen::Matrix4f pose(Eigen::Matrix4f::Identity());

  std::shared_ptr<Frame> frame(new Frame(color,depth,depth_raw, depth_sim, roi, pose, id, index_str, _K, yml, NULL));

  return frame;
}



DataLoaderYcbineoat::DataLoaderYcbineoat(std::shared_ptr<YAML::Node> yml1) : DataLoaderBase(yml1)
{
  const std::string data_dir = (*yml)["data_dir"].as<std::string>();
  _model_dir = (*yml)["model_dir"].as<std::string>();


  Utils::parseMatrixTxt(data_dir+"/cam_K.txt", _K);
  std::cout<<"cam K=\n"<<_K<<std::endl;

  {
    _gt_dir = data_dir+"/annotated_poses/";
    std::vector<std::string> gt_names;
    Utils::readDirectory(_gt_dir, gt_names);
    assert(gt_names.size()>0);
    Utils::parsePoseTxt(_gt_dir + gt_names[0], _ob_in_cam0);
    std::cout<<"ob_in_cam0\n"<<_ob_in_cam0<<"\n\n";

    for (int i=0;i<gt_names.size();i++)
    {
      _gt_files.push_back(_gt_dir+gt_names[i]);
    }
  }

  pcl::io::loadOBJFile(_model_dir,*_real_model);
  _mesh = boost::make_shared<pcl::PolygonMesh>();
  pcl::io::loadOBJFile(_model_dir,*_mesh);
  assert(_real_model->points.size()>0);
  Utils::downsamplePointCloud(_real_model,_real_model,0.015);

  std::vector<std::string> files;
  Utils::readDirectory(data_dir+"/rgb/",files);
  printf("data has %d images\n",files.size());
  assert(files.size()>0);

  std::vector<std::string> names;
  for (int i=0;i<files.size();i++)
  {
    auto f = files[i];
    _color_files.push_back(data_dir+"/rgb/"+f);
    std::vector<std::string> strs;
    boost::split(strs, f, boost::is_any_of("."));
    names.push_back(strs.front());
  }
  _start_digit = 0;


  pcl::PointXYZRGBNormal minPt, maxPt;
  pcl::getMinMax3D(*_real_model, minPt, maxPt);

}

DataLoaderYcbineoat::~DataLoaderYcbineoat()
{

}

std::shared_ptr<Frame> DataLoaderYcbineoat::next()
{
  assert(_id<_color_files.size());
  const std::string data_dir = (*yml)["data_dir"].as<std::string>();

  std::string color_file = _color_files[_id];
  std::cout<<"color file: "<<color_file<<std::endl;
  cv::Mat color = cv::imread(color_file);
  std::string index_str;
  {
    std::vector<std::string> strs;
    boost::split(strs, color_file, boost::is_any_of("/"));
    boost::split(strs, strs.back(), boost::is_any_of("."));
    index_str = strs[0];
  }

  cv::Mat depth_raw;
  std::string depth_dir = data_dir+"/depth/"+index_str+".png";
  Utils::readDepthImage(depth_raw, depth_dir);

  cv::Mat depth_sim;
  depth_sim = depth_raw.clone();

  cv::Mat depth;
  depth = depth_raw.clone();

  Eigen::Matrix4f pose(Eigen::Matrix4f::Identity());
  if (_id==0)
  {
    pose = _ob_in_cam0.inverse();
  }

  Eigen::Vector4f roi;
  roi << 99999,0,99999,0;

  std::shared_ptr<Frame> frame(new Frame(color,depth,depth_raw,depth_sim, roi, pose, _id, index_str.substr(_start_digit,index_str.size()-_start_digit), _K, yml, NULL, _real_model));
  _id++;

  return frame;
}


