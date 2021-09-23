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

#include "LossGPU.h"
#include <cuda.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

OptimizerGpu::OptimizerGpu(std::shared_ptr<YAML::Node> yml1)
{
  yml = yml1;
}


OptimizerGpu::~OptimizerGpu()
{

}



void OptimizerGpu::optimizeFrames(const std::vector<EntryJ> &global_corres, const std::vector<int> &n_match_per_pair, int n_frames, int H, int W, const std::vector<float*> &depths_gpu, const std::vector<uchar4*> &colors_gpu, const std::vector<float4*> &normals_gpu, std::vector<Eigen::Matrix4f,Eigen::aligned_allocator<Eigen::Matrix4f> > &poses, const Eigen::Matrix3f &K)
{
  const float image_downscale = (*yml)["bundle"]["image_downscale"].as<float>();
  const int W_down = W/image_downscale;
  const int H_down = H/image_downscale;
  Eigen::Matrix4f tmp(Eigen::Matrix4f::Identity());
  tmp.block(0,0,3,3) = K;
  mat4f inputIntrinsics;
  for (int h=0;h<4;h++)
  {
    for (int w=0;w<4;w++)
    {
      inputIntrinsics(h,w) = tmp(h,w);
    }
  }

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

  CUDACache cuda_cache(W, H, W_down, H_down, n_frames, inputIntrinsics);
  for (int i=0;i<n_frames;i++)
  {
    cuda_cache.storeFrame(W, H, depths_gpu[i], colors_gpu[i], normals_gpu[i]);
  }

#ifdef _DEBUG
  cutilSafeCall(cudaDeviceSynchronize());
  cutilCheckMsg(__FUNCTION__);
#endif

  float4x4* d_transforms;
  cudaMalloc(&d_transforms, sizeof(float4x4)*n_frames);
  std::vector<float4x4> transforms_cpu(n_frames);
  for (int i=0;i<n_frames;i++)
  {
    for (int h=0;h<4;h++)
    {
      for (int w=0;w<4;w++)
      {
        transforms_cpu[i].entries2[h][w] = poses[i](h,w);
      }
    }
  }
  cudaMemcpy(d_transforms, transforms_cpu.data(), sizeof(float4x4)*n_frames, cudaMemcpyHostToDevice);

  std::cout<<"global_corres="<<global_corres.size()<<std::endl;

  const uint max_n_residuals = n_frames*(n_frames-1)/2*H_down*W_down/4 + global_corres.size();

  std::map<int,int> n_corres_per_frame;
  for (int i=0;i<global_corres.size();i++)
  {
    const auto &corr = global_corres[i];
    n_corres_per_frame[corr.imgIdx_i]++;
    n_corres_per_frame[corr.imgIdx_j]++;
  }
  int max_corr_per_image = 0;
  for (const auto &h:n_corres_per_frame)
  {
    max_corr_per_image = std::max(max_corr_per_image, h.second);
  }
  SBA _sba(n_frames, max_n_residuals, max_corr_per_image, yml);
  bool removed = _sba.align(global_corres, n_match_per_pair, n_frames, &cuda_cache, d_transforms, false, true, false, true, false, false, -1);

  transforms_cpu.clear();
  cudaMemcpy(transforms_cpu.data(), d_transforms, sizeof(float4x4)*n_frames, cudaMemcpyDeviceToHost);
  for (int i=0;i<n_frames;i++)
  {
    for (int h=0;h<4;h++)
    {
      for (int w=0;w<4;w++)
      {
        poses[i](h,w) = transforms_cpu[i].entries2[h][w];
      }
    }
  }

  cutilSafeCall(cudaFree(d_transforms));

#ifdef _DEBUG
	cutilSafeCall(cudaDeviceSynchronize());
	cutilCheckMsg(__FUNCTION__);
#endif

}


