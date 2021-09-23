#
# Authors: Bowen Wen
# Contact: wenbowenxjtu@gmail.com
# Created in 2021

# Copyright (c) Rutgers University, 2021 All rights reserved.

# Bowen Wen and Kostas Bekris. "BundleTrack: 6D Pose Tracking for Novel Objects
#  without Instance or Category-Level 3D Models."
#  In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). 2021.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Bowen Wen, Kostas Bekris, Rutgers University,
#       nor the names of its contributors may be used to
#       endorse or promote products derived from this software without
#       specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#



import os,sys,re,yaml,cv2,argparse
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir)
import benchmark
import glob,copy
import numpy as np
import math
from Utils import *
import _pickle as cPickle
import matplotlib.pyplot as plt
import matplotlib as mpl


synset_names = ['BG',
              'bottle',
              'bowl',
              'camera',
              'can',
              'laptop',
              'mug'
              ]

np.random.seed(0)



def eval_all_ours(use_6pack_eval_code=True, verbose=True, add_noise=False, noise_pair=[0.02,0]):
  '''
  @noise_pair: range of tran/rot, in meter/deg. Default is following 6-pack
  '''
  if not use_6pack_eval_code:
    class_ids = np.arange(1,7)
    scene_ids = np.arange(1,7)
  else:     #########!NOTE  use 6pack eval code, convert to their format first
    scene_ids = np.arange(1,7)
    out_dir = '/tmp/nocs_res_ours'
    os.system('rm -rf '+out_dir)
    data_list_files = sorted(glob.glob(f'{nocs_dir}/NOCS-REAL275-additional/data_list/real_val/*/*/list.txt'))
    for data_list_file in data_list_files:
      with open(data_list_file,'r') as ff:
        lines = ff.readlines()
        last_scene_id = None
        init_pose = None
        for i in range(len(lines)):
          line = lines[i].rstrip().split('/')
          scene_id = int(line[1][-1])
          frame_id = int(line[2])
          ob_name = data_list_file.split('/')[-2]
          class_id = None
          for id,class_name in enumerate(synset_names):
            if class_name in ob_name:
              class_id = id
              break
          pred_file = f'{results_dir}/{ob_name}_{scene_id}/poses/{frame_id:04d}.txt'
          # pred_file = f'{results_dir}/model_free_tracking_model_{ob_name}_scene_{scene_id}/poses/{frame_id:04d}.txt'

          try:
            pred_pose = np.loadtxt(pred_file)
          except Exception as e:
            print('ERROR',e)
            pred_pose = np.eye(4)
          if last_scene_id is None or last_scene_id!=scene_id:   #### init pose
            init_pose = pred_pose.copy()
            if add_noise:
              print("Adding noise (noise_pair={}) to init pose {}".format(noise_pair,line))
              trans_noise = noise_pair[0]
              rot_noise = noise_pair[1]/180.0 * np.pi
              pred_pose[:3,3] += np.array([np.random.uniform(-trans_noise, trans_noise) for _ in range(3)])
              mag_R = np.random.uniform(-rot_noise, rot_noise)
              direction_R = random_direction()
              noise_R = cv2.Rodrigues(direction_R*mag_R)[0].reshape(3,3)
              pred_pose[:3,:3] = pred_pose[:3,:3]@noise_R
              init_pose_new = pred_pose.copy()
          else:
            if add_noise:
              cam_in_firstcam = init_pose @ np.linalg.inv(pred_pose)
              pred_new = np.linalg.inv(cam_in_firstcam) @ init_pose_new
              pred_pose = pred_new.copy()

          out_file = '{}/TEMP_50/temp{}/{}_scene_{}_{}_pose.txt'.format(out_dir,class_id,ob_name,scene_id,i)
          os.makedirs(os.path.dirname(out_file),exist_ok=True)
          with open(out_file,'w') as f1:
            z_180_RT = np.zeros((4, 4), dtype=np.float32)
            z_180_RT[:3, :3] = np.diag([-1, -1, 1])
            z_180_RT[3, 3] = 1
            pred_pose = z_180_RT @ pred_pose
            pred_pose[:3,3] *= 1000
            for row in range(3):
              f1.write('{} {} {}\n'.format(pred_pose[row,0],pred_pose[row,1],pred_pose[row,2]))
            f1.write('{} {} {}\n'.format(pred_pose[0,3],pred_pose[1,3],pred_pose[2,3]))
          last_scene_id = copy.deepcopy(scene_id)

    print("Finished converting to 6pack format, now run the benchmark eval code !!")

    args.verbose = int(verbose)
    args.pred_data = f'{out_dir}/TEMP_'
    args.pred_list = '50'
    benchmark.main(args)





if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--nocs_dir', type=str, default='/media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/DATASET/NOCS')
  parser.add_argument('--results_dir', type=str, default='/tmp/BundleTrack')

  args = parser.parse_args()
  nocs_dir = args.nocs_dir
  results_dir = args.results_dir

  eval_all_ours(use_6pack_eval_code=True, verbose=True, add_noise=True, noise_pair=[0.02,0])
