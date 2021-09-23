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



import os,sys
import open3d as o3d
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(code_dir+'/../')
import numpy as np
import glob
import scipy.io
import numpy as np
import scipy.io
import Utils as U
import argparse
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat
import matplotlib.pyplot as plt

def VOCap(rec):
  rec = np.sort(np.array(rec))
  n = len(rec)
  if n==0:
    return 0
  prec = np.arange(1,n+1) / float(n)
  rec = rec.reshape(-1)
  prec = prec.reshape(-1)
  index = np.where(rec<0.1)[0]
  if len(index)==0:
    return 0
  rec = list(rec[index])
  prec = list(prec[index])

  rec.sort()
  prec.sort()

  mrec=[0, *list(rec), 0.1]
  mpre=[0, *list(prec), prec[-1]]

  for i in range(1,len(mpre)):
    mpre[i] = max(mpre[i], mpre[i-1])
  mpre = np.array(mpre)
  mrec = np.array(mrec)
  i = np.where(mrec[1:]!=mrec[0:len(mrec)-1])[0] + 1
  ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
  return ap




def eval_all(ycbineoat_dir,ycb_model_dir,results_dir):
  videoname_to_objects = {
      'bleach0': '021_bleach_cleanser',
      'bleach_hard_00_03_chaitanya': '021_bleach_cleanser',
      'cracker_box_reorient': '003_cracker_box',
      'cracker_box_yalehand0': '003_cracker_box',
      'mustard0': '006_mustard_bottle',
      'mustard_easy_00_02': '006_mustard_bottle',
      'sugar_box1': '004_sugar_box',
      'sugar_box_yalehand0': '004_sugar_box',
      'tomato_soup_can_yalehand0': '005_tomato_soup_can',
  }
  models = {}
  tmp = glob.glob(f'{ycb_model_dir}/*/points.xyz')
  for k,v in videoname_to_objects.items():
    points_dir = f'{ycb_model_dir}/{v}/points.xyz'
    with open(points_dir,'r') as ff:
      lines = ff.readlines()
    model_pts = []
    for i in range(len(lines)):
      line = list(map(float,lines[i].rstrip().split()))
      model_pts.append(line)
    model_pts = np.array(model_pts)
    model_pts.reshape(-1,3)
    model = U.toOpen3dCloud(model_pts,colors=np.zeros(model_pts.shape,dtype=np.float64))
    models[k] = model

  class_res = {}
  for videoname,obj_name in videoname_to_objects.items():
    class_res[obj_name] = {'add':[],'add-s':[]}

  folders = sorted(glob.glob(f'{ycbineoat_dir}/*'))

  for folder in folders:
    if '.tar.gz' in folder:
      continue
    videoname = folder.split('/')[-1]
    rgb_files = sorted(glob.glob(folder+'/rgb/*.png'))
    # pred_files = sorted(glob.glob(f'{results_dir}/{videoname}/poses/*.txt'))
    pred_files = sorted(glob.glob(f'{results_dir}/model_free_tracking_model_{videoname}/poses/*.txt'))

    gt_files = sorted(glob.glob(folder+"/annotated_poses/*.txt"))
    print('{}: #rgb_files={}, #pred_files={}, #gt_files={}'.format(videoname, len(rgb_files),len(pred_files), len(gt_files)))
    if len(rgb_files)!=len(pred_files) or len(pred_files)!=len(gt_files):
      raise RuntimeError("ERROR files not match", folder)

    obj_name = videoname_to_objects[videoname]
    for i in range(len(rgb_files)):
      try:
        pred = np.loadtxt(pred_files[i])
      except:
        print('ERROR pred file not found {}'.format(rgb_files[i]))
        pred = np.eye(4)
      try:
        gt = np.loadtxt(gt_files[i])
      except:
        print('ERROR gt not found {}'.format(gt_files[i]))
        gt = np.eye(4)
      add = U.add(pred,gt,models[videoname])
      adi = U.adi(pred,gt,models[videoname])
      class_res[obj_name]['add'].append(add)
      class_res[obj_name]['add-s'].append(adi)

  adds = []
  adis = []
  print("----------------------------------")
  for k in class_res.keys():
    adi = class_res[k]['add-s']
    adis += adi
    adi_auc = VOCap(adi) * 100
    add = class_res[k]['add']
    adds += add
    add_auc = VOCap(add) * 100
    print('{} #pred={}: adi={} add={}'.format(k,len(adi),adi_auc,add_auc))

  adi_auc = VOCap(adis) * 100
  add_auc = VOCap(adds) * 100
  print('Total pose:',len(adis))
  print('\nOverall, adi={} add={}'.format(adi_auc,add_auc))



if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--ycbineoat_dir', type=str, default='/media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/catkin_ws/src/iros20_dataset/video_rosbag/IROS_SELECTED/FINISHED_LABEL.iros_submission_version')
  parser.add_argument('--ycb_model_dir', type=str, default='/media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/DATASET/YCB_Video_Dataset/CADmodels')
  parser.add_argument('--results_dir', type=str, default='/tmp/BundleTrack/ycbineoat')
  args = parser.parse_args()

  ycbineoat_dir = args.ycbineoat_dir
  ycb_model_dir = args.ycb_model_dir
  results_dir = args.results_dir
  eval_all(ycbineoat_dir,ycb_model_dir,results_dir)






