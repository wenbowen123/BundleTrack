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


import numpy as np
import os,sys,glob,subprocess,yaml,shutil,time,copy,argparse
code_dir = os.path.dirname(os.path.realpath(__file__))
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat


synset_names = ['BG',
            'bottle',
            'bowl',
            'camera',
            'can',
            'laptop',
            'mug'
            ]



def run_one_scene_one_model(scene_id,model_name,cfg1,port):
  cfg = copy.deepcopy(cfg1)
  cur_out_dir = f'/tmp/BundleTrack/nocs/{model_name}_{scene_id}'

  data_dir = f'{args.nocs_dir}/real_test/scene_{scene_id}/'
  cfg['data_dir'] = data_dir
  code_dir = os.path.dirname(os.path.realpath(__file__))
  cfg['mask_dir'] = f'{code_dir}/../masks/transductive_vos.nocs.maskrcnn_realtrain/scene_{scene_id}_model_{model_name}'
  cfg['model_name'] = model_name
  cfg['debug_dir'] = cur_out_dir
  cfg['LOG'] = 0
  cfg['port'] = port
  tmp_config_dir = '/tmp/config_model_{}_scene{}.yml'.format(model_name,scene_id)
  with open(tmp_config_dir,'w') as ff:
    yaml.dump(cfg,ff)

  code_dir = os.path.dirname(os.path.realpath(__file__))
  cmd = f'{code_dir}/../build/bundle_track_nocs {tmp_config_dir}'
  print(cmd)
  try:
    subprocess.call(cmd,shell=True)
  except:
    pass



if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--nocs_dir', type=str, default='/media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/DATASET/NOCS')
  parser.add_argument('--scene_id', type=int, default=1)
  parser.add_argument('--port', type=int, default=5555)
  parser.add_argument('--model_name', type=str, default='can_arizona_tea_norm')

  args = parser.parse_args()

  code_dir = os.path.dirname(os.path.realpath(__file__))
  config_dir = f'{code_dir}/../config_nocs.yml'
  with open(config_dir,'r') as ff:
    cfg = yaml.safe_load(ff)

  run_one_scene_one_model(args.scene_id,args.model_name,cfg,args.port)
