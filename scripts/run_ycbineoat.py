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
try:
  multiprocessing.set_start_method('spawn')
except:
  pass


def run_one_video(data_dir,model_name,model_dir,custom_yc,cfg1,port):
  cfg = copy.deepcopy(cfg1)
  name = data_dir.split('/')[-1]

  cur_out_dir = '/tmp/BundleTrack/ycbineoat/{}/'.format(name)
  os.system(f'mkdir -p {cur_out_dir}')

  cfg['data_dir'] = data_dir
  cfg['mask_dir'] = f'/home/shreesh/DATASET/YCBInEOAT/{name}/masks'
  cfg['model_name'] = model_name
  cfg['model_dir'] = model_dir
  cfg['debug_dir'] = cur_out_dir
  cfg['custom_yc'] = custom_yc
  cfg['LOG'] = 0
  cfg['port'] = port
  tmp_config_dir = '/tmp/config_{}.yml'.format(name)
  with open(tmp_config_dir,'w') as ff:
    yaml.dump(cfg,ff)

  cmd = f'{code_dir}/../build/bundle_track_ycbineoat {tmp_config_dir}'
  print(cmd)
  try:
    subprocess.call(cmd,shell=True)
  except:
    pass




if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/home/shreesh/DATASET/YCBInEOAT/cracker_box_icg_1')
  parser.add_argument('--port', type=int, default=5555)
  parser.add_argument('--model_name', type=str, default='003_cracker_box')
  parser.add_argument('--model_dir', type=str, default='/home/shreesh/DATASET/YCBInEOAT/ycb/003_cracker_box/google_512k/textured.obj')
  parser.add_argument('--custom_yc', type=str, default='false')


  args = parser.parse_args()

  data_dir = args.data_dir
  if not os.path.exists(data_dir):
    raise RuntimeError(f"Make sure data_dir={data_dir} exists")

  code_dir = os.path.dirname(os.path.realpath(__file__))
  config_dir = '/home/shreesh/BundleTrack/config_ycbineoat.yml'
  with open(config_dir,'r') as ff:
    cfg = yaml.safe_load(ff)

  run_one_video(args.data_dir,args.model_name,args.model_dir,args.custom_yc,cfg,args.port)
