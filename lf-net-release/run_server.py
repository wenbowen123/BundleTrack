#!/usr/bin/env python
from __future__ import print_function
import os
import sys
import numpy as np
import tensorflow as tf
import importlib,zmq
import time
import cv2
from tqdm import tqdm
import pickle

from mydatasets import *

from det_tools import *
from eval_tools import draw_keypoints
from common.tf_train_utils import get_optimizer
from imageio import imread, imsave
from inference import *
from utils import embed_breakpoint, print_opt


MODEL_PATH = './models'
if MODEL_PATH not in sys.path:
    sys.path.append(MODEL_PATH)


def build_networks(config, photo, is_training):

    DET = importlib.import_module(config.detector)
    detector = DET.Model(config, is_training)

    if config.input_inst_norm:
        print('Apply instance norm on input photos')
        photos1 = instance_normalization(photo)

    heatmaps, det_endpoints = build_detector_helper(config, detector, photo)

    kpts = det_endpoints['kpts']
    batch_inds = det_endpoints['batch_inds']

    kp_patches = build_patch_extraction(config, det_endpoints, photo)

    DESC = importlib.import_module(config.descriptor)
    descriptor = DESC.Model(config, is_training)
    desc_feats, desc_endpoints = descriptor.build_model(kp_patches, reuse=False)

    ops = {
        'photo': photo,
        'is_training': is_training,
        'kpts': kpts,
        'feats': desc_feats,
    }

    return ops

def build_detector_helper(config, detector, photo):
    if config.use_nms3d:
        heatmaps, det_endpoints = build_multi_scale_deep_detector_3DNMS(config, detector, photo, reuse=False)
    else:
        heatmaps, det_endpoints = build_multi_scale_deep_detector(config, detector, photo, reuse=False)
    return heatmaps, det_endpoints



if __name__ == '__main__':
  from common.argparse_utils import *
  parser = get_parser()

  general_arg = add_argument_group('General', parser)
  io_arg = add_argument_group('In/Out', parser)
  model_arg = add_argument_group('Model', parser)
  code_dir = os.path.dirname(os.path.realpath(__file__))
  model_arg.add_argument('--model', type=str, default=f'{code_dir}/release/models/indoor/',
                          help='model file or directory')
  model_arg.add_argument('--port', type=str, default='5555')
  model_arg.add_argument('--top_k', type=int, default=500,help='number of keypoints')
  model_arg.add_argument('--net_min_scale', type=float, default=None)
  model_arg.add_argument('--net_max_scale', type=float, default=None)
  model_arg.add_argument('--net_num_scales', type=int, default=None)

  tmp_config, unparsed = get_config(parser)

  if len(unparsed) > 0:
      raise ValueError('Miss finding argument: unparsed={}\n'.format(unparsed))

  if os.path.isdir(tmp_config.model):
      config_path = os.path.join(tmp_config.model, 'config.pkl')
  else:
      config_path = os.path.join(os.path.dirname(tmp_config.model), 'config.pkl')
  try:
      print("Loading from {}".format(config_path))
      with open(config_path, 'rb') as f:
          config = pickle.load(f)
          print_opt(config)
  except:
      raise ValueError('Fail to open {}'.format(config_path))

  for attr, dst_val in sorted(vars(tmp_config).items()):
      if hasattr(config, attr):
          src_val = getattr(config, attr)
          if dst_val is not None:
            if src_val != dst_val:
                setattr(config, attr, dst_val)
      else:
          setattr(config, attr, dst_val)

  tf.reset_default_graph()

  photo_ph = tf.placeholder(tf.float32, [1, None, None, 1])
  is_training = tf.constant(False)

  ops = build_networks(config, photo_ph, is_training)

  tfconfig = tf.ConfigProto()
  tfconfig.gpu_options.allow_growth = True
  sess = tf.Session(config=tfconfig)
  sess.run(tf.global_variables_initializer())

  saver = tf.train.Saver()
  print('Load trained models...')

  if os.path.isdir(config.model):
      checkpoint = tf.train.latest_checkpoint(config.model)
      model_dir = config.model
  else:
      checkpoint = config.model
      model_dir = os.path.dirname(config.model)

  if checkpoint is not None:
      print('Checkpoint', os.path.basename(checkpoint))
      saver.restore(sess, checkpoint)
  else:
      raise ValueError('Cannot load model from {}'.format(model_dir))

  context = zmq.Context()
  socket = context.socket(zmq.REP)
  port = f"tcp://*:{tmp_config.port}"
  print('port',port)
  socket.bind(port)

  while 1:
    print(f'lfnet listending to {port}')
    msgs = socket.recv_multipart(0)
    assert len(msgs)==2, '#msgs={}'.format(len(msgs))
    wh = np.frombuffer(msgs[0],dtype=np.int32)
    W = wh[0]
    H = wh[1]
    print(f'W={W}, H={H}')
    msg = msgs[1]
    photo = np.frombuffer(msg,dtype=np.uint8).reshape(H,W,-1).squeeze()
    photo_ori = photo.copy()

    rgb = photo.copy()
    if photo.ndim == 3 and photo.shape[-1] == 3:
        photo = cv2.cvtColor(photo, cv2.COLOR_RGB2GRAY)
    photo = photo[None,...,None].astype(np.float32) / 255.0
    assert photo.ndim == 4

    feed_dict = {
        photo_ph: photo,
    }
    fetch_dict = {
        'kpts': ops['kpts'],
        'feats': ops['feats'],
    }

    outs = sess.run(fetch_dict, feed_dict=feed_dict)

    num_feat = len(outs['kpts'])
    feat_dim = outs['feats'].shape[1]
    msg = np.array([num_feat,feat_dim]).reshape(-1).astype(np.int32).tobytes()
    socket.send(msg, 2)
    msg = outs['kpts'].astype(np.float32).reshape(-1).tobytes()
    socket.send(msg, 2)
    msg = outs['feats'].astype(np.float32).reshape(-1).tobytes()
    socket.send(msg, 0)
