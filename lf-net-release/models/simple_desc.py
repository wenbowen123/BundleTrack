# -*- coding: utf-8 -*-

import math
import tensorflow as tf
from common.tf_layer_utils import *
from common.tf_train_utils import get_activation_fn
from det_tools import instance_normalization


def get_model(inputs, is_training, 
              out_dim=128,
              init_num_channels=64,
              num_conv_layers=3,
              conv_ksize=3,
              activation_fn=tf.nn.relu,
              use_xavier=True, perform_bn=True,
              bn_trainable=True,
              bn_decay=None, bn_affine=True, use_bias=True,
              feat_norm='l2norm',
              reuse=False, name='SimpleDesc'):

    channels_list = [init_num_channels * 2**i for i in range(num_conv_layers)]
    print('===== {} (reuse={}) ====='.format(name, reuse))

    with tf.variable_scope(name, reuse=reuse) as net_sc:
        curr_in = inputs

        for i, num_channels in enumerate(channels_list):
            curr_in = conv2d(curr_in, num_channels,
                        kernel_size=conv_ksize, scope='conv{}'.format(i+1),
                        stride=2, padding='SAME',
                        use_xavier=use_xavier, use_bias=use_bias)
            curr_in = batch_norm_act(curr_in, activation_fn, 
                                     perform_bn=perform_bn,
                                     is_training=is_training, 
                                     bn_decay=bn_decay,
                                     bn_affine=bn_affine,
                                     bnname='bn{}'.format(i+1)
                                    )
            print('#{} conv-bn-act {}'.format(i+1, curr_in.shape))
        #----- FC
        curr_in = tf.layers.flatten(curr_in)
        print('FLAT {}'.format(curr_in.shape))
        curr_in = fully_connected(curr_in, 512, scope='fc1', 
                                  use_xavier=use_xavier, use_bias=use_bias)
        curr_in = batch_norm_act(curr_in, activation_fn, 
                                 perform_bn=perform_bn,
                                 is_training=is_training, 
                                 bn_decay=bn_decay,
                                 bn_affine=bn_affine,
                                 bnname='fc-bn1'
                                )
        raw_feats = fully_connected(curr_in, out_dim, scope='fc2',
                                use_xavier=use_xavier, use_bias=use_bias)
        if feat_norm == 'l2norm':
            norm_feats = tf.nn.l2_normalize(raw_feats, dim=1)
            print('Feat-Norm: L2-NORM')
        elif feat_norm == 'inst':
            norm_feats = instance_normalization(raw_feats)
            print('Feat-Norm: INSTANCE-NORM')
        elif feat_norm == 'rootsift':
            # need pre-L2 normalization ?
            # norm_feats = tf.nn.l2_normalize(raw_feats, dim=1)
            eps = 1e-6
            norm_feats = raw_feats
            l1norm = tf.norm(norm_feats, ord=1, axis=1, keep_dims=True)
            norm_feats = norm_feats / (l1norm + eps)
            norm_feats = tf.maximum(norm_feats, eps) # need to avoid unstability around too small values
            norm_feats = tf.sqrt(norm_feats)
        elif feat_norm == 'rootsift2':
            # need pre-L2 normalization ? because SIFT in RootSIFT has also already normalized 
            eps = 1e-6
            norm_feats = tf.nn.l2_normalize(raw_feats, dim=1)
            l1norm = tf.norm(norm_feats, ord=1, axis=1, keep_dims=True)
            norm_feats = norm_feats / (l1norm + eps)
            norm_feats = tf.maximum(norm_feats, eps) # need to avoid unstability around too small values
            norm_feats = tf.sqrt(norm_feats)
        elif feat_norm == 'non':
          norm_feats = raw_feats
          print('Feat-Norm: Nothing')
        else:
          raise ValueError('Unknown feat_norm: {}'.format(feat_norm))
        print('FEAT {}'.format(norm_feats.shape))

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, net_sc.name)

        endpoints = {}
        endpoints['raw_feats'] = raw_feats
        endpoints['norm_feats'] = norm_feats
        endpoints['var_list'] = var_list
        return norm_feats, endpoints


class Model(object):
    def __init__(self, config, is_training):
        self.config = config
        self.is_training = is_training
        self.activation_fn = get_activation_fn(config.desc_activ_fn, **{'alpha': config.desc_leaky_alpha})

    def build_model(self, feat_maps, reuse=False, name='SimpleDesc'):
        # out_dim = getattr(self.config, 'desc_dim', 128)
        feats, endpoints = get_model(feat_maps, self.is_training,
                        out_dim=self.config.desc_dim,
                        init_num_channels=self.config.desc_net_channel,
                        num_conv_layers=self.config.desc_net_depth,
                        conv_ksize=self.config.desc_conv_ksize,
                        activation_fn=self.activation_fn,
                        perform_bn=self.config.desc_perform_bn,
                        feat_norm=self.config.desc_norm,
                        reuse=reuse, name=name)
        
        return feats, endpoints 
