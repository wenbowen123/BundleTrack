# -*- coding: utf-8 -*-

# Detector with Multi Scale and Orientation (MSO)

import math
import tensorflow as tf
from common.tf_layer_utils import *
from common.tf_train_utils import get_activation_fn

def building_block(inputs, out_channels, 
                 projection_shortcut, 
                 stride,
                 scope, 
                 conv_ksize=3,
                 use_xavier=True, 
                 activation_fn=tf.nn.relu,
                 perform_bn=False, 
                 bn_decay=None,
                 bn_affine=True,
                 is_training=None,
                 use_bias=True,
                ):
    
    with tf.variable_scope(scope):
        curr_in = inputs
        shortcut = curr_in # activate_before_residual=False
        curr_in = batch_norm_act(curr_in, activation_fn=activation_fn,
                                 perform_bn=perform_bn, 
                                 is_training=is_training,
                                 bn_decay=bn_decay,
                                 bn_affine=bn_affine, 
                                 bnname='pre-bn'
                                )
        # The projection shortcut should come after the first batch norm and ReLU
        # since it performs a 1x1 convolution.
        if projection_shortcut is not None:
            shortcut = projection_shortcut(curr_in) # discard previous shortcut

        # conv-bn-act
        curr_in = conv2d_fixed_padding(curr_in, out_channels, 
                         kernel_size=conv_ksize, 
                         scope='conv1',
                         stride=stride,
                         use_xavier=use_xavier,
                         use_bias=use_bias,
                        )
        curr_in = batch_norm_act(curr_in, activation_fn=activation_fn,
                                 perform_bn=perform_bn, 
                                 is_training=is_training,
                                 bn_decay=bn_decay,
                                 bn_affine=bn_affine, 
                                 bnname='mid-bn'
                                )
        # conv only
        curr_in = conv2d_fixed_padding(curr_in, out_channels, 
                         kernel_size=conv_ksize, 
                         scope='conv2',
                         stride=1,
                         use_xavier=use_xavier,
                         use_bias=use_bias,
                        )
        return curr_in + shortcut

def get_model(inputs, is_training, 
              num_block=3,
              num_channels=16,
              conv_ksize=3,
              activation_fn=tf.nn.relu,
              use_xavier=True, perform_bn=True,
              bn_trainable=True,
              bn_decay=None, bn_affine=True, use_bias=True,
              reuse=False, name='ConvOnlyResNet'):
    num_conv = 0

    with tf.variable_scope(name, reuse=reuse) as net_sc:
        curr_in = tf.identity(inputs)

        # init-conv
        curr_in = conv2d_fixed_padding(curr_in, num_channels,
                        kernel_size=conv_ksize, scope='init_conv',
                        use_xavier=use_xavier, use_bias=use_bias)
        num_conv += 1
        for i in range(num_block):
            curr_in = building_block(curr_in, num_channels, None,
                        stride=1, scope='block-{}'.format(i+1),
                        conv_ksize=conv_ksize,
                        use_xavier=use_xavier,
                        activation_fn=activation_fn,
                        perform_bn=perform_bn,
                        bn_decay=bn_decay, bn_affine=bn_affine,
                        is_training=is_training, use_bias=use_bias
                        )
            num_conv += 2
        curr_in = batch_norm_act(curr_in, activation_fn=activation_fn,
                                 perform_bn=perform_bn, 
                                 is_training=is_training,
                                 bn_decay=bn_decay,
                                 bn_affine=bn_affine, 
                                 bnname='fin-bn')

        feat_maps = tf.identity(curr_in)

        score_maps = conv2d_fixed_padding(feat_maps, 1, 
                    kernel_size=conv_ksize, scope='score_conv',
                    use_xavier=use_xavier, use_bias=use_bias)

        batch_size, height, width = tf.unstack(tf.shape(feat_maps))[:3]
        init_cos = tf.ones([batch_size, height, width, 1])
        init_sin = tf.zeros([batch_size, height, width, 1])
        ori_maps = tf.concat([init_cos, init_sin], axis=-1)

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, net_sc.name)
        mso_var_list = []

        endpoints = {}
        endpoints['ori_maps'] = ori_maps
        endpoints['var_list'] = var_list
        endpoints['mso_var_list'] = mso_var_list
        endpoints['mso'] = False
        endpoints['multi_scores'] = False
        endpoints['feat_maps'] = feat_maps
        endpoints['pad_size'] = num_conv * (conv_ksize//2)
        print('PAD={}, #conv={}, ksize={}'.format(endpoints['pad_size'], num_conv, conv_ksize))
        return score_maps, endpoints

class Model(object):
    def __init__(self, config, is_training):
        self.config = config
        self.is_training = is_training
        self.activation_fn = get_activation_fn(config.activ_fn, **{'alpha': config.leaky_alpha})

    def build_model(self, photos, reuse=False, name='ConvOnlyResNet'):
        conv_ksize = getattr(self.config, 'conv_ksize', 3)
        use_xavier = getattr(self.config, 'use_xavier', True)
        use_bias = getattr(self.config, 'use_bias', True)
        bn_trainable = getattr(self.config, 'bn_trainable', True)
        bn_decay = getattr(self.config, 'bn_decay', None)
        bn_affine = getattr(self.config, 'bn_affine', True)

        logits, endpoints = get_model(photos, self.is_training, 
                        num_block=self.config.net_block,
                        num_channels=self.config.net_channel,
                        conv_ksize=conv_ksize,
                        activation_fn=self.activation_fn,
                        use_xavier=use_xavier, use_bias=use_bias,
                        perform_bn=self.config.perform_bn, bn_trainable=bn_trainable, 
                        bn_decay=bn_decay, bn_affine=bn_affine, 
                        reuse=reuse, name=name)
        
        return logits, endpoints 
