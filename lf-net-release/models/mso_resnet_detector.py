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
              ori_conv_ksize=None,
              activation_fn=tf.nn.relu,
              use_xavier=True, perform_bn=True,
              bn_trainable=True,
              bn_decay=None, bn_affine=True, use_bias=True,
              min_scale=2**-3, max_scale=1, num_scales=9,
              reuse=False, name='ConvOnlyResNet'):
    num_conv = 0
    if ori_conv_ksize is None:
        ori_conv_ksize = conv_ksize # same ksize as others
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

        if num_scales == 1:
            scale_factors = [1.0]
        else:
            scale_log_factors = np.linspace(np.log(max_scale), np.log(min_scale), num_scales)
            scale_factors = np.exp(scale_log_factors)
        # print('MODEL SET SCALE MANUALLY!!!!')        # scale_log_factors = np.linspace(np.log(max_scale), np.log(min_scale), num_scales)
        # # scale_factors = np.exp(scale_log_factors)

        # max_scale = np.sqrt(2)
        # assert num_scales % 2 == 1
        # if num_scales == 1:
        #     scale_factors = [1.0]
        #     base_ind = 0
        # else:
        #     scale_log_factors = np.linspace(np.log(max_scale), -np.log(max_scale), num_scales)
        #     scale_factors = np.exp(scale_log_factors)
        #     base_ind = num_scales // 2
        print('Scales ({:2f}~{:.2f} #{}): {}'.format(min_scale, max_scale, num_scales, scale_factors))
        score_maps_list = []

        base_height_f = tf.to_float(tf.shape(curr_in)[1])
        base_width_f = tf.to_float(tf.shape(curr_in)[2])

        for i, s in enumerate(scale_factors):
            inv_s = 1.0 / s # scale are defined by extracted patch size (s of s*default_patch_size) so we need use inv-scale for resizing images
            feat_height = tf.cast(base_height_f * inv_s+0.5, tf.int32)
            feat_width = tf.cast(base_width_f * inv_s+0.5, tf.int32)
            rs_feat_maps = tf.image.resize_images(curr_in, tf.stack([feat_height, feat_width]))
            score_maps = conv2d_fixed_padding(rs_feat_maps, 1, 
                        kernel_size=conv_ksize, scope='score_conv_{}'.format(i),
                        use_xavier=use_xavier, use_bias=use_bias)
            score_maps_list.append(score_maps)

        num_conv += 1

        # orientation (initial map start from 0.0)
        ori_W_init = tf.zeros_initializer
        ori_b_init = tf.constant(np.array([1,0], dtype=np.float32)) # init with 1 for cos(q), 0 for sin(q)
        ori_maps = conv2d_custom(curr_in, 2,
                    kernel_size=ori_conv_ksize, scope='ori_conv',
                    W_initializer=ori_W_init,
                    b_initializer=ori_b_init)
        ori_maps = tf.nn.l2_normalize(ori_maps, dim=-1)

        all_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, net_sc.name)

        var_list = []
        mso_var_list = []

        for var in all_var_list:
            if 'ori_conv' in var.name:
                mso_var_list.append(var)
                if not reuse:
                    tf.summary.histogram(var.name, var)
            else:
                var_list.append(var)

        endpoints = {}
        endpoints['ori_maps'] = ori_maps
        endpoints['var_list'] = var_list
        endpoints['mso_var_list'] = mso_var_list
        endpoints['mso'] = True
        endpoints['multi_scores'] = True
        endpoints['scale_factors'] = scale_factors
        endpoints['feat_maps'] = feat_maps
        endpoints['pad_size'] = num_conv * (conv_ksize//2)
        print('PAD={}, #conv={}, ksize={} ori-ksize={}'.format(endpoints['pad_size'], num_conv, conv_ksize, ori_conv_ksize))
        return score_maps_list, endpoints

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

        min_scale = self.config.net_min_scale
        max_scale = self.config.net_max_scale
        num_scales = self.config.net_num_scales

        logits, endpoints = get_model(photos, self.is_training, 
                        num_block=self.config.net_block,
                        num_channels=self.config.net_channel,
                        conv_ksize=conv_ksize,
                        activation_fn=self.activation_fn,
                        use_xavier=use_xavier, use_bias=use_bias,
                        perform_bn=self.config.perform_bn, bn_trainable=bn_trainable, 
                        bn_decay=bn_decay, bn_affine=bn_affine, 
                        min_scale=min_scale, max_scale=max_scale, num_scales=num_scales,
                        reuse=reuse, name=name)
        
        return logits, endpoints 