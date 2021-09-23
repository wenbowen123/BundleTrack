# -*- coding: utf-8 -*-

import numpy as np
import os
import random
import sys

import tensorflow as tf
from .dataset_tools import *

class ColorRawSfMDataset(object):
    def __init__(self, longer_edge=640, output_filenames=False, num_threads=8):
        self.num_threads = num_threads
        self.longer_edge = longer_edge
        self.depth_factor = 0.001
        self.far_depth_val = 1000
        self.depth_thresh = 10.0
        self.compression_type = None
        self.output_filenames = output_filenames
    def get_dataset(self, root_dir, imroot_dir, render_paths, phase, shuffle=True, num_epoch=None, seed=None, max_examples=-1):

        self.random_transformer = RandomTransformer('', 'none', max_scale=1, min_scale=1, max_degree=0)

        if isinstance(render_paths, str):
            render_paths = [render_paths]
        num_seq = len(render_paths)
        
        if not root_dir.endswith('/'):
            root_dir += '/'
        self.root_dir = tf.convert_to_tensor(root_dir)

        if not imroot_dir.endswith('/'):
            imroot_dir += '/'
        self.imroot_dir = tf.convert_to_tensor(imroot_dir)
        
        pose_tfrecords = []
        total_num_photos = 0
        for render in render_paths:
            if phase == 'train':
                if max_examples > 0:
                    pose_fname = 'train_{}.tfrecord'.format(max_examples)
                    size_fname = None
                    print('Found {} (limited sample={})'.format(pose_fname, max_examples))
                else:
                    pose_fname = 'train.tfrecord'
                    size_fname = 'train_size.txt'
            elif phase == 'valid':
                pose_fname = 'valid.tfrecord'
                size_fname = 'valid_size.txt'
            else:
                if max_examples > 0 and os.path.exists(os.path.join(root_dir, render, 'pose_{}.tfrecord'.format(max_examples))):
                    pose_fname = 'pose_{}.tfrecord'.format(max_examples)
                    size_fname = None
                    print('Found {} (limited sample={})'.format(pose_fname, max_examples))
                else:
                    pose_fname = 'pose.tfrecord'
                    size_fname = 'size.txt'

            pose_tfrecords.append(os.path.join(root_dir, render, pose_fname))
            if size_fname is None:
                size = max_examples
            else:
                with open(os.path.join(root_dir, render, size_fname)) as f:
                    size = int(f.readline())
                print('{} has {} examples'.format(render, size))
                if max_examples > 0:
                    size = min(size, max_examples)
                    print('---> actual size={}'.format(size))

            total_num_photos += size

        self.total_num_photos = total_num_photos
        self.num_photos_per_seq_data = np.array([total_num_photos], dtype=np.int32)

        dataset = tf.data.TFRecordDataset(pose_tfrecords, compression_type=self.compression_type)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.total_num_photos, seed=seed)
        dataset = dataset.repeat(count=num_epoch)
        dataset = dataset.map(self.parser, num_parallel_calls=self.num_threads)
        dataset = dataset.batch(1)
        # dataset = dataset.prefetch(buffer_size=2)
        return dataset

    def parser(self, serialized):
        with tf.name_scope('parse_example'):
            example = tf.parse_single_example(serialized, features={
                'rgb1_filename': tf.FixedLenFeature([], tf.string),
                'rgb2_filename': tf.FixedLenFeature([], tf.string),
                'depth1_filename': tf.FixedLenFeature([], tf.string),
                'depth2_filename': tf.FixedLenFeature([], tf.string),
                'shape1': tf.FixedLenFeature([2], tf.int64),
                'shape2': tf.FixedLenFeature([2], tf.int64),
                'bbox1': tf.FixedLenFeature([4], tf.int64), # x1,x2,y1,y2
                'bbox2': tf.FixedLenFeature([4], tf.int64),
                'c1Tw': tf.FixedLenFeature([16], tf.float32),
                'c2Tw': tf.FixedLenFeature([16], tf.float32),
                'K1': tf.FixedLenFeature([9], tf.float32),
                'K2': tf.FixedLenFeature([9], tf.float32),
            })

        shape1 = example['shape1']
        shape2 = example['shape2']
        c1Tw = tf.reshape(example['c1Tw'], [4,4])
        c2Tw = tf.reshape(example['c2Tw'], [4,4])
        K1 = tf.reshape(example['K1'], [3,3])
        K2 = tf.reshape(example['K2'], [3,3])

        bb1 = example['bbox1']
        bb2 = example['bbox2']

        rgb1_filename = self.imroot_dir + example['rgb1_filename']
        rgb2_filename = self.imroot_dir + example['rgb2_filename']
        depth1_filename = self.root_dir + example['depth1_filename']
        depth2_filename = self.root_dir + example['depth2_filename']
        
        # return rgb1_filename, rgb2_filename, depth1_filename, depth2_filename

        rgb1, gray1 = self._decode_rgb(rgb1_filename, shape1)
        rgb2, gray2 = self._decode_rgb(rgb2_filename, shape2)
        depth1, valid_mask1 = self._decode_depth(depth1_filename, shape1)
        depth2, valid_mask2 = self._decode_depth(depth2_filename, shape2)

        dv1 = tf.concat([depth1, valid_mask1], axis=-1)
        dv2 = tf.concat([depth2, valid_mask2], axis=-1)

        if self.longer_edge is not None:
            gray1, sy1, sx1 = self._resize_by_longer_edge(gray1, self.longer_edge)
            rgb1, _, _ = self._resize_by_longer_edge(rgb1, self.longer_edge)
            dv1, _, _ = self._resize_by_longer_edge(dv1, self.longer_edge, inter_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            gray2, sy2, sx2 = self._resize_by_longer_edge(gray2, self.longer_edge)
            rgb2, _, _ = self._resize_by_longer_edge(rgb2, self.longer_edge)
            dv2, _, _ = self._resize_by_longer_edge(dv2, self.longer_edge, inter_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # modify intrinsic matrix
            S1 = make_scale_theta(sx1, sy1)
            S2 = make_scale_theta(sx2, sy2)
            K1 = tf.matmul(S1, K1)
            K2 = tf.matmul(S2, K2)

        depth1 = tf.slice(dv1, [0,0,0], [-1,-1,1])        
        valid_mask1 = tf.slice(dv1, [0,0,1], [-1,-1,1])        
        depth2 = tf.slice(dv2, [0,0,0], [-1,-1,1])        
        valid_mask2 = tf.slice(dv2, [0,0,1], [-1,-1,1])        
        
        # Pose
        c2Tc1, c1Tc2 = get_delta_pose(c1Tw, c2Tw)
        
        # get random thetas (doesnot support table-random)
        theta_params, use_augs = self.random_transformer.get_theta_params(index=None)

        if self.output_filenames:
            print('[RawSfMDataset] return filenames (mainly for lift evaluation)')
            return gray1, gray2, depth1, depth2, valid_mask1, valid_mask2, c2Tc1, c1Tc2, c1Tw, c2Tw, K1, K2, theta_params, use_augs, example['rgb1_filename'], example['rgb2_filename'], rgb1, rgb2
        else:
            return gray1, gray2, depth1, depth2, valid_mask1, valid_mask2, c2Tc1, c1Tc2, c1Tw, c2Tw, K1, K2, theta_params, use_augs, rgb1, rgb2
        
    def _decode_rgb(self, filename, shape):
        img = tf.read_file(filename)
        rgb = tf.image.decode_jpeg(img, 3)
        gray = tf.image.decode_jpeg(img, 1)
        rgb = tf.cast(rgb, tf.float32) / 255.0
        gray = tf.cast(gray, tf.float32) / 255.0
        return rgb, gray
    
    def _decode_depth(self, filename, shape):
        depth = tf.read_file(filename)
        depth = tf.image.decode_png(depth, 1, dtype=tf.uint16) # force to load as grayscale
        depth = tf.scalar_mul(self.depth_factor, tf.cast(depth, tf.float32))
        is_zero = tf.equal(depth, tf.constant(0, dtype=tf.float32))
        valid_mask = tf.cast(tf.logical_not(is_zero), tf.float32)
        far_depth = tf.scalar_mul(self.far_depth_val, tf.ones_like(depth, dtype=tf.float32))
        depth = tf.where(is_zero, far_depth, depth)
        return depth, valid_mask

    def _resize_by_longer_edge(self, image, longer_edge, inter_method=tf.image.ResizeMethod.BILINEAR):

        def _compute_shoter_edge(prev_shorter_edge, prev_longer_edge, new_longer_edge):
            return tf.cast( prev_shorter_edge * new_longer_edge / prev_longer_edge, tf.int32)

        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        height_smaller_than_width = tf.less_equal(height, width)

        new_height_and_width = tf.cond(
            height_smaller_than_width,
            lambda: (_compute_shoter_edge(height, width, longer_edge), longer_edge),
            lambda: (longer_edge, _compute_shoter_edge(width, height, longer_edge))
        )

        new_image = tf.image.resize_images(image, new_height_and_width, method=inter_method)

        # scale ration
        sy = tf.to_float(new_height_and_width[0]) / tf.to_float(height)
        sx = tf.to_float(new_height_and_width[1]) / tf.to_float(width)

        return new_image, sy, sx


class RawSfMDataset(object):
    def __init__(self, longer_edge=640, output_filenames=False, num_threads=8):
        self.num_threads = num_threads
        self.longer_edge = longer_edge
        self.depth_factor = 0.001
        self.far_depth_val = 1000
        self.depth_thresh = 10.0
        self.compression_type = None
        self.output_filenames = output_filenames
    def get_dataset(self, root_dir, imroot_dir, render_paths, phase, shuffle=True, num_epoch=None, seed=None, max_examples=-1):

        self.random_transformer = RandomTransformer('', 'none', max_scale=1, min_scale=1, max_degree=0)

        if isinstance(render_paths, str):
            render_paths = [render_paths]
        num_seq = len(render_paths)
        
        if not root_dir.endswith('/'):
            root_dir += '/'
        self.root_dir = tf.convert_to_tensor(root_dir)

        if not imroot_dir.endswith('/'):
            imroot_dir += '/'
        self.imroot_dir = tf.convert_to_tensor(imroot_dir)
        
        pose_tfrecords = []
        total_num_photos = 0
        for render in render_paths:
            if phase == 'train':
                if max_examples > 0:
                    pose_fname = 'train_{}.tfrecord'.format(max_examples)
                    size_fname = None
                    print('Found {} (limited sample={})'.format(pose_fname, max_examples))
                else:
                    pose_fname = 'train.tfrecord'
                    size_fname = 'train_size.txt'
            elif phase == 'valid':
                pose_fname = 'valid.tfrecord'
                size_fname = 'valid_size.txt'
            else:
                if max_examples > 0 and os.path.exists(os.path.join(root_dir, render, 'pose_{}.tfrecord'.format(max_examples))):
                    pose_fname = 'pose_{}.tfrecord'.format(max_examples)
                    size_fname = None
                    print('Found {} (limited sample={})'.format(pose_fname, max_examples))
                else:
                    pose_fname = 'pose.tfrecord'
                    size_fname = 'size.txt'

            pose_tfrecords.append(os.path.join(root_dir, render, pose_fname))
            if size_fname is None:
                size = max_examples
            else:
                with open(os.path.join(root_dir, render, size_fname)) as f:
                    size = int(f.readline())
                print('{} has {} examples'.format(render, size))
                if max_examples > 0:
                    size = min(size, max_examples)
                    print('---> actual size={}'.format(size))

            total_num_photos += size

        self.total_num_photos = total_num_photos
        self.num_photos_per_seq_data = np.array([total_num_photos], dtype=np.int32)

        dataset = tf.data.TFRecordDataset(pose_tfrecords, compression_type=self.compression_type)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.total_num_photos, seed=seed)
        dataset = dataset.repeat(count=num_epoch)
        dataset = dataset.map(self.parser, num_parallel_calls=self.num_threads)
        dataset = dataset.batch(1)
        # dataset = dataset.prefetch(buffer_size=2)
        return dataset

    def parser(self, serialized):
        with tf.name_scope('parse_example'):
            example = tf.parse_single_example(serialized, features={
                'rgb1_filename': tf.FixedLenFeature([], tf.string),
                'rgb2_filename': tf.FixedLenFeature([], tf.string),
                'depth1_filename': tf.FixedLenFeature([], tf.string),
                'depth2_filename': tf.FixedLenFeature([], tf.string),
                'shape1': tf.FixedLenFeature([2], tf.int64),
                'shape2': tf.FixedLenFeature([2], tf.int64),
                'bbox1': tf.FixedLenFeature([4], tf.int64), # x1,x2,y1,y2
                'bbox2': tf.FixedLenFeature([4], tf.int64),
                'c1Tw': tf.FixedLenFeature([16], tf.float32),
                'c2Tw': tf.FixedLenFeature([16], tf.float32),
                'K1': tf.FixedLenFeature([9], tf.float32),
                'K2': tf.FixedLenFeature([9], tf.float32),
            })

        shape1 = example['shape1']
        shape2 = example['shape2']
        c1Tw = tf.reshape(example['c1Tw'], [4,4])
        c2Tw = tf.reshape(example['c2Tw'], [4,4])
        K1 = tf.reshape(example['K1'], [3,3])
        K2 = tf.reshape(example['K2'], [3,3])

        bb1 = example['bbox1']
        bb2 = example['bbox2']

        rgb1_filename = self.imroot_dir + example['rgb1_filename']
        rgb2_filename = self.imroot_dir + example['rgb2_filename']
        depth1_filename = self.root_dir + example['depth1_filename']
        depth2_filename = self.root_dir + example['depth2_filename']
        
        # return rgb1_filename, rgb2_filename, depth1_filename, depth2_filename

        rgb1 = self._decode_rgb(rgb1_filename, shape1)
        rgb2 = self._decode_rgb(rgb2_filename, shape2)
        depth1, valid_mask1 = self._decode_depth(depth1_filename, shape1)
        depth2, valid_mask2 = self._decode_depth(depth2_filename, shape2)

        dv1 = tf.concat([depth1, valid_mask1], axis=-1)
        dv2 = tf.concat([depth2, valid_mask2], axis=-1)

        if self.longer_edge is not None:
            rgb1, sy1, sx1 = self._resize_by_longer_edge(rgb1, self.longer_edge)
            dv1, _, _ = self._resize_by_longer_edge(dv1, self.longer_edge, inter_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            rgb2, sy2, sx2 = self._resize_by_longer_edge(rgb2, self.longer_edge)
            dv2, _, _ = self._resize_by_longer_edge(dv2, self.longer_edge, inter_method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            # modify intrinsic matrix
            S1 = make_scale_theta(sx1, sy1)
            S2 = make_scale_theta(sx2, sy2)
            K1 = tf.matmul(S1, K1)
            K2 = tf.matmul(S2, K2)

        depth1 = tf.slice(dv1, [0,0,0], [-1,-1,1])        
        valid_mask1 = tf.slice(dv1, [0,0,1], [-1,-1,1])        
        depth2 = tf.slice(dv2, [0,0,0], [-1,-1,1])        
        valid_mask2 = tf.slice(dv2, [0,0,1], [-1,-1,1])        
        
        # Pose
        c2Tc1, c1Tc2 = get_delta_pose(c1Tw, c2Tw)
        
        # get random thetas (doesnot support table-random)
        theta_params, use_augs = self.random_transformer.get_theta_params(index=None)

        if self.output_filenames:
            print('[RawSfMDataset] return filenames (mainly for lift evaluation)')
            return rgb1, rgb2, depth1, depth2, valid_mask1, valid_mask2, c2Tc1, c1Tc2, c1Tw, c2Tw, K1, K2, theta_params, use_augs, example['rgb1_filename'], example['rgb2_filename']
        else:
            return rgb1, rgb2, depth1, depth2, valid_mask1, valid_mask2, c2Tc1, c1Tc2, c1Tw, c2Tw, K1, K2, theta_params, use_augs
        
    def _decode_rgb(self, filename, shape):
        rgb = tf.read_file(filename)
        rgb = tf.image.decode_jpeg(rgb, 1)
        rgb = tf.cast(rgb, tf.float32) / 255.0
        return rgb
    
    def _decode_depth(self, filename, shape):
        depth = tf.read_file(filename)
        depth = tf.image.decode_png(depth, 1, dtype=tf.uint16) # force to load as grayscale
        depth = tf.scalar_mul(self.depth_factor, tf.cast(depth, tf.float32))
        is_zero = tf.equal(depth, tf.constant(0, dtype=tf.float32))
        valid_mask = tf.cast(tf.logical_not(is_zero), tf.float32)
        far_depth = tf.scalar_mul(self.far_depth_val, tf.ones_like(depth, dtype=tf.float32))
        depth = tf.where(is_zero, far_depth, depth)
        return depth, valid_mask

    def _resize_by_longer_edge(self, image, longer_edge, inter_method=tf.image.ResizeMethod.BILINEAR):

        def _compute_shoter_edge(prev_shorter_edge, prev_longer_edge, new_longer_edge):
            return tf.cast( prev_shorter_edge * new_longer_edge / prev_longer_edge, tf.int32)

        height = tf.shape(image)[0]
        width = tf.shape(image)[1]
        height_smaller_than_width = tf.less_equal(height, width)

        new_height_and_width = tf.cond(
            height_smaller_than_width,
            lambda: (_compute_shoter_edge(height, width, longer_edge), longer_edge),
            lambda: (longer_edge, _compute_shoter_edge(width, height, longer_edge))
        )

        new_image = tf.image.resize_images(image, new_height_and_width, method=inter_method)

        # scale ration
        sy = tf.to_float(new_height_and_width[0]) / tf.to_float(height)
        sx = tf.to_float(new_height_and_width[1]) / tf.to_float(width)

        return new_image, sy, sx


class SfMDataset(object):
    def __init__(self, out_size=(320, 320), warp_aug_mode='none', flip_pair=False, max_degree=180, max_scale=np.sqrt(2), min_scale=None, compress=False, num_threads=8):
        self.num_threads = num_threads
        self.out_size = out_size # [height, width]
        self.warp_aug_mode = warp_aug_mode
        # self.warp_aug_mode = 'none'
        # print('Disable warp_aug_mode @ SfMDataset')
        self.compression_type = 'GZIP' if compress else None

        self.depth_factor = 0.001
        self.far_depth_val = 1000
        self.depth_thresh = 10.0
        self.flip_pair = flip_pair
        self.max_scale = max_scale
        self.min_scale = min_scale
        self.max_degree = max_degree

    def get_dataset(self, root_dir, imroot_dir, render_paths, phase, batch_size=32, shuffle=True, num_epoch=None, seed=None, max_examples=-1):
        
        table_dir = os.path.join(root_dir, '../../../scannet/params/')
        self.random_transformer = RandomTransformer(table_dir, self.warp_aug_mode, max_scale=self.max_scale, min_scale=self.min_scale, max_degree=self.max_degree)

        if isinstance(render_paths, str):
            render_paths = [render_paths]
        num_seq = len(render_paths)
        
        if not root_dir.endswith('/'):
            root_dir += '/'
        self.root_dir = tf.convert_to_tensor(root_dir)

        if not imroot_dir.endswith('/'):
            imroot_dir += '/'
        self.imroot_dir = tf.convert_to_tensor(imroot_dir)
        
        pose_tfrecords = []
        total_num_photos = 0
        for render in render_paths:
            if phase == 'train':
                if max_examples > 0:
                    pose_fname = 'train_{}.tfrecord'.format(max_examples)
                    size_fname = None
                    print('Found {} (limited sample={})'.format(pose_fname, max_examples))
                else:
                    pose_fname = 'train.tfrecord'
                    size_fname = 'train_size.txt'
            elif phase == 'valid':
                pose_fname = 'valid.tfrecord'
                size_fname = 'valid_size.txt'
            else:
                if max_examples > 0 and os.path.exists(os.path.join(root_dir, render, 'pose_{}.tfrecord'.format(max_examples))):
                    pose_fname = 'pose_{}.tfrecord'.format(max_examples)
                    size_fname = None
                    print('Found {} (limited sample={})'.format(pose_fname, max_examples))
                else:
                    pose_fname = 'pose.tfrecord'
                    size_fname = 'size.txt'

            pose_tfrecords.append(os.path.join(root_dir, render, pose_fname))
            if size_fname is None:
                size = max_examples
            else:
                with open(os.path.join(root_dir, render, size_fname)) as f:
                    size = int(f.readline())
                print('{} has {} examples'.format(render, size))
                if max_examples > 0:
                    size = min(size, max_examples)
                    print('---> actual size={}'.format(size))

            total_num_photos += size

        self.total_num_photos = total_num_photos
        self.num_photos_per_seq_data = np.array([total_num_photos], dtype=np.int32)

        dataset = tf.data.TFRecordDataset(pose_tfrecords, compression_type=self.compression_type)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.total_num_photos, seed=seed)
        dataset = dataset.repeat(count=num_epoch)
        dataset = dataset.map(self.parser, num_parallel_calls=self.num_threads)
        dataset = dataset.batch(batch_size)
        # dataset = dataset.prefetch(buffer_size=2)
        return dataset

    def parser(self, serialized):
        with tf.name_scope('parse_example'):
            example = tf.parse_single_example(serialized, features={
                'rgb1_filename': tf.FixedLenFeature([], tf.string),
                'rgb2_filename': tf.FixedLenFeature([], tf.string),
                'depth1_filename': tf.FixedLenFeature([], tf.string),
                'depth2_filename': tf.FixedLenFeature([], tf.string),
                'shape1': tf.FixedLenFeature([2], tf.int64),
                'shape2': tf.FixedLenFeature([2], tf.int64),
                'bbox1': tf.FixedLenFeature([4], tf.int64), # x1,x2,y1,y2
                'bbox2': tf.FixedLenFeature([4], tf.int64),
                'c1Tw': tf.FixedLenFeature([16], tf.float32),
                'c2Tw': tf.FixedLenFeature([16], tf.float32),
                'K1': tf.FixedLenFeature([9], tf.float32),
                'K2': tf.FixedLenFeature([9], tf.float32),
            })

        # Flip images
        if self.flip_pair:
            # pair is always idx1 < idx2 so that it will be effective to switch pairs randomly
            flip_example = {
                'rgb1_filename': example['rgb2_filename'],
                'rgb2_filename': example['rgb1_filename'],
                'depth1_filename': example['depth2_filename'],
                'depth2_filename': example['depth1_filename'],
                'shape1': example['shape2'],
                'shape2': example['shape1'],
                'bbox1': example['bbox2'], 
                'bbox2': example['bbox1'],
                'c1Tw': example['c2Tw'],
                'c2Tw': example['c1Tw'],
                'K1': example['K2'],
                'K2': example['K1'],
            }
            is_flip = tf.less_equal(tf.random_uniform([]), 0.5)
            example = tf.cond(is_flip, lambda: flip_example, lambda: example)            

        shape1 = example['shape1']
        shape2 = example['shape2']
        c1Tw = tf.reshape(example['c1Tw'], [4,4])
        c2Tw = tf.reshape(example['c2Tw'], [4,4])
        K1 = tf.reshape(example['K1'], [3,3])
        K2 = tf.reshape(example['K2'], [3,3])

        bb1 = example['bbox1']
        bb2 = example['bbox2']

        rgb1_filename = self.imroot_dir + example['rgb1_filename']
        rgb2_filename = self.imroot_dir + example['rgb2_filename']
        depth1_filename = self.root_dir + example['depth1_filename']
        depth2_filename = self.root_dir + example['depth2_filename']
        
        # return rgb1_filename, rgb2_filename, depth1_filename, depth2_filename

        rgb1 = self._decode_rgb(rgb1_filename, shape1)
        rgb2 = self._decode_rgb(rgb2_filename, shape2)
        depth1, valid_mask1 = self._decode_depth(depth1_filename, shape1)
        depth2, valid_mask2 = self._decode_depth(depth2_filename, shape2)

        dv1 = tf.concat([depth1, valid_mask1], axis=-1)
        dv2 = tf.concat([depth2, valid_mask2], axis=-1)

        # rgbd1 = tf.concat([rgb1, depth1, valid_mask1], axis=-1)
        # rgbd2 = tf.concat([rgb2, depth2, valid_mask2], axis=-1)

        width1 = bb1[1] - bb1[0]
        height1 = bb1[3] - bb1[2]
        width2 = bb2[1] - bb2[0]
        height2 = bb2[3] - bb2[2]
        
        # crop
        rgb1 = tf.slice(rgb1, [bb1[2],bb1[0],0], [height1, width1, -1])
        dv1 = tf.slice(dv1, [bb1[2],bb1[0],0], [height1, width1, -1])
        rgb2 = tf.slice(rgb2, [bb2[2],bb2[0],0], [height2, width2, -1])
        dv2 = tf.slice(dv2, [bb2[2],bb2[0],0], [height2, width2, -1])
        # rgbd1 = tf.slice(rgbd1, [bb1[2],bb1[0],0], [height1, width1, -1])
        # rgbd2 = tf.slice(rgbd2, [bb2[2],bb2[0],0], [height2, width2, -1])

        # modify intrinsic matrix
        K1 = fix_intrinsic_center(K1, tf.to_float(width1)/2, tf.to_float(height1)/2)
        K2 = fix_intrinsic_center(K2, tf.to_float(width2)/2, tf.to_float(height2)/2)
        
        if self.out_size is not None:
            sy1 = float(self.out_size[0]) / tf.to_float(tf.shape(rgb1)[0])
            sx1 = float(self.out_size[1]) / tf.to_float(tf.shape(rgb1)[1])
            sy2 = float(self.out_size[0]) / tf.to_float(tf.shape(rgb2)[0])
            sx2 = float(self.out_size[1]) / tf.to_float(tf.shape(rgb2)[1])
            S1 = make_scale_theta(sx1, sy1)
            S2 = make_scale_theta(sx2, sy2)
            K1 = tf.matmul(S1, K1)
            K2 = tf.matmul(S2, K2)
            
            # do not use linear interpolation on depth and valid_masks
            rgb1 = tf.image.resize_images(rgb1, (self.out_size[0],self.out_size[1]))
            dv1 = tf.image.resize_images(dv1, (self.out_size[0],self.out_size[1]),
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            rgb2 = tf.image.resize_images(rgb2, (self.out_size[0],self.out_size[1]))
            dv2 = tf.image.resize_images(dv2, (self.out_size[0],self.out_size[1]),
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # rgbd1 = tf.image.resize_images(rgbd1, (self.out_size[0],self.out_size[1]))
            # rgbd2 = tf.image.resize_images(rgbd2, (self.out_size[0],self.out_size[1]))

        depth1 = tf.slice(dv1, [0,0,0], [-1,-1,1])        
        valid_mask1 = tf.slice(dv1, [0,0,1], [-1,-1,1])        
        depth2 = tf.slice(dv2, [0,0,0], [-1,-1,1])        
        valid_mask2 = tf.slice(dv2, [0,0,1], [-1,-1,1])        
        # rgb1 = tf.slice(rgbd1, [0,0,0], [-1,-1,1])
        # depth1 = tf.slice(rgbd1, [0,0,1], [-1,-1,1])
        # valid_mask1 = tf.slice(rgbd1, [0,0,2],[-1,-1,1])
        # valid_mask1 = tf.cast(tf.equal(valid_mask1, 1.0), tf.float32) # eliminate interpolated pixels
        # rgb2 = tf.slice(rgbd2, [0,0,0], [-1,-1,1])
        # depth2 = tf.slice(rgbd2, [0,0,1], [-1,-1,1])
        # valid_mask2 = tf.slice(rgbd2, [0,0,2],[-1,-1,1])
        # valid_mask2 = tf.cast(tf.equal(valid_mask2, 1.0), tf.float32)
        
        # Pose
        c2Tc1, c1Tc2 = get_delta_pose(c1Tw, c2Tw)
        
        # get random thetas (doesnot support table-random)
        theta_params, use_augs = self.random_transformer.get_theta_params(None)

        # add in-plane rotation
        intheta_c2Rc1 = tf.py_func(get_inplane_rotation, [c2Tc1[:3,:3]], [tf.float32])
        intheta_c1Rc2 = tf.py_func(get_inplane_rotation, [c1Tc2[:3,:3]], [tf.float32])
        theta_params = tf.concat([theta_params, intheta_c2Rc1, intheta_c1Rc2], axis=0)

        return rgb1, rgb2, depth1, depth2, valid_mask1, valid_mask2, c2Tc1, c1Tc2, c1Tw, c2Tw, K1, K2, theta_params, use_augs
        
    def _decode_rgb(self, filename, shape):
        rgb = tf.read_file(filename)
        rgb = tf.image.decode_jpeg(rgb, 1)
        rgb = tf.cast(rgb, tf.float32) / 255.0
        return rgb
    
    def _decode_depth(self, filename, shape):
        depth = tf.read_file(filename)
        depth = tf.image.decode_png(depth, 1, dtype=tf.uint16) # force to load as grayscale
        depth = tf.scalar_mul(self.depth_factor, tf.cast(depth, tf.float32))
        is_zero = tf.equal(depth, tf.constant(0, dtype=tf.float32))
        valid_mask = tf.cast(tf.logical_not(is_zero), tf.float32)
        far_depth = tf.scalar_mul(self.far_depth_val, tf.ones_like(depth, dtype=tf.float32))
        depth = tf.where(is_zero, far_depth, depth)
        return depth, valid_mask
