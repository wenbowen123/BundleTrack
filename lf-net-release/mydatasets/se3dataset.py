# -*- coding: utf-8 -*-

from PIL import Image
import math
import numpy as np
import os
import random
import sys
import glob
import h5py

import tensorflow as tf
from .dataset_tools import *

def read_text(filename):
    v = []
    for l in open(filename, 'r'):
        v.append(l.strip())
        # v.append([x for x in l.strip().split(' ')])
    return np.array(v) 

class SE3PairwiseDataset(object):
    # class for ScanNet or SfM dataset
    def __init__(self, batch_size, offset_val=2, random_offset=False, out_width=320, out_height=240, crop_center=False, max_degree=0, max_scale=np.sqrt(2), warp_aug_mode='none', num_threads=8):
        self.batch_size = batch_size
        self.offset_val = offset_val
        self.random_offset = random_offset
        self.depth_factor = 0.001
        self.far_depth_val = 1000
        self.src_width = 640
        self.src_height = 480
        self.dst_width = out_width
        self.dst_height = out_height
        self.num_threads = num_threads
        self.max_rad = np.pi / 180 * max_degree 
        self.max_scale_log = np.log(max_scale)
        self.warp_aug_mode = warp_aug_mode
        self.scale_table_path = '../../params/scale/scl-{:.2f}/rnd_table.npy'.format(max_scale)
        self.ori_table_path = '../../params/ori/deg-{}/rnd_table.npy'.format(max_degree)
        self.depth_thresh = 1.0
        self.crop_center = crop_center
        
    def set_files(self, root_dir, render_paths, max_img_num=-1):
        if isinstance(render_paths, str):
            render_paths = [render_paths]
        num_seq = len(render_paths)
        print('Number of sequences:{0}'.format(len(render_paths)))
        
        # load from render_paths
        photo_intrinsics = [None] * num_seq
        depth_intrinsics = [None] * num_seq
        cTws = [None] * num_seq
        num_photos_per_seq = [None] * num_seq
        photo_dirs = [None] * num_seq
        depth_dirs = [None] * num_seq

        for i, render_path in enumerate(render_paths):
            calib = np.load(os.path.join(root_dir, render_path, 'calib/calib.npz'))
            photo_intrinsics[i] = calib['photo_intrinsic']
            depth_intrinsics[i] = calib['depth_intrinsic']
            
            cTw = np.load(os.path.join(root_dir, render_path, 'pose/cTw.npy'))
            num_photos = (len(cTw) // self.batch_size) * self.batch_size # guarantee to be divisible by batch size
            if max_img_num > 0:
                num_photos = min(num_photos, max_img_num)
            print('#{} {} use {} / {} images'.format(i, render_path, num_photos, len(cTw)))
            cTws[i] = cTw[:num_photos]
            num_photos_per_seq[i] = num_photos
            photo_dirs[i] = os.path.join(root_dir, render_path, 'photo')
            depth_dirs[i] = os.path.join(root_dir, render_path, 'depth')


        self.cTws_data = np.concatenate(cTws, axis=0)
        self.cTws = tf.convert_to_tensor(self.cTws_data, dtype=tf.float32)
        self.photo_intrinsics_data = np.array(photo_intrinsics)
        self.photo_intrinsics = tf.convert_to_tensor(self.photo_intrinsics_data, dtype=tf.float32)
        self.depth_intrinsics_data = np.array(depth_intrinsics)
        self.depth_intrinsics = tf.convert_to_tensor(self.depth_intrinsics_data, dtype=tf.float32)
        self.num_photos_per_seq_data = np.array(num_photos_per_seq, dtype=np.int32)
        self.num_photos_per_seq = tf.convert_to_tensor(self.num_photos_per_seq_data, dtype=tf.int32)
        self.seq_offsets_data = np.concatenate([np.zeros([1], dtype=np.int32), np.cumsum(self.num_photos_per_seq_data)])
        self.seq_offsets = tf.convert_to_tensor(self.seq_offsets_data, tf.int32)

        self.intrinsics_3x3_data = self.photo_intrinsics_data

        photo_path_list = []
        depth_path_list = []
        seq_id_list = []
        for i, (pdir, ddir) in enumerate(zip(photo_dirs, depth_dirs)):
            numbers = sorted([int(os.path.splitext(x.name)[0]) for x in os.scandir(pdir)])
            num_photos = self.num_photos_per_seq_data[i]
            numbers = numbers[:num_photos]
            photos = [os.path.join(pdir, '{}.jpg'.format(img_n)) for img_n in numbers]
            depths = [os.path.join(ddir, '{}.png'.format(img_n)) for img_n in numbers]
            seq_ids = [i] * len(numbers)
            
            assert len(photos) == len(depths)
            assert len(photos) == num_photos_per_seq[i]
            
            photo_path_list += photos
            depth_path_list += depths
            seq_id_list += seq_ids
            
        # set other members
        self.photo_path_list_data = photo_path_list
        self.depth_path_list_data = depth_path_list
        self.seq_id_list_data = np.array(seq_id_list, dtype=np.int32)
        self.photo_path_list = tf.convert_to_tensor(photo_path_list)
        self.depth_path_list = tf.convert_to_tensor(depth_path_list)
        self.seq_id_list = tf.convert_to_tensor(self.seq_id_list_data, dtype=tf.int32)
        
        self.num_seq = num_seq
        self.total_num_photos = len(photo_path_list)

        if self.warp_aug_mode == 'table':
            ori_table_path = os.path.join(root_dir, self.ori_table_path)
            scale_table_path = os.path.join(root_dir, self.scale_table_path)
            self.ori_random_table_data = self._load_random_table(ori_table_path, self.total_num_photos)
            self.scale_random_table_data = self._load_random_table(scale_table_path, self.total_num_photos)
            self.ori_random_table = tf.convert_to_tensor(self.ori_random_table_data, dtype=tf.float32)
            self.scale_random_table = tf.convert_to_tensor(self.scale_random_table_data, dtype=tf.float32)

        print('[ScanNet] #sep={}, #total={}'.format(self.num_seq, self.total_num_photos))

    def get_dataset(self, shuffle=True, num_epoch=None, seed=None):

        dataset = tf.data.Dataset.range(self.total_num_photos)
        if shuffle:
            dataset = dataset.shuffle(self.total_num_photos, seed=seed)
        dataset = dataset.repeat(count=num_epoch)
        dataset = dataset.map(lambda x: self.mapfn_read_and_decode(x), num_parallel_calls=self.num_threads)
        dataset = dataset.map(self.mapfn_augment, num_parallel_calls=self.num_threads)
        dataset = dataset.batch(self.batch_size)

        return dataset

    def mapfn_read_and_decode(self, tgt_idx): 
        tgt_idx = tf.cast(tgt_idx, tf.int32) # tf.int64->tf.int32
        seq_idx = self.seq_id_list[tgt_idx]
        file_idx = tgt_idx - self.seq_offsets[seq_idx]
        num_photos = self.num_photos_per_seq[seq_idx]
        
        if self.random_offset:
            offset = tf.random_uniform((), -self.offset_val, self.offset_val, dtype=tf.int32)
        else:
            offset = self.offset_val # fixed value
        
        offset = tf.clip_by_value(file_idx+offset, 0, num_photos-1) - file_idx
        ref_idx = tgt_idx + offset
        
        photo1 = self._load_photo(tgt_idx)
        photo2 = self._load_photo(ref_idx)
        depth1, valid_mask1 = self._load_depth(tgt_idx)
        depth2, valid_mask2 = self._load_depth(ref_idx)
        
        # pose
        c1Tw = self.cTws[tgt_idx]
        c2Tw = self.cTws[ref_idx]
        c2Tc1, c1Tc2 = self._get_delta_pose(c1Tw, c2Tw)
        
        intrinsics_3x3 = self.photo_intrinsics[seq_idx,:3,:3]
        intrinsics_3x3.set_shape([3,3])



        # warp params
        print('WARP_AUG_MODE={} max_rad={}, max_scale_log={}'.format(self.warp_aug_mode, self.max_rad, self.max_scale_log))
        if self.warp_aug_mode == 'none':
            scales = tf.zeros([2], tf.float32)
            oris = tf.zeros([2], tf.float32)
        elif self.warp_aug_mode == 'fix':
            scales = tf.constant([self.max_scale_log, self.max_scale_log], tf.float32)
            oris = tf.constant([self.max_rad, self.max_rad], tf.float32)
        elif self.warp_aug_mode == 'random':
            scales = tf.random_uniform([2], minval=-self.max_scale_log, maxval=self.max_scale_log, dtype=tf.float32)
            oris = tf.random_uniform([2], minval=-self.max_rad, maxval=self.max_rad, dtype=tf.float32)
        elif self.warp_aug_mode == 'table':
            scales = self.scale_random_table[tgt_idx]
            oris = self.ori_random_table[tgt_idx]
        else:
            raise ValueError('Unknown warp_aug_mode: {}'.format(self.warp_aug_mode))
        theta_params = tf.concat([scales, oris], axis=0)

        use_aug = tf.constant(False) if self.warp_aug_mode == 'none' else tf.constant(True)

        # add in-plane rotation
        intheta_c2Rc1 = tf.py_func(get_inplane_rotation, [c2Tc1[:3,:3]], [tf.float32])
        intheta_c1Rc2 = tf.py_func(get_inplane_rotation, [c1Tc2[:3,:3]], [tf.float32])
        theta_params = tf.concat([theta_params, intheta_c2Rc1, intheta_c1Rc2], axis=0)

        return photo1, photo2, depth1, depth2, valid_mask1, valid_mask2, c2Tc1, c1Tc2, c1Tw, c2Tw, intrinsics_3x3, intrinsics_3x3, theta_params, use_aug
    
    def mapfn_augment(self, photo1, photo2, depth1, depth2, valid_mask1, valid_mask2, c2Tc1, c1Tc2, c1Tw, c2Tw, K1, K2, theta_params, use_aug):

        dv1 = tf.concat([depth1, valid_mask1], axis=-1)
        dv2 = tf.concat([depth2, valid_mask2], axis=-1)

        # Crop center
        if self.crop_center:
            # image size = [480,640] --> [480,640]
            assert self.src_width > self.src_height
            x_offset = (self.src_width-self.src_height) // 2
            new_height = new_width = self.src_height
            photo1 = tf.slice(photo1, [0,x_offset,0], [-1, new_width, -1])
            photo2 = tf.slice(photo2, [0,x_offset,0], [-1, new_width, -1])
            dv1 = tf.slice(dv1, [0,x_offset,0], [-1, new_width, -1])
            dv2 = tf.slice(dv2, [0,x_offset,0], [-1, new_width, -1])
            # modify intrinsic matrix
            K1 = fix_intrinsic_center(K1, tf.to_float(new_width)/2, tf.to_float(new_height)/2)
            K2 = fix_intrinsic_center(K2, tf.to_float(new_width)/2, tf.to_float(new_height)/2)

            dx = float(self.dst_width) / new_width
            dy = float(self.dst_height) / new_height
        else:
            dx = float(self.dst_width) / self.src_width
            dy = float(self.dst_height) / self.src_height

        # Resizing
        scale_T = self._make_scale_theta(dx, dy)
        K1 = tf.matmul(scale_T, K1)
        K2 = tf.matmul(scale_T, K2)
        photo1 = tf.image.resize_images(photo1, (self.dst_height, self.dst_width))
        photo2 = tf.image.resize_images(photo2, (self.dst_height, self.dst_width))

        # do not use linear interpolation on valid_masks1
        dv1 = tf.image.resize_images(dv1, (self.dst_height, self.dst_width), 
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        dv2 = tf.image.resize_images(dv2, (self.dst_height, self.dst_width), 
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        depth1 = tf.slice(dv1, [0,0,0],[-1,-1,1])
        valid_mask1 = tf.slice(dv1, [0,0,1],[-1,-1,1])
        depth2 = tf.slice(dv2, [0,0,0],[-1,-1,1])
        valid_mask2 = tf.slice(dv2, [0,0,1],[-1,-1,1])

        return photo1, photo2, depth1, depth2, valid_mask1, valid_mask2, c2Tc1, c1Tc2, c1Tw, c2Tw, K1, K2, theta_params, use_aug

    def _load_photo(self, idx):
        photo = tf.read_file(self.photo_path_list[idx])
        photo = tf.image.decode_jpeg(photo, 1) # force to load as grayscale
        # photo = tf.image.rgb_to_grayscale(photo)
        photo = tf.cast(photo, tf.float32) / 255.0 # normalize

        photo.set_shape((self.src_height, self.src_width, 1))
        photo.set_shape((None, None, 1))

        return photo
    
    def _load_depth(self, idx):
        depth = tf.read_file(self.depth_path_list[idx])
        depth = tf.image.decode_png(depth, 1, dtype=tf.uint16) # force to load as grayscale
        depth = tf.scalar_mul(self.depth_factor, tf.cast(depth, tf.float32))
        is_zero = tf.equal(depth, tf.constant(0, dtype=tf.float32))
        valid_mask = tf.cast(tf.logical_not(is_zero), tf.float32)
        far_depth = tf.scalar_mul(self.far_depth_val, tf.ones_like(depth, dtype=tf.float32))
        # pay attention when you visualize depth due to dynamic range (0~1000)
        depth = tf.where(is_zero, far_depth, depth)
        # depth.set_shape((self.src_height, self.src_width, 1))
        depth.set_shape((None, None, 1))

        return depth, valid_mask
        
    def _get_delta_pose(self, c1Tw, c2Tw):
        # cTw = world to camera pose [4x4 matrix]
        # return = c2Tc1, which means c1 to c2 pose
        c1Rw = tf.slice(c1Tw, [0,0], [3,3])
        c2Rw = tf.slice(c2Tw, [0,0], [3,3])
        c1Pw = tf.slice(c1Tw, [0,3], [3,1])
        c2Pw = tf.slice(c2Tw, [0,3], [3,1])
        wPc1 = -tf.matmul(c1Rw, c1Pw, transpose_a=True) # wPc = -wRc cPw
        wPc2 = -tf.matmul(c2Rw, c2Pw, transpose_a=True) # wPc = -wRc cPw
        c2Rc1 = tf.matmul(c2Rw, c1Rw, transpose_b=True) # c2Rc1 = c2Rw wRc1
        c2Pc1 = tf.matmul(c2Rw, wPc1-wPc2) # c2Pc1 = c2Rw (wPc1-wPc2)
        # c2Tc1 (4x4) = 
        # | c2Rc1 c2Pc1 |
        # |   0     1   |
        c2Tc1 = tf.concat([c2Rc1, c2Pc1], axis=1)
        c2Tc1 = tf.concat([c2Tc1, tf.constant([[0,0,0,1]], dtype=tf.float32)], axis=0)
        c1Tc2 = tf.matrix_inverse(c2Tc1)
        return c2Tc1, c1Tc2

    def _make_scale_theta(self, sx, sy):
        # 3x3 matrix
        theta = tf.stack(
            [sx, 0, 0,
             0, sy, 0,
             0, 0, 1])
        return tf.cast(tf.reshape(theta, [3,3]), tf.float32)

    def _load_random_table(self, table_path, min_table_size):
        if not os.path.join(table_path):
            raise ValueError('Cannot load random-table from {}'.format(table_path))
        random_table = np.load(table_path) # [N, 2]

        if len(random_table) < min_table_size:
            raise ValueError('Shortage of table size, table size should be larger than {} but the actual size is {} in {}'.format(min_table_size, random_table, table_path))
        print('load random table ({}) from {}'.format(random_table.shape, table_path))
        return random_table

class SubsampleSE3PairwiseDataset(SE3PairwiseDataset):
    def __init__(self, batch_size, offset_val=2, random_offset=False, out_width=320, out_height=240, max_degree=0, max_scale=np.sqrt(2), warp_aug_mode='none', num_threads=8):
        self.batch_size = batch_size
        self.offset_val = offset_val
        self.random_offset = random_offset
        self.depth_factor = 0.001
        self.far_depth_val = 1000
        self.src_width = 640
        self.src_height = 480
        self.dst_width = out_width
        self.dst_height = out_height
        self.num_threads = num_threads
        self.max_rad = np.pi / 180 * max_degree 
        self.max_scale_log = np.log(max_scale)
        self.warp_aug_mode = warp_aug_mode
        self.scale_table_path = '../../params/scale/scl-{:.2f}/rnd_table.npy'.format(max_scale)
        self.ori_table_path = '../../params/ori/deg-{}/rnd_table.npy'.format(max_degree)
        self.crop_center = False
        self.depth_thresh = 1.0
        
    def set_files(self, root_dir, render_paths, max_img_num=-1):
        if isinstance(render_paths, str):
            render_paths = [render_paths]
        num_seq = len(render_paths)
        print('Number of sequences:{0}'.format(len(render_paths)))
        
        # load from render_paths
        photo_intrinsics = [None] * num_seq
        depth_intrinsics = [None] * num_seq
        cTws = [None] * num_seq
        num_photos_per_seq = [None] * num_seq
        photo_dirs = [None] * num_seq
        depth_dirs = [None] * num_seq

        photo_path_list = []
        depth_path_list = []
        seq_id_list = []

        for i, render_path in enumerate(render_paths):
            valid_numbers = read_text(os.path.join(root_dir, render_path, 'valid_number.txt')).astype(np.int32)
            subsample_inds = np.where(valid_numbers%10==0)[0]
            subsample_numbers = valid_numbers[subsample_inds]

            calib = np.load(os.path.join(root_dir, render_path, 'calib/calib.npz'))
            photo_intrinsics[i] = calib['photo_intrinsic']
            depth_intrinsics[i] = calib['depth_intrinsic']
            
            cTw = np.load(os.path.join(root_dir, render_path, 'pose/cTw.npy'))
            cTws[i] = cTw[subsample_inds]
            num_photos_per_seq[i] = len(cTws[i])

            photos = [os.path.join(root_dir, render_path, 'photo/{}.jpg'.format(n)) for n in subsample_numbers]
            depths = [os.path.join(root_dir, render_path, 'depth/{}.png'.format(n)) for n in subsample_numbers]
            seq_ids = [i] * len(subsample_numbers)
            assert len(photos) == len(depths)
            assert len(photos) == num_photos_per_seq[i]

            photo_path_list += photos
            depth_path_list += depths
            seq_id_list += seq_ids

        self.cTws_data = np.concatenate(cTws, axis=0)
        self.cTws = tf.convert_to_tensor(self.cTws_data, dtype=tf.float32)
        self.photo_intrinsics_data = np.array(photo_intrinsics)
        self.photo_intrinsics = tf.convert_to_tensor(self.photo_intrinsics_data, dtype=tf.float32)
        self.depth_intrinsics_data = np.array(depth_intrinsics)
        self.depth_intrinsics = tf.convert_to_tensor(self.depth_intrinsics_data, dtype=tf.float32)
        self.num_photos_per_seq_data = np.array(num_photos_per_seq, dtype=np.int32)
        self.num_photos_per_seq = tf.convert_to_tensor(self.num_photos_per_seq_data, dtype=tf.int32)
        self.seq_offsets_data = np.concatenate([np.zeros([1], dtype=np.int32), np.cumsum(self.num_photos_per_seq_data)])
        self.seq_offsets = tf.convert_to_tensor(self.seq_offsets_data, tf.int32)

        self.intrinsics_3x3_data = self.photo_intrinsics_data

        self.photo_path_list_data = photo_path_list
        self.depth_path_list_data = depth_path_list
        self.seq_id_list_data = np.array(seq_id_list, dtype=np.int32)
        self.photo_path_list = tf.convert_to_tensor(photo_path_list)
        self.depth_path_list = tf.convert_to_tensor(depth_path_list)
        self.seq_id_list = tf.convert_to_tensor(self.seq_id_list_data, dtype=tf.int32)
        self.num_seq = num_seq
        self.total_num_photos = len(photo_path_list)

        print('[SubsampleScanNet10] #sep={}, #total={}'.format(self.num_seq, self.total_num_photos))

class ColorSubsampleSE3PairwiseDataset(SE3PairwiseDataset):
    def __init__(self, batch_size, offset_val=2, random_offset=False, out_width=320, out_height=240, max_degree=0, max_scale=np.sqrt(2), warp_aug_mode='none', num_threads=8):
        self.batch_size = batch_size
        self.offset_val = offset_val
        self.random_offset = random_offset
        self.depth_factor = 0.001
        self.far_depth_val = 1000
        self.src_width = 640
        self.src_height = 480
        self.dst_width = out_width
        self.dst_height = out_height
        self.num_threads = num_threads
        self.max_rad = np.pi / 180 * max_degree 
        self.max_scale_log = np.log(max_scale)
        self.warp_aug_mode = warp_aug_mode
        self.scale_table_path = '../../params/scale/scl-{:.2f}/rnd_table.npy'.format(max_scale)
        self.ori_table_path = '../../params/ori/deg-{}/rnd_table.npy'.format(max_degree)
        self.crop_center = False
        self.depth_thresh = 1.0
        
    def set_files(self, root_dir, render_paths, max_img_num=-1):
        if isinstance(render_paths, str):
            render_paths = [render_paths]
        num_seq = len(render_paths)
        print('Number of sequences:{0}'.format(len(render_paths)))
        
        # load from render_paths
        photo_intrinsics = [None] * num_seq
        depth_intrinsics = [None] * num_seq
        cTws = [None] * num_seq
        num_photos_per_seq = [None] * num_seq
        photo_dirs = [None] * num_seq
        depth_dirs = [None] * num_seq

        photo_path_list = []
        depth_path_list = []
        seq_id_list = []

        for i, render_path in enumerate(render_paths):
            valid_numbers = read_text(os.path.join(root_dir, render_path, 'valid_number.txt')).astype(np.int32)
            subsample_inds = np.where(valid_numbers%10==0)[0]
            subsample_numbers = valid_numbers[subsample_inds]

            calib = np.load(os.path.join(root_dir, render_path, 'calib/calib.npz'))
            photo_intrinsics[i] = calib['photo_intrinsic']
            depth_intrinsics[i] = calib['depth_intrinsic']
            
            cTw = np.load(os.path.join(root_dir, render_path, 'pose/cTw.npy'))
            cTws[i] = cTw[subsample_inds]
            num_photos_per_seq[i] = len(cTws[i])

            photos = [os.path.join(root_dir, render_path, 'photo/{}.jpg'.format(n)) for n in subsample_numbers]
            depths = [os.path.join(root_dir, render_path, 'depth/{}.png'.format(n)) for n in subsample_numbers]
            seq_ids = [i] * len(subsample_numbers)
            assert len(photos) == len(depths)
            assert len(photos) == num_photos_per_seq[i]

            photo_path_list += photos
            depth_path_list += depths
            seq_id_list += seq_ids

        self.cTws_data = np.concatenate(cTws, axis=0)
        self.cTws = tf.convert_to_tensor(self.cTws_data, dtype=tf.float32)
        self.photo_intrinsics_data = np.array(photo_intrinsics)
        self.photo_intrinsics = tf.convert_to_tensor(self.photo_intrinsics_data, dtype=tf.float32)
        self.depth_intrinsics_data = np.array(depth_intrinsics)
        self.depth_intrinsics = tf.convert_to_tensor(self.depth_intrinsics_data, dtype=tf.float32)
        self.num_photos_per_seq_data = np.array(num_photos_per_seq, dtype=np.int32)
        self.num_photos_per_seq = tf.convert_to_tensor(self.num_photos_per_seq_data, dtype=tf.int32)
        self.seq_offsets_data = np.concatenate([np.zeros([1], dtype=np.int32), np.cumsum(self.num_photos_per_seq_data)])
        self.seq_offsets = tf.convert_to_tensor(self.seq_offsets_data, tf.int32)

        self.intrinsics_3x3_data = self.photo_intrinsics_data

        self.photo_path_list_data = photo_path_list
        self.depth_path_list_data = depth_path_list
        self.seq_id_list_data = np.array(seq_id_list, dtype=np.int32)
        self.photo_path_list = tf.convert_to_tensor(photo_path_list)
        self.depth_path_list = tf.convert_to_tensor(depth_path_list)
        self.seq_id_list = tf.convert_to_tensor(self.seq_id_list_data, dtype=tf.int32)
        self.num_seq = num_seq
        self.total_num_photos = len(photo_path_list)

        print('[SubsampleScanNet10] #sep={}, #total={}'.format(self.num_seq, self.total_num_photos))

    def get_dataset(self, shuffle=True, num_epoch=None, seed=None):

        dataset = tf.data.Dataset.range(self.total_num_photos)
        if shuffle:
            dataset = dataset.shuffle(self.total_num_photos, seed=seed)
        dataset = dataset.repeat(count=num_epoch)
        dataset = dataset.map(lambda x: self.mapfn_read_and_decode(x), num_parallel_calls=self.num_threads)
        dataset = dataset.map(self.mapfn_augment, num_parallel_calls=self.num_threads)
        dataset = dataset.batch(self.batch_size)

        return dataset

    def mapfn_read_and_decode(self, tgt_idx): 
        tgt_idx = tf.cast(tgt_idx, tf.int32) # tf.int64->tf.int32
        seq_idx = self.seq_id_list[tgt_idx]
        file_idx = tgt_idx - self.seq_offsets[seq_idx]
        num_photos = self.num_photos_per_seq[seq_idx]
        
        if self.random_offset:
            offset = tf.random_uniform((), -self.offset_val, self.offset_val, dtype=tf.int32)
        else:
            offset = self.offset_val # fixed value
        
        offset = tf.clip_by_value(file_idx+offset, 0, num_photos-1) - file_idx
        ref_idx = tgt_idx + offset
        
        photo1, rgb1 = self._load_photo(tgt_idx)
        photo2, rgb2 = self._load_photo(ref_idx)
        depth1, valid_mask1 = self._load_depth(tgt_idx)
        depth2, valid_mask2 = self._load_depth(ref_idx)
        
        # pose
        c1Tw = self.cTws[tgt_idx]
        c2Tw = self.cTws[ref_idx]
        c2Tc1, c1Tc2 = self._get_delta_pose(c1Tw, c2Tw)
        
        intrinsics_3x3 = self.photo_intrinsics[seq_idx,:3,:3]
        intrinsics_3x3.set_shape([3,3])



        # warp params
        print('WARP_AUG_MODE={} max_rad={}, max_scale_log={}'.format(self.warp_aug_mode, self.max_rad, self.max_scale_log))
        if self.warp_aug_mode == 'none':
            scales = tf.zeros([2], tf.float32)
            oris = tf.zeros([2], tf.float32)
        elif self.warp_aug_mode == 'fix':
            scales = tf.constant([self.max_scale_log, self.max_scale_log], tf.float32)
            oris = tf.constant([self.max_rad, self.max_rad], tf.float32)
        elif self.warp_aug_mode == 'random':
            scales = tf.random_uniform([2], minval=-self.max_scale_log, maxval=self.max_scale_log, dtype=tf.float32)
            oris = tf.random_uniform([2], minval=-self.max_rad, maxval=self.max_rad, dtype=tf.float32)
        elif self.warp_aug_mode == 'table':
            scales = self.scale_random_table[tgt_idx]
            oris = self.ori_random_table[tgt_idx]
        else:
            raise ValueError('Unknown warp_aug_mode: {}'.format(self.warp_aug_mode))
        theta_params = tf.concat([scales, oris], axis=0)

        use_aug = tf.constant(False) if self.warp_aug_mode == 'none' else tf.constant(True)

        return photo1, photo2, depth1, depth2, valid_mask1, valid_mask2, c2Tc1, c1Tc2, c1Tw, c2Tw, intrinsics_3x3, intrinsics_3x3, theta_params, use_aug, rgb1, rgb2
    
    def mapfn_augment(self, photo1, photo2, depth1, depth2, valid_mask1, valid_mask2, c2Tc1, c1Tc2, c1Tw, c2Tw, K1, K2, theta_params, use_aug, rgb1, rgb2):

        dv1 = tf.concat([depth1, valid_mask1], axis=-1)
        dv2 = tf.concat([depth2, valid_mask2], axis=-1)

        # Crop center
        if self.crop_center:
            # image size = [480,640] --> [480,640]
            assert self.src_width > self.src_height
            x_offset = (self.src_width-self.src_height) // 2
            new_height = new_width = self.src_height
            photo1 = tf.slice(photo1, [0,x_offset,0], [-1, new_width, -1])
            photo2 = tf.slice(photo2, [0,x_offset,0], [-1, new_width, -1])
            rgb1 = tf.slice(rgb1, [0,x_offset,0], [-1, new_width, -1])
            rgb2 = tf.slice(rgb2, [0,x_offset,0], [-1, new_width, -1])
            dv1 = tf.slice(dv1, [0,x_offset,0], [-1, new_width, -1])
            dv2 = tf.slice(dv2, [0,x_offset,0], [-1, new_width, -1])
            # modify intrinsic matrix
            K1 = fix_intrinsic_center(K1, tf.to_float(new_width)/2, tf.to_float(new_height)/2)
            K2 = fix_intrinsic_center(K2, tf.to_float(new_width)/2, tf.to_float(new_height)/2)

            dx = float(self.dst_width) / new_width
            dy = float(self.dst_height) / new_height
        else:
            dx = float(self.dst_width) / self.src_width
            dy = float(self.dst_height) / self.src_height

        # Resizing
        scale_T = self._make_scale_theta(dx, dy)
        K1 = tf.matmul(scale_T, K1)
        K2 = tf.matmul(scale_T, K2)
        photo1 = tf.image.resize_images(photo1, (self.dst_height, self.dst_width))
        photo2 = tf.image.resize_images(photo2, (self.dst_height, self.dst_width))
        rgb1 = tf.image.resize_images(rgb1, (self.dst_height, self.dst_width))
        rgb2 = tf.image.resize_images(rgb2, (self.dst_height, self.dst_width))

        # do not use linear interpolation on valid_masks1
        dv1 = tf.image.resize_images(dv1, (self.dst_height, self.dst_width), 
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        dv2 = tf.image.resize_images(dv2, (self.dst_height, self.dst_width), 
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        depth1 = tf.slice(dv1, [0,0,0],[-1,-1,1])
        valid_mask1 = tf.slice(dv1, [0,0,1],[-1,-1,1])
        depth2 = tf.slice(dv2, [0,0,0],[-1,-1,1])
        valid_mask2 = tf.slice(dv2, [0,0,1],[-1,-1,1])

        return photo1, photo2, depth1, depth2, valid_mask1, valid_mask2, c2Tc1, c1Tc2, c1Tw, c2Tw, K1, K2, theta_params, use_aug, rgb1, rgb2

    def _load_photo(self, idx):

        img = tf.read_file(self.photo_path_list[idx])
        rgb = tf.image.decode_jpeg(img, 3)
        photo = tf.image.decode_jpeg(img, 1) # force to load as grayscale
        # photo = tf.image.rgb_to_grayscale(photo)
        rgb = tf.cast(rgb, tf.float32) / 255.0
        photo = tf.cast(photo, tf.float32) / 255.0 # normalize

        photo.set_shape((self.src_height, self.src_width, 1))
        photo.set_shape((None, None, 1))

        rgb.set_shape((self.src_height, self.src_width, 3))
        rgb.set_shape((None, None, 3))

        return photo, rgb
