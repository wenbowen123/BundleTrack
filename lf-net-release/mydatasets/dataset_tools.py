# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import math

def get_delta_pose(c1Tw, c2Tw):
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

def make_scale_theta(sx, sy):
    # 3x3 matrix
    theta = tf.stack(
        [sx, 0, 0,
         0, sy, 0,
         0, 0, 1])
    return tf.cast(tf.reshape(theta, [3,3]), tf.float32)

def fix_intrinsic_center(intrinsics_3x3, u0, v0):
    # intrinsics_3x3 : 3x3
    # u0, v0 = scalar
    left3x2 = tf.slice(intrinsics_3x3, [0,0],[-1,2]) # [3,2]
    uv0s = tf.stack([u0, v0, 1.0])[:,None]
    intrinsics_3x3 = tf.concat([left3x2, uv0s], axis=-1)

    return intrinsics_3x3

def logarithm_so3(R_mat):
    SE3_ANGLE_APPROX_THRESH_PI = 1.0
    SE3_ANGLE_APPROX_THRESH_ZERO = 0.03    
    OneOver6 = 1.0/6.0
    OneOver7 = 1.0/7.0
    OneOver9 = 1.0/9.0
    def A_approximation(theta2):
        return 1.0 - theta2 * OneOver6 * (1.0 - 0.05 * theta2 * (1.0 - theta2 * OneOver6 * OneOver7))

    trR = float(R_mat.trace().clip(-1.0, 3.0))
    theta = math.acos((trR - 1.0) * 0.5)
    ln_Rvec = np.empty(3, dtype=np.float32)
    if theta > np.pi - SE3_ANGLE_APPROX_THRESH_PI:
        R_diag = R_mat.diagonal()
        a = int(R_diag.argmax())
        b = (a+1) % 3
        c = (a+2) % 3
        s = math.sqrt(1.0 + R_diag[a]-R_diag[b]-R_diag[c])
        ln_Rvec[a] = s * 0.5
        ln_Rvec[b] = (R_mat[b,a] + R_mat[a,b]) * 0.5 / s
        ln_Rvec[c] = (R_mat[c,a] + R_mat[a,c]) * 0.5 / s
        vNorm = ln_Rvec.dot(ln_Rvec)
        if vNorm != 0:
            ln_Rvec /= vNorm
        wNorm = (R_mat[c,b] - R_mat[b,c]) * 0.5 / s
        ln_Rvec *= 2.0 * math.atan(vNorm/wNorm)
    else:
        theta2 = 0
        A = 0
        if theta > SE3_ANGLE_APPROX_THRESH_ZERO:
            theta2 = theta * theta
            A = math.sin(theta) / theta
        else:
            theta2 = R_mat[0,1] * R_mat[0,1] + R_mat[0,2] * R_mat[0,2] + R_mat[1,2] * R_mat[1,2]
            A = A_approximation(theta2)
        ln_R = (R_mat - R_mat.T) * 0.5 / A
        ln_Rvec[0] = 0.5 * (ln_R[2,1] - ln_R[1,2])
        ln_Rvec[1] = 0.5 * (ln_R[0,2] - ln_R[2,0])
        ln_Rvec[2] = 0.5 * (ln_R[1,0] - ln_R[0,1])
    return ln_Rvec    

def get_inplane_rotation(R_mat):
    rvec = logarithm_so3(R_mat)
    zvec = np.array([0,0,1], dtype=np.float32)
    rot_val = np.dot(rvec, zvec) # radian
    return rot_val

class RandomTransformer(object):
    def __init__(self, table_dir, aug_mode, max_scale, max_degree, dataset_size=10000, min_scale=None):
        self.max_rad = np.pi / 180 * max_degree 
        self.max_scale_log = np.log(max_scale)
        self.min_scale_log = -self.max_scale_log if min_scale is None else np.log(min_scale)
        self.aug_mode = aug_mode
        self.scale_table_path = os.path.join(table_dir, 'scale/scl-{:.2f}/rnd_table.npy'.format(max_scale))
        self.ori_table_path = os.path.join(table_dir, 'ori/deg-{}/rnd_table.npy'.format(max_degree))

        if self.aug_mode == 'table':
            self.ori_random_table_data = self._load_random_table(ori_table_path, self.total_num_photos, min_table_size=dataset_size)
            self.scale_random_table_data = self._load_random_table(scale_table_path, self.total_num_photos, min_table_size=dataset_size)
            self.ori_random_table = tf.convert_to_tensor(self.ori_random_table_data, dtype=tf.float32)
            self.scale_random_table = tf.convert_to_tensor(self.scale_random_table_data, dtype=tf.float32)


    def _load_random_table(self, table_path, min_table_size):
        if not os.path.join(table_path):
            raise ValueError('Cannot load random-table from {}'.format(table_path))
        random_table = np.load(table_path) # [N, 2]

        if len(random_table) < min_table_size:
            raise ValueError('Shortage of table size, table size should be larger than {} but the actual size is {} in {}'.format(min_table_size, random_table, table_path))
        print('load random table ({}) from {}'.format(random_table.shape, table_path))
        return random_table

    def get_theta_params(self, index):
        print('aug_mode={} max_rad={}, max_scale_log={}'.format(self.aug_mode, self.max_rad, self.max_scale_log))
        if self.aug_mode == 'none':
            scales = tf.zeros([2], tf.float32) # exp(0) = 1
            oris = tf.zeros([2], tf.float32)
        elif self.aug_mode == 'fix':
            scales = tf.constant([self.max_scale_log, self.max_scale_log], tf.float32)
            oris = tf.constant([self.max_rad, self.max_rad], tf.float32)
        elif self.aug_mode == 'random':
            print('Add random logscale={:.2f}~{:.2f}, ori={}~{}'.format(self.min_scale_log, self.max_scale_log, -self.max_rad, self.max_rad))
            scales = tf.random_uniform([2], minval=self.min_scale_log, maxval=self.max_scale_log, dtype=tf.float32)
            oris = tf.random_uniform([2], minval=-self.max_rad, maxval=self.max_rad, dtype=tf.float32)
        elif self.aug_mode == 'table':
            scales = self.scale_random_table[index]
            oris = self.ori_random_table[index]
        else:
            raise ValueError('Unknown aug_mode: {}'.format(self.aug_mode))
        theta_params = tf.concat([scales, oris], axis=0)

        use_aug = tf.constant(False) if self.aug_mode == 'none' else tf.constant(True)

        return theta_params, use_aug

