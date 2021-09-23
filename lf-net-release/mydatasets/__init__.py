from .se3dataset import SE3PairwiseDataset
from .sfmdataset import SfMDataset, RawSfMDataset
from spatial_transformer import transformer_crop

import tensorflow as tf

def fix_intrinsic_center(intrinsics_3x3, u0, v0):
    batch_size = tf.shape(intrinsics_3x3)[0]
    left3x2 = tf.slice(intrinsics_3x3, [0,0,0],[-1,-1,2]) # [B,3,2]
    uv0s = tf.tile(tf.constant([u0, v0, 1.0], dtype=tf.float32)[None], [batch_size,1])[...,None] # [B,3] --> [B,3,1]
    intrinsics_3x3 = tf.concat([left3x2, uv0s], axis=-1)
    
    return intrinsics_3x3

def make_thetas(batch_size, scales=None, oris=None, shifts=None):
    # scales, oris : [B,1]
    # shifts: [B,2] u,v
    T = tf.tile(tf.eye(3)[None], [batch_size, 1, 1])
    zeros = tf.zeros([batch_size, 1], dtype=tf.float32)
    ones = tf.ones([batch_size, 1], dtype=tf.float32)
    if oris is not None:
        sins = tf.sin(oris)
        coss = tf.cos(oris)
        R = tf.concat([coss, -sins, zeros,
                      sins,  coss, zeros,
                      zeros, zeros, ones,
                     ], axis=1)
        R = tf.reshape(R, [-1,3,3])
        T = R
    if scales is not None:
        S = tf.concat([scales, zeros, zeros,
                      zeros, scales, zeros,
                      zeros, zeros, ones,
                     ], axis=1)
        S = tf.reshape(S, [-1,3,3])
        T = tf.matmul(T, S)
    if shifts is not None:
        us = tf.slice(shifts, [0,0], [-1,1])
        vs = tf.slice(shifts, [0,1], [-1,1])
        D = tf.concat([ones,  zeros, us,
                       zeros, ones,  vs,
                       zeros, zeros, ones,
                      ], axis=1)
        D = tf.reshape(D, [-1,3,3])
        T = tf.matmul(T, D)
    return T

def euclidean_augmentation(next_batch, out_size, rot_aug, scale_aug):
    # Apply data augmentation of euclidean transformation 
    photos1, photos2, depths1, depths2, valid_masks1, valid_masks2, c2Tc1s, c1Tc2s, c1Tws, c2Tws, Ks1, Ks2, theta_params, use_aug = next_batch
    
    batch_size = tf.shape(photos1)[0]
    data_height, data_width = photos1.get_shape().as_list()[1:3]
    out_height, out_width = out_size
    theta_params4 = theta_params[:,:4]
    print(theta_params4)
    scales1_log, scales2_log, oris1, oris2 = tf.split(theta_params4, 4, axis=1)
    scales1 = tf.exp(scales1_log)
    scales2 = tf.exp(scales2_log)

    thetas1 = make_thetas(batch_size, None,None) # don't apply augmentation

    if not scale_aug:
        scales2 = None
    else:
        tf.summary.histogram('scale_aug', scales2_log)
    if not rot_aug:
        oris2 = None
    else:
        tf.summary.histogram('rot_aug', oris2)
    thetas2 = make_thetas(batch_size, scales2, oris2)


    inv_thetas1 = tf.matrix_inverse(thetas1)
    inv_thetas2 = tf.matrix_inverse(thetas2)

    center_xy = tf.tile(tf.stack([data_width/2, data_height/2])[None], [batch_size,1])

    rgbdv1 = tf.concat([photos1, depths1, valid_masks1], axis=-1)
    rgbdv2 = tf.concat([photos2, depths2, valid_masks2], axis=-1)

    rgbdv1_t = tf.cond(use_aug[0], 
                      lambda: transformer_crop(rgbdv1, (out_width, out_height), tf.range(batch_size), kpts_xy=center_xy, thetas=inv_thetas1),
                      lambda: tf.identity(rgbdv1)
                     )
    rgbdv2_t = tf.cond(use_aug[0],
                      lambda: transformer_crop(rgbdv2, (out_width, out_height), tf.range(batch_size), kpts_xy=center_xy, thetas=inv_thetas2),
                      lambda: tf.identity(rgbdv2)
                     )

    photos1 = tf.slice(rgbdv1_t, [0,0,0,0], [-1,-1,-1,1])
    depths1 = tf.slice(rgbdv1_t, [0,0,0,1], [-1,-1,-1,1])
    valid_masks1 = tf.slice(rgbdv1_t, [0,0,0,2], [-1,-1,-1,1])
    valid_masks1 = tf.cast(tf.equal(valid_masks1, 1.0), tf.float32) # eliminate interpolated pixels

    photos2 = tf.slice(rgbdv2_t, [0,0,0,0], [-1,-1,-1,1])
    depths2 = tf.slice(rgbdv2_t, [0,0,0,1], [-1,-1,-1,1])
    valid_masks2 = tf.slice(rgbdv2_t, [0,0,0,2], [-1,-1,-1,1])
    valid_masks2 = tf.cast(tf.equal(valid_masks2, 1.0), tf.float32) # eliminate interpolated pixels

    Ks1 = tf.cond(use_aug[0],
                  lambda: fix_intrinsic_center(Ks1, out_width/2, out_height/2),
                  lambda: tf.identity(Ks1)
                     )
    Ks2 = tf.cond(use_aug[0],
                  lambda: fix_intrinsic_center(Ks2, out_width/2, out_height/2),
                  lambda: tf.identity(Ks2)
                     )    
    next_batch = [photos1, photos2, depths1, depths2, valid_masks1, valid_masks2, 
                  c2Tc1s, c1Tc2s, c1Tws, c2Tws, Ks1, Ks2, 
                  thetas1, thetas2, inv_thetas1, inv_thetas2, theta_params]

    return next_batch    


# def euclidean_augmentation(next_batch, out_size, rot_aug, scale_aug):
#     # Apply data augmentation of euclidean transformation 
#     photos1, photos2, depths1, depths2, c2Tc1s, c1Tc2s, c1Tws, c2Tws, intrinsics_3x3, theta_params, use_aug = next_batch
    
#     batch_size = tf.shape(photos1)[0]
#     data_height, data_width = photos1.get_shape().as_list()[1:3]
#     out_height, out_width = out_size

#     scales1_log, scales2_log, oris1, oris2 = tf.split(theta_params, 4, axis=1)
#     scales1 = tf.exp(scales1_log)
#     scales2 = tf.exp(scales2_log)

#     thetas1 = make_thetas(batch_size, None,None) # don't apply augmentation

#     if not scale_aug:
#         scales2 = None
#     else:
#         tf.summary.histogram('scale_aug', scales2_log)
#     if not rot_aug:
#         oris2 = None
#     else:
#         tf.summary.histogram('rot_aug', oris2)
#     thetas2 = make_thetas(batch_size, scales2, oris2)


#     inv_thetas1 = tf.matrix_inverse(thetas1)
#     inv_thetas2 = tf.matrix_inverse(thetas2)

#     center_xy = tf.tile(tf.stack([data_width/2, data_height/2])[None], [batch_size,1])

#     rgbd1 = tf.concat([photos1, depths1], axis=-1)
#     rgbd2 = tf.concat([photos2, depths2], axis=-1)
    
#     rgbd1_t = tf.cond(use_aug[0], 
#                       lambda: transformer_crop(rgbd1, (out_width, out_height), tf.range(batch_size), kpts_xy=center_xy, thetas=inv_thetas1),
#                       lambda: tf.identity(rgbd1)
#                      )
#     rgbd2_t = tf.cond(use_aug[0],
#                       lambda: transformer_crop(rgbd2, (out_width, out_height), tf.range(batch_size), kpts_xy=center_xy, thetas=inv_thetas2),
#                       lambda: tf.identity(rgbd2)
#                      )

#     photos1 = tf.slice(rgbd1_t, [0,0,0,0], [-1,-1,-1,1])
#     depths1 = tf.slice(rgbd1_t, [0,0,0,1], [-1,-1,-1,1])
    
#     photos2 = tf.slice(rgbd2_t, [0,0,0,0], [-1,-1,-1,1])
#     depths2 = tf.slice(rgbd2_t, [0,0,0,1], [-1,-1,-1,1])

#     intrinsics_3x3 = tf.cond(use_aug[0],
#                       lambda: fix_intrinsic_center(intrinsics_3x3, out_width/2, out_height/2),
#                       lambda: tf.identity(intrinsics_3x3)
#                      )
    
#     next_batch = [photos1, photos2, depths1, depths2, c2Tc1s, c1Tc2s, c1Tws, c2Tws, intrinsics_3x3, thetas1, thetas2, inv_thetas1, inv_thetas2, theta_params]

#     return next_batch    

