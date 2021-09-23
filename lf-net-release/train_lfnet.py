#!/usr/bin/env python

# Self Supervised Joint Training for Keypoints Detector & Descriptor

from __future__ import print_function
import os
import sys
import numpy as np
import tensorflow as tf
import importlib
import time
import cv2
from tqdm import tqdm
import pickle

LOCAL_PATH = './'
if LOCAL_PATH not in sys.path:
    sys.path.append(LOCAL_PATH)

from mydatasets import *
# from datasets.scenenet import SceneNetPairwiseDataset
# from datasets.se3dataset import SE3PairwiseDataset

from det_tools import *
from eval_tools import compute_sift, compute_sift_multi_scale, draw_match, draw_keypoints, draw_match2
from common.tf_layer_utils import *
from common.tf_train_utils import get_optimizer, get_piecewise_lr, get_activation_fn
from common.tfvisualizer import log_images, convert_tile_image

from inference import *


MODEL_PATH = './models'
if MODEL_PATH not in sys.path:
    sys.path.append(MODEL_PATH)

g_sift_metrics = {
    # 'default': [None] * 100,
    # 'i_hpatches': [None] * 100,
    # 'v_hpatches': [None] * 100,
}

SAVE_MODEL = True

def eval_one_epoch(sess, ops, ev_params, name='default'):
    global g_sift_metrics
    dataset_size = ev_params['dataset_size']
    num_photos_per_seq = ev_params['num_photos_per_seq']
    num_seq = len(num_photos_per_seq)
    summary_writer = ev_params['summary_writer']
    best_saver = ev_params['best_saver']
    log_dir = ev_params['log_dir']
    best_score = ev_params['best_score']
    best_score_filename = ev_params['best_score_filename']
    num_kp = ev_params['num_kp']

    assert dataset_size == num_photos_per_seq.sum()

    sess.run(ev_params['ev_init_op'])

    NUM_COMMON_MET = 2 # loss, loss_det
    NUM_DET_MET = 1
    NUM_DESC_MET = 5
    NUM_SIFT_MET = 2

    mA_common_metrics = np.zeros(NUM_COMMON_MET, dtype=np.float32) # mean average
    mA_det_metrics = np.zeros(NUM_DET_MET, dtype=np.float32) # det_loss, pred_num_kp
    mA_desc_metrics = np.zeros(NUM_DESC_MET, dtype=np.float32)
    mA_sift_metrics = np.zeros(NUM_SIFT_MET, dtype=np.float32)

    num_canvas = 5
    canvas_indices = list(np.linspace(10, dataset_size-10, num_canvas, dtype=np.int32)) # not include some of begining and end of frames
    # print(72 in canvas_indices)
    ours_canvas_list = []
    ours_canvas_list2 = []
    sift_canvas_list = []
    curr_idx = 0

    for n in range(num_seq):
        seq_length = num_photos_per_seq[n]
        _common_metrics = np.zeros(NUM_COMMON_MET, dtype=np.float32)
        _det_metrics = np.zeros(NUM_DET_MET, dtype=np.float32)
        _desc_metrics = np.zeros(NUM_DESC_MET, dtype=np.float32)
        _sift_metrics = np.zeros(NUM_SIFT_MET, dtype=np.float32)

        for i in range(seq_length-1):
            #------------------------
            # Evaluate Ours
            #------------------------
            feed_dict = {
                ops['is_training']: False,
                ops['handle']: ev_params['handle'],
            }

            fetch_dict = {
                'loss': ops['loss'],
                'loss_det': ops['loss_det'],
                'det_loss': ops['det_loss'],
                'desc_loss': ops['desc_loss'],
                'dist_pos': ops['desc_dist_pos'],
                'dist_neg': ops['desc_dist_neg'],
                'photos1': ops['photos1'],
                'photos2': ops['photos2'],
                'kpts1': ops['desc_kpts1'],
                'kpts2': ops['desc_kpts2'],
                'kpts2_corr': ops['eval_kpts2_corr'],
                'is_match': ops['eval_is_match'],
                'match_score': ops['eval_match_score'],
                'match_dist': ops['eval_match_dist'],
                'xy_maps1to2': ops['desc_xy_maps1to2'],
                'visible_masks1': ops['desc_visible_masks1'],

            }

            ours_outs = sess.run(fetch_dict, feed_dict=feed_dict)

            _common_metrics += np.array([ours_outs['loss'], ours_outs['loss_det']])
            _det_metrics += np.array([ours_outs['det_loss']])
            _desc_metrics += np.array([ours_outs['desc_loss'], ours_outs['dist_pos'], ours_outs['dist_neg'], 
                                       ours_outs['match_score'], ours_outs['match_dist']])
            
            im1 = ours_outs['photos1'][0]
            im2 = ours_outs['photos2'][0]
            if curr_idx in canvas_indices:
                # visualize matching results

                assert len(ours_outs['photos1']) == 1 # make sure batch_size == 1
                match_image = draw_match(im1, im2, ours_outs['kpts1'], ours_outs['kpts2_corr'], 
                                        ours_outs['is_match'], ours_outs['match_score'])
                ours_canvas_list.append(match_image)

            #------------------------
            # Evaluate SIFT
            #------------------------
            if g_sift_metrics[name][n] is None:
                im1_uint8 = np.squeeze(np.clip(im1*255,0,255).astype(np.uint8))
                im2_uint8 = np.squeeze(np.clip(im2*255,0,255).astype(np.uint8))

                kpts1, feats1 = compute_sift_multi_scale(im1_uint8, num_kp=num_kp)
                kpts2, feats2 = compute_sift_multi_scale(im2_uint8, num_kp=num_kp)

                feed_dict = {
                    ops['sift_kpts1_ph']: kpts1,
                    ops['sift_kpts2_ph']: kpts2,
                    ops['sift_feats1_ph']: feats1,
                    ops['sift_feats2_ph']: feats2,
                    ops['sift_xy_maps1to2_ph']: ours_outs['xy_maps1to2'],
                    ops['sift_visible_masks1_ph']: ours_outs['visible_masks1'],
                }
                fetch_dict = {
                    'kpts2_corr': ops['sift_kpts2_corr'],
                    'is_match': ops['sift_is_match'],
                    'match_score': ops['sift_match_score'],
                    'match_dist': ops['sift_match_dist'],
                }
                sift_outs = sess.run(fetch_dict, feed_dict=feed_dict)
                _sift_metrics += np.array([sift_outs['match_score'], sift_outs['match_dist']])
                if curr_idx in canvas_indices:
                    match_image = draw_match(im1, im2, kpts1, sift_outs['kpts2_corr'], 
                                            sift_outs['is_match'], sift_outs['match_score'])
                    sift_canvas_list.append(match_image)

            # go to next
            curr_idx += 1
        #------- END OF SEQUENCE
        _common_metrics /= seq_length
        _det_metrics /= seq_length
        _desc_metrics /= seq_length
        _sift_metrics /= seq_length
        # print('SEQ ', _sift_metrics)
        if g_sift_metrics[name][n] is None:
            g_sift_metrics[name][n] = _sift_metrics
            print('#{}-{} Finish computing SIFT evaluation metrics...'.format(name, n))
        else:
            _sift_metrics = g_sift_metrics[name][n]

        # Update metrics
        mA_common_metrics += _common_metrics
        mA_det_metrics += _det_metrics
        mA_desc_metrics += _desc_metrics
        mA_sift_metrics += _sift_metrics

    #---------- END ALL SAMPLES
    step = sess.run(ops['step'])

    mA_common_metrics /= num_seq
    mA_det_metrics /= num_seq
    mA_desc_metrics /= num_seq
    mA_sift_metrics /= num_seq

    loss, loss_det = mA_common_metrics
    det_loss, = mA_det_metrics
    desc_loss, dist_pos, dist_neg, ours_match_score, ours_match_dist = mA_desc_metrics
    sift_match_score, sift_match_dist = mA_sift_metrics

    print('')
    print('[{}] iter={} Loss: {:.3f}({:.3f}|{:.3f}) Match(C|S)={:.3f}|{:3f} Dist(C|S)={:.3f}|{:.3f}'.format(
                name, step, 
                loss, det_loss, desc_loss,
                ours_match_score, sift_match_score,
                ours_match_dist, sift_match_dist
                ))

    tag_list = ['loss',
                'ours_match_score', 'ours_match_dist',
                'sift_match_score', 'sift_match_dist',
                ]
    prefix = '' if name == 'default' else name+'-'
    summaries = []
    for _tag in tag_list:
        summaries.append( tf.Summary.Value(tag=prefix+_tag, simple_value=eval(_tag)) )
    summary_writer.add_summary(tf.Summary(value=summaries), global_step=step)

    ours_canvas_list = np.clip(np.array(ours_canvas_list), 0, 1)
    log_images(summary_writer, prefix+'ours_match_results', ours_canvas_list, step)

    if len(sift_canvas_list) > 0:
        sift_canvas_list = np.clip(np.array(sift_canvas_list), 0, 1)
        log_images(summary_writer, prefix+'sift_match_results', sift_canvas_list, step)

    return ours_match_score

    # if SAVE_MODEL and loss < best_score and best_saver is not None:
    #     ev_params['best_score'] = best_score = loss
    #     print("Saving best model with valid-score = {}".format(best_score))
    #     best_saver.save(sess, os.path.join(log_dir, 'models-best'), write_meta_graph=False)
    #     with open(best_score_filename, 'w') as f:
    #         f.write('{} {:g}\n'.format(step, best_score))

def apply_scale_on_intrinsic(K, sx, sy):
    # K : [B,3,3]
    batch_size = tf.shape(K)[0]
    # 3x3 matrix
    S = tf.stack(
        [sx, 0, 0,
         0, sy, 0,
         0, 0, 1])
    S = tf.cast(tf.reshape(S, [3,3]), tf.float32)
    S = tf.tile(S[None], [batch_size, 1, 1])
    return tf.matmul(S, K)

def build_training_network(config, next_batch, is_training, psf, global_step):

    max_outputs = 5
    axis123 = list(range(1,4)) # 1,2,3
    photos1, photos2, depths1, depths2, valid_masks1, valid_masks2, c2Tc1s, c1Tc2s, c1Tws, c2Tws, Ks1, Ks2, thetas1, thetas2, inv_thetas1, inv_thetas2, theta_params = next_batch
    raw_photos1 = tf.identity(photos1)
    raw_photos2 = tf.identity(photos2)
    if config.input_inst_norm:
        print('Apply instance norm on input photos')
        photos1 = instance_normalization(photos1)
        photos2 = instance_normalization(photos2)

    batch_size = tf.shape(photos1)[0]
    crop_radius = config.crop_radius
    patch_size = config.patch_size
    mining_type = config.mining_type.lower()
    det_loss_type = config.det_loss.lower()
    desc_loss_type = config.desc_loss.lower()
    K = config.top_k

    # Show tensorboard
    c_red = tf.constant([1,0,0], dtype=tf.float32)
    c_green = tf.constant([0,1,0], dtype=tf.float32)
    c_blue = tf.constant([0,0,1], dtype=tf.float32)
    rgbs1 = tf.concat([raw_photos1, raw_photos1, raw_photos1], axis=-1)
    rgbs2 = tf.concat([raw_photos2, raw_photos2, raw_photos2], axis=-1)

    #----------------------------------
    #  Detector
    #----------------------------------
    DET = importlib.import_module(config.detector)
    detector = DET.Model(config, is_training)

    if config.use_nms3d:
        print('Apply 3D NMS instead.')
        heatmaps1, det_endpoints = build_multi_scale_deep_detector_3DNMS(config, detector, photos1, reuse=False)
        heatmaps2, det_endpoints2 = build_multi_scale_deep_detector_3DNMS(config, detector, photos2, reuse=True)
    else:
        heatmaps1, det_endpoints = build_multi_scale_deep_detector(config, detector, photos1, reuse=False)
        heatmaps2, det_endpoints2 = build_multi_scale_deep_detector(config, detector, photos2, reuse=True)

    for i, score_maps in enumerate(det_endpoints['score_maps_list']):
        tf.summary.image('logits1_{}'.format(i), score_maps, max_outputs=max_outputs)
    tf.summary.histogram('heatmaps1', heatmaps1)

    #------------------------------------------------------
    #  Score loss (warp heatmaps and take loss)
    #------------------------------------------------------

    # Heatmap transfer one another
    heatmaps1w, visible_masks1, xy_maps1to2 = \
        inverse_warp_view_2_to_1(heatmaps2, depths2, depths1, c2Tc1s, 
                                K1=Ks1, K2=Ks2, 
                                inv_thetas1=inv_thetas1, thetas2=thetas2,
                                depth_thresh=config.depth_thresh)
    heatmaps2w, visible_masks2, xy_maps2to1 = \
        inverse_warp_view_2_to_1(heatmaps1, depths1, depths2, c1Tc2s, 
                                K1=Ks2, K2=Ks1,
                                inv_thetas1=inv_thetas2, thetas2=thetas1,
                                depth_thresh=config.depth_thresh)
    visible_masks1 = visible_masks1 * valid_masks1 # take 'and'
    visible_masks2 = visible_masks2 * valid_masks2
    
    heatmaps1w.set_shape(heatmaps2.get_shape().as_list())
    heatmaps1w = tf.stop_gradient(heatmaps1w) # to be safe
    heatmaps2w.set_shape(heatmaps1.get_shape().as_list())
    heatmaps2w = tf.stop_gradient(heatmaps2w) # to be safe

    nms_maps1w = non_max_suppression(heatmaps1w, config.nms_thresh, config.nms_ksize)
    nms_maps2w = non_max_suppression(heatmaps2w, config.nms_thresh, config.nms_ksize)
    nms_score1w = heatmaps1w * nms_maps1w # not filter out with mask because this tensor are used to compare with heatmaps
    nms_score2w = heatmaps2w * nms_maps2w
    top_k1w = make_top_k_sparse_tensor(nms_score1w, k=K)
    top_k1w = top_k1w * nms_maps1w
    top_k1w = tf.stop_gradient(top_k1w)
    top_k2w = make_top_k_sparse_tensor(nms_score2w, k=K)
    top_k2w = top_k2w * nms_maps2w
    top_k2w = tf.stop_gradient(top_k2w)

    topk1_canvas = (1.0-det_endpoints['top_ks']) * rgbs1 + det_endpoints['top_ks'] * c_red
    topk2_canvas = (1.0-det_endpoints2['top_ks']) * rgbs2 + det_endpoints2['top_ks'] * c_green
    tf.summary.image('TOPK1-TOPK2', tf.concat([topk1_canvas, topk2_canvas], axis=2), max_outputs=max_outputs)

    tgt_heatmaps1 = heatmaps1
    tgt_heatmaps2 = heatmaps2

    ## regenerate GT-heatmaps otherwise DET outputs goes blur
    gt_heatmaps1 = tf.nn.conv2d(top_k1w, psf, [1,1,1,1], padding='SAME')
    gt_heatmaps1 = tf.minimum(gt_heatmaps1, 1.0)
    gt_heatmaps2 = tf.nn.conv2d(top_k2w, psf, [1,1,1,1], padding='SAME')
    gt_heatmaps2 = tf.minimum(gt_heatmaps2, 1.0)

    Nvis1 = tf.maximum(tf.reduce_sum(visible_masks1, axis=axis123), 1.0)
    Nvis2 = tf.maximum(tf.reduce_sum(visible_masks2, axis=axis123), 1.0)

    if det_loss_type == 'l2loss':
        l2diff1 = tf.squared_difference(tgt_heatmaps1, gt_heatmaps1)
        loss1 = tf.reduce_mean( tf.reduce_sum(l2diff1 * visible_masks1, axis=axis123) / Nvis1 ) 

        l2diff2 = tf.squared_difference(tgt_heatmaps2, gt_heatmaps2)
        loss2 = tf.reduce_mean( tf.reduce_sum(l2diff2 * visible_masks2, axis=axis123) / Nvis2 ) 

        det_loss = (loss1 + loss2) / 2.0
    else:
        raise ValueError('Unknown det_loss: {}'.format(det_loss_type))

    tf.summary.scalar('score_loss', det_loss)

    #------------------------------------------------------
    #  Orientation loss (warp orientation and take loss)
    #------------------------------------------------------
    aug_ori2 = theta_params[:, 3] if config.rot_aug else 0 # rot-aug are applied only on image2
    intheta_c2Rc1 = theta_params[:, 4]
    dori_1to2 = (intheta_c2Rc1 + aug_ori2)[:,None,None,None]

    ori_maps1 = det_endpoints['ori_maps']
    ori_maps2 = det_endpoints2['ori_maps']

    degree_maps1, atan_maps1 = get_degree_maps(ori_maps1)
    degree_maps2, atan_maps2 = get_degree_maps(ori_maps2)
    atan_maps2w = nearest_neighbor_sampling(atan_maps1+dori_1to2, xy_maps2to1) # warp from 1 to 2
    atan_maps1w = nearest_neighbor_sampling(atan_maps2-dori_1to2, xy_maps1to2) # warp from 2 to 1

    ori_maps2w = tf.concat([tf.cos(atan_maps2w), tf.sin(atan_maps2w)], axis=-1)
    ori_maps1w = tf.concat([tf.cos(atan_maps1w), tf.sin(atan_maps1w)], axis=-1)

    angle2rgb = tf.constant(get_angle_colorbar())
    degree_diff1 = tf.reduce_sum(ori_maps1 * ori_maps1w, axis=-1, keep_dims=True)
    degree_diff1 = tf.acos(degree_diff1) # radian
    degree_diff1 = tf.cast(tf.clip_by_value(degree_diff1*180/np.pi+180, 0, 360), tf.int32) 
    degree_diff1 = tf.gather(angle2rgb, degree_diff1[...,0])

    degree_diff2 = tf.reduce_sum(ori_maps2 * ori_maps2w, axis=-1, keep_dims=True)
    degree_diff2 = tf.acos(degree_diff2) # radian
    degree_diff2 = tf.cast(tf.clip_by_value(degree_diff2*180/np.pi+180, 0, 360), tf.int32) 
    degree_diff2 = tf.gather(angle2rgb, degree_diff2[...,0])

    degree_maps1w, _ = get_degree_maps(ori_maps1w)
    degree_maps2w, _ = get_degree_maps(ori_maps2w)

    degree_canvas = tf.concat([
            tf.concat([degree_maps1, degree_maps1w, degree_diff1], axis=2),
            tf.concat([degree_maps2, degree_maps2w, degree_diff2], axis=2),
        ], axis=1)
    tf.summary.image('degree_maps', degree_canvas, max_outputs=max_outputs)

    if config.ori_loss == 'l2loss':
        ori_loss1 = tf.squared_difference(ori_maps1, ori_maps1w)
        ori_loss1 = tf.reduce_mean( tf.reduce_sum(ori_loss1 * visible_masks1, axis=axis123) / Nvis1 ) 
        ori_loss2 = tf.squared_difference(ori_maps2, ori_maps2w)
        ori_loss2 = tf.reduce_mean( tf.reduce_sum(ori_loss2 * visible_masks2, axis=axis123) / Nvis2 ) 
        ori_loss = (ori_loss1 + ori_loss2) * 0.5
    elif config.ori_loss == 'cosine':
        ori_loss1 = tf.reduce_sum(ori_maps1 * ori_maps1w, axis=-1, keep_dims=True) # both ori_maps have already normalized
        ori_loss1 = tf.reduce_mean( tf.reduce_sum(tf.square(1.0-ori_loss1) * visible_masks1, axis=axis123) / Nvis1)
        ori_loss2 = tf.reduce_mean(ori_maps2 * ori_maps2w, axis=-1, keep_dims=True)
        ori_loss2 = tf.reduce_mean( tf.reduce_sum(tf.square(1.0-ori_loss2) * visible_masks2, axis=axis123) / Nvis2)
        ori_loss = (ori_loss1 + ori_loss2) * 0.5
    else:
        raise ValueError('Unknown ori_loss: {}'.format(config.ori_loss))

    tf.summary.scalar('ori_loss_{}'.format(config.ori_loss), ori_loss)

    #------------------------------------------------------
    #  Scale loss (warp orientation and take loss)
    #------------------------------------------------------
    fx1 = tf.reshape(tf.slice(Ks1, [0,0,0], [-1,1,1]), [-1]) # assume fx == fy
    fx2 = tf.reshape(tf.slice(Ks2, [0,0,0], [-1,1,1]), [-1])
    ones = tf.ones_like(depths1)
    aug_scale2 = tf.exp(theta_params[:,1]) if config.scale_aug else 1.0
    scale_maps1 = det_endpoints['scale_maps'][...,None] # [B,H,W,1]
    scale_maps2 = det_endpoints2['scale_maps'][...,None]
    depths1w = nearest_neighbor_sampling(depths2, xy_maps1to2)
    depths1w = tf.where(tf.greater(depths1w, 500), ones, depths1w) # invalid depths are suppressed by 1
    depths2w = nearest_neighbor_sampling(depths1, xy_maps2to1)
    depths2w = tf.where(tf.greater(depths2w, 500), ones, depths2w) 
    scale_maps2w = scale_maps1 * tf.reshape(fx2/fx1*aug_scale2, [-1,1,1,1]) * depths1 / (depths1w+1e-6)
    scale_maps2w = nearest_neighbor_sampling(scale_maps2w, xy_maps2to1)
    scale_maps2w = tf.clip_by_value(scale_maps2w, config.net_min_scale, config.net_max_scale)
    scale_maps2w = tf.stop_gradient(scale_maps2w)
    scale_maps1w = scale_maps2 * tf.reshape(fx1/fx2/aug_scale2, [-1,1,1,1]) * depths2 / (depths2w+1e-6)
    scale_maps1w = nearest_neighbor_sampling(scale_maps1w, xy_maps1to2)
    scale_maps1w = tf.clip_by_value(scale_maps1w, config.net_min_scale, config.net_max_scale)
    scale_maps1w = tf.stop_gradient(scale_maps1w)

    # logscale L2 loss
    scale_loss1 = tf.squared_difference(tf.log(scale_maps1), tf.log(scale_maps1w))
    max_scale_loss1 = tf.reduce_max(scale_loss1)
    scale_loss1 = tf.reduce_mean(tf.reduce_sum(scale_loss1 * visible_masks1, axis=axis123) / Nvis1)
    scale_loss2 = tf.squared_difference(tf.log(scale_maps2), tf.log(scale_maps2w))
    max_scale_loss2 = tf.reduce_max(scale_loss2)
    scale_loss2 = tf.reduce_mean(tf.reduce_sum(scale_loss2 * visible_masks2, axis=axis123) / Nvis2)
    scale_loss = (scale_loss1 + scale_loss2) * 0.5
    tf.summary.scalar('scale_loss', scale_loss)
    det_endpoints['scale_loss'] = scale_loss

    scale_canvas = tf.concat([det_endpoints['scale_maps'], det_endpoints2['scale_maps']], axis=2)[...,None]
    tf.summary.image('Scalemaps1-2', scale_canvas, max_outputs=max_outputs)

    #----------------------------------
    #  Extract patches
    #----------------------------------
    kpts1 = det_endpoints['kpts']
    kpts2 = det_endpoints2['kpts']
    kpts1_int = tf.cast(kpts1, tf.int32)
    kpts2_int = tf.cast(kpts2, tf.int32)
    kpts_scale1 = det_endpoints['kpts_scale']
    kpts_scale2 = det_endpoints2['kpts_scale']
    kpts_ori1 = det_endpoints['kpts_ori']
    kpts_ori2 = det_endpoints2['kpts_ori']

    num_kpts1 = det_endpoints['num_kpts']    
    batch_inds1 = det_endpoints['batch_inds']

    kpts2w = batch_gather_keypoints(xy_maps1to2, batch_inds1, kpts1_int)
    kpts2w_int = tf.cast(kpts2w, tf.int32)
    kpvis2w = batch_gather_keypoints(visible_masks1, batch_inds1, kpts1_int)[:,0] # or visible_masks2, batch_inds2, kpts2w


    kpts_scale2w = batch_gather_keypoints(det_endpoints2['scale_maps'], batch_inds1, kpts2w_int)
    kpts_ori2w = batch_gather_keypoints(ori_maps2, batch_inds1, kpts2w_int)

    # visualization of orientation
    cos_maps1 = tf.slice(ori_maps1, [0,0,0,0], [-1,-1,-1,1])
    sin_maps1 = tf.slice(ori_maps1, [0,0,0,1], [-1,-1,-1,1])
    atan_maps1 = tf.atan2(sin_maps1, cos_maps1)
    cos_maps2 = tf.slice(ori_maps2, [0,0,0,0], [-1,-1,-1,1])
    sin_maps2 = tf.slice(ori_maps2, [0,0,0,1], [-1,-1,-1,1])
    atan_maps2 = tf.atan2(sin_maps2, cos_maps2)
    angle2rgb = tf.constant(get_angle_colorbar())
    degree_maps1 = tf.cast(tf.clip_by_value(atan_maps1*180/np.pi+180, 0, 360), tf.int32) 
    degree_maps1 = tf.gather(angle2rgb, degree_maps1[...,0])
    degree_maps2 = tf.cast(tf.clip_by_value(atan_maps2*180/np.pi+180, 0, 360), tf.int32) 
    degree_maps2 = tf.gather(angle2rgb, degree_maps2[...,0])
    degree_maps = tf.concat([degree_maps1, degree_maps2], axis=2)
    tf.summary.image('ori_maps_degree', degree_maps, max_outputs=max_outputs)

    # extract patches
    kp_patches1 = build_patch_extraction(config, det_endpoints, photos1)
    kp_patches2 = build_patch_extraction(config, det_endpoints2, photos2)

    det_endpoints2w = {
        'batch_inds': batch_inds1,
        'kpts': kpts2w,
        'kpts_scale': kpts_scale2w,
        'kpts_ori': kpts_ori2w,
        'feat_maps': det_endpoints2['feat_maps'],
    }
    kp_patches1_pos = build_patch_extraction(config, det_endpoints2w, photos2) # positive pair of kp1

    # Add supervision for orientation
    kpts_ori2w_gt = batch_gather_keypoints(ori_maps2w, batch_inds1, kpts2w_int)
    
    # Visualize patches
    det_endpoints2w_gt = {
        'batch_inds': batch_inds1,
        'kpts': kpts2w,
        'kpts_scale': kpts_scale2w,
        'kpts_ori': kpts_ori2w_gt,
        'feat_maps': det_endpoints2['feat_maps'],
    }
    kp_patches1_pos_gt = build_patch_extraction(config, det_endpoints2w_gt, photos2) # positive pair of kp1

    patches1_canvas = tf.reduce_max(kp_patches1, axis=-1, keep_dims=True) # need channel compression in case feat_maps are not photos
    patches1_pos_canvas = tf.reduce_max(kp_patches1_pos, axis=-1, keep_dims=True)
    patches1_pos_gt_canvas = tf.reduce_max(kp_patches1_pos_gt, axis=-1, keep_dims=True)
    app_patches = tf.concat([patches1_canvas, patches1_pos_canvas * kpvis2w[:,None,None,None], patches1_pos_gt_canvas * kpvis2w[:,None,None,None]], axis=2) # anchor, positive, negative
    app_patches = tf.random_shuffle(app_patches)
    app_patches = convert_tile_image(app_patches[:64])
    app_patches = tf.clip_by_value(app_patches, 0, 1)
    tf.summary.image('GT_app_patches', app_patches, max_outputs=1)

    #----------------------------------
    #  Descriptor
    #----------------------------------
    DESC = importlib.import_module(config.descriptor)
    descriptor = DESC.Model(config, is_training)

    desc_feats1, desc_endpoints = build_deep_descriptor(config, descriptor, kp_patches1, reuse=False) # [B*K,D]
    desc_feats2, _              = build_deep_descriptor(config, descriptor, kp_patches2, reuse=True)
    desc_feats1_pos, _             = build_deep_descriptor(config, descriptor, kp_patches1_pos, reuse=True)

    tf.summary.histogram('desc_feats1', desc_feats1)

    ## Negative samples selection
    if mining_type == 'hard':
        _, neg_inds = find_hard_negative_from_myself_less_memory(desc_feats1, batch_inds1, num_kpts1, batch_size)
        desc_feats1_neg = tf.gather(desc_feats1, neg_inds)
        kp_patches1_neg = tf.gather(kp_patches1, neg_inds)
    elif mining_type == 'random':
        neg_inds = find_random_negative_from_myself_less_memory(desc_feats1, batch_inds1, num_kpts1, batch_size)
        desc_feats1_neg = tf.gather(desc_feats1, neg_inds)
        kp_patches1_neg = tf.gather(kp_patches1, neg_inds)
    elif mining_type == 'hard2':
        print('Mine hardest negative sample from image2')
        print('[WARNING] find_hard_negative_from_myself_less_memory has bug. it try to search the closest samples from feat2 but it should search from feat1')
        _, neg_inds = find_hard_negative_from_myself_less_memory(desc_feats1_pos, batch_inds1, num_kpts1, batch_size)
        desc_feats1_neg = tf.gather(desc_feats1_pos, neg_inds)
        kp_patches1_neg = tf.gather(kp_patches1_pos, neg_inds)
    elif mining_type == 'hard2geom':
        # too difficult to train because negative is more similar to anchor than positive 
        # geom_sq_thresh = config.hard_geom_thresh ** 2
        # print('Mine hardest negative sample from image2 and geometric constrain (thresh={}, square={})'.format(config.hard_geom_thresh, geom_sq_thresh))
        # _, neg_inds = find_hard_negative_from_myself_with_geom_constrain_less_memory(
        #                 desc_feats1, desc_feats1_pos, kpts2w, batch_inds1, num_kpts1, batch_size, geom_sq_thresh)
        # desc_feats1_neg = tf.gather(desc_feats1_pos, neg_inds)
        # kp_patches1_neg = tf.gather(kp_patches1_pos, neg_inds)
        geom_sq_thresh = config.hard_geom_thresh ** 2
        print('Mine hardest negative sample from image2 and geometric constrain (thresh={}, square={})'.format(config.hard_geom_thresh, geom_sq_thresh))
        _, neg_inds = imperfect_find_hard_negative_from_myself_with_geom_constrain_less_memory(
                        desc_feats1_pos, kpts2w, batch_inds1, num_kpts1, batch_size, geom_sq_thresh)
        desc_feats1_neg = tf.gather(desc_feats1_pos, neg_inds)
        kp_patches1_neg = tf.gather(kp_patches1_pos, neg_inds)
    elif mining_type == 'random2':
        print('Mine random negative sample from image2')
        neg_inds = find_random_negative_from_myself_less_memory(desc_feats1_pos, batch_inds1, num_kpts1, batch_size)
        desc_feats1_neg = tf.gather(desc_feats1, neg_inds)
        kp_patches1_neg = tf.gather(kp_patches1_pos, neg_inds)
    elif mining_type == 'rand_hard':
        num_pickup = config.init_num_mine # e.g. 512 // 10
        print('Random Hard Mining #pickup={}'.format(num_pickup))
        geom_sq_thresh = config.hard_geom_thresh ** 2
        neg_inds = find_random_hard_negative_from_myself_with_geom_constrain_less_memory(
                        num_pickup, desc_feats1, desc_feats1_pos, kpts2w, batch_inds1, num_kpts1, batch_size, geom_sq_thresh)
        desc_feats1_neg = tf.gather(desc_feats1_pos, neg_inds)
        kp_patches1_neg = tf.gather(kp_patches1_pos, neg_inds)
    elif mining_type == 'rand_hard_sch':
        print('Random Hard Mining with scheduling #pickup={}-->{} (decay={})'.format(config.init_num_mine, config.min_num_pickup, config.pickup_delay))
        num_pickup = tf.maximum(tf.cast(tf.train.exponential_decay(float(config.init_num_mine), global_step, 1000, config.pickup_delay), tf.int32), config.min_num_pickup) # stop decay @ num_pickup=1
        tf.summary.scalar('num_negative_mining', num_pickup)
        geom_sq_thresh = config.hard_geom_thresh ** 2
        neg_inds = find_random_hard_negative_from_myself_with_geom_constrain_less_memory(
                        num_pickup, desc_feats1, desc_feats1_pos, kpts2w, batch_inds1, num_kpts1, batch_size, geom_sq_thresh)
        desc_feats1_neg = tf.gather(desc_feats1_pos, neg_inds)
        kp_patches1_neg = tf.gather(kp_patches1_pos, neg_inds)
    else:
        raise ValueError('Unknown mining_type: {}'.format(mining_type))

    if desc_loss_type == 'triplet':
        desc_margin = config.desc_margin
        d_pos = tf.reduce_sum(tf.square(desc_feats1-desc_feats1_pos), axis=1) # [B*K,]
        d_neg = tf.reduce_sum(tf.square(desc_feats1-desc_feats1_neg), axis=1) # [B*K,]

        d_pos = kpvis2w * d_pos # ignore unvisible anchor-positve pairs

        desc_loss = tf.reduce_mean(tf.maximum(0., desc_margin+d_pos-d_neg))
        desc_pair_loss = tf.reduce_mean(d_pos)
        desc_dist_pos = tf.reduce_mean(tf.sqrt(d_pos + 1e-10), name='pos-dist')
        desc_dist_neg = tf.reduce_mean(tf.sqrt(d_neg + 1e-10), name='neg-dist')
        tf.summary.scalar('desc_triplet_loss', desc_loss)
        tf.summary.scalar('desc_pair_loss', desc_pair_loss)
        tf.summary.scalar('dist_pos', desc_dist_pos)
        tf.summary.scalar('dist_neg', desc_dist_neg)
    else:
        raise ValueError('Unknown desc_loss: {}'.format(desc_loss_type))

    patches1_canvas = tf.reduce_max(kp_patches1, axis=-1, keep_dims=True) # need channel compression in case feat_maps are not photos
    patches1_pos_canvas = tf.reduce_max(kp_patches1_pos, axis=-1, keep_dims=True)
    patches1_neg_canvas = tf.reduce_max(kp_patches1_neg, axis=-1, keep_dims=True)
    apn_patches = tf.concat([patches1_canvas, patches1_pos_canvas * kpvis2w[:,None,None,None], patches1_neg_canvas], axis=2) # anchor, positive, negative
    apn_patches = tf.random_shuffle(apn_patches)
    apn_patches = convert_tile_image(apn_patches[:64])
    apn_patches = tf.clip_by_value(apn_patches, 0, 1)
    tf.summary.image('apn_patches', apn_patches, max_outputs=1)

    desc_endpoints['loss'] = desc_loss
    desc_endpoints['feats1'] = desc_feats1
    desc_endpoints['feats2'] = desc_feats2
    desc_endpoints['dist_pos'] = desc_dist_pos
    desc_endpoints['dist_neg'] = desc_dist_neg
    desc_endpoints['kpts1'] = kpts1
    desc_endpoints['kpts2'] = kpts2
    desc_endpoints['kpts2w'] = kpts2w
    desc_endpoints['kpts_scale1'] = kpts_scale1
    desc_endpoints['kpts_scale2'] = kpts_scale2
    desc_endpoints['kpts_scale2w'] = kpts_scale2w
    desc_endpoints['kpts_ori1'] = kpts_ori1
    desc_endpoints['kpts_ori2'] = kpts_ori2
    desc_endpoints['kpts_ori2w'] = kpts_ori2w
    desc_endpoints['kpvis2w'] = kpvis2w

    desc_endpoints['xy_maps1to2'] = xy_maps1to2
    desc_endpoints['visible_masks1'] = visible_masks1
    desc_endpoints['apn_patches'] = apn_patches
    desc_endpoints['neg_inds'] = neg_inds

    #----------------------------------
    #  Training Loss
    #----------------------------------
    final_det_loss = det_loss + config.weight_det_loss * desc_pair_loss + config.ori_weight * ori_loss + config.scale_weight * scale_loss
    final_desc_loss = desc_loss
    tf.summary.scalar('final_det_loss', final_det_loss)
    tf.summary.scalar('final_desc_loss', final_desc_loss)

    det_endpoints['loss'] = final_det_loss

    #----------------------------------
    #  Evaluation of Descriptor (make sure the following code only works if batch_size=1)
    #----------------------------------

    eval_endpoints = build_matching_estimation(config, desc_feats1, desc_feats2, 
                                                        kpts1, kpts2,
                                                        kpts2w, kpvis2w, dist_thresh=config.match_reproj_thresh)
    sift_endpoints = build_competitor_matching_estimation(config, dist_thresh=config.match_reproj_thresh)

    return final_det_loss, final_desc_loss, det_endpoints, desc_endpoints, eval_endpoints, sift_endpoints
    # return loss, loss_det, det_endpoints, desc_endpoints, eval_endpoints, sift_endpoints

def main(config):
    tf.reset_default_graph() # for sure
    set_summary_visibility(variables=False, gradients=False)

    log_dir = config.log_dir
    batch_size = config.batch_size
    optim_method = config.optim_method
    learning_rate = config.lr
    va_batch_size = 1

    if config.dataset == 'sfm':
        render_paths = config.sfm_seq.split(',')
        print('[SfM-SPLIT] Setup sfm ({}-seqs)'.format(len(render_paths)))
        tr_loader = SfMDataset(out_size=(config.data_raw_size, config.data_raw_size), 
                               warp_aug_mode='random', flip_pair=True, max_degree=config.aug_max_degree, max_scale=config.aug_max_scale,
                               num_threads=config.num_threads)
        tr_dataset = tr_loader.get_dataset(config.sfm_dpt_dir, config.sfm_img_dir, 
                            render_paths, phase='train',
                            batch_size=batch_size, shuffle=True)
        va_loader = SfMDataset(out_size=(config.data_raw_size, config.data_raw_size), 
                               warp_aug_mode='none', flip_pair=False,
                               num_threads=config.num_threads)
        va_dataset_list = [va_loader.get_dataset(config.sfm_dpt_dir, config.sfm_img_dir, 
                            render_paths, phase='valid',
                            batch_size=va_batch_size, shuffle=False, max_examples=2000)]
        va_attributes = [{'name': config.dataset,
                          'num_photos_per_seq': va_loader.num_photos_per_seq_data,
                          'total_num_photos': va_loader.total_num_photos,
                        }]
    else:
        raise ValueError('Unknown dataset: {}'.format(config.dataset))

    config.depth_thresh = tr_loader.depth_thresh
    print('Reset depth_thresh: {}, it may be better to use placeholder'.format(config.depth_thresh))

    # use feedable iterator to switch training / validation dataset without unnecessary initialization
    handle = tf.placeholder(tf.string, shape=[])
    dataset_iter = tf.data.Iterator.from_string_handle(handle, tr_dataset.output_types, tr_dataset.output_shapes) # create mock of iterator
    next_batch = list(dataset_iter.get_next()) #tuple --> list to make it possible to modify each elements

    # intrinsics_3x3 = tr_loader.intrinsics_3x3

    tr_iter = tr_dataset.make_one_shot_iterator() # infinite loop
    va_iter_list = [va.make_initializable_iterator() for va in va_dataset_list]

    is_training_ph = tf.placeholder(tf.bool, shape=(), name='is_training')

    psf = tf.constant(get_gauss_filter_weight(config.hm_ksize, config.hm_sigma)[:,:,None,None], dtype=tf.float32) 
    global_step = tf.Variable(0, name='global_step', trainable=False)
    global_step2 = tf.Variable(0, name='global_step2', trainable=False)

    # Euclidean transformation data augmentation
    next_batch = euclidean_augmentation(next_batch, (config.data_size, config.data_size), config.rot_aug, config.scale_aug)

    det_loss, desc_loss, det_endpoints, desc_endpoints, eval_endpoints, sift_endpoints = \
                        build_training_network(config, next_batch, is_training_ph, psf, global_step)
    # var_list = det_endpoints['var_list'] + desc_endpoints['var_list']
    det_var_list = det_endpoints['var_list'] + det_endpoints['mso_var_list']
    desc_var_list = desc_endpoints['var_list']

    if config.lr_decay:
        boundaries = [5000, 15000, 30000, 50000]
        lr_levels = [0.1**i for i in range(len(boundaries))]
        lr_values = [learning_rate * decay for decay in lr_levels]
        learning_rate = get_piecewise_lr(global_step, boundaries, lr_values, show_summary=True)
        print('Enable adaptive learning. LR will decrease {} when #iter={}'.format(lr_values, boundaries))        

    # We should rename desc_minimize_op as desc_minimizer and so on.
    # descriptor minimizer
    desc_minimize_op = get_optimizer(optim_method, global_step, learning_rate, desc_loss, desc_var_list, show_var_and_grad=config.show_histogram)
    # detector minimizer
    det_minimize_op = get_optimizer(optim_method, global_step, learning_rate, det_loss, det_var_list, show_var_and_grad=config.show_histogram)

    print('Done.')

    # Create a session
    print('Create & Initialize session...')

    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True # almost the same as tf.InteractiveSession
    sess = tf.Session(config=tfconfig)

    summary = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    tr_handle = sess.run(tr_iter.string_handle())
    va_handle_list = sess.run([va.string_handle() for va in va_iter_list])

    if config.clear_logs and tf.gfile.Exists(log_dir):
        print('Clear all files in {}'.format(log_dir))
        try:
            tf.gfile.DeleteRecursively(log_dir) 
        except:
            print('Fail to delete {}. You probably have to kill tensorboard process.'.format(log_dir))

    # load pretrained model
    if len(config.pretrain_dir) > 0:
        if os.path.isdir(config.pretrain_dir):
            checkpoint = tf.train.latest_checkpoint(config.pretrain_dir)
        else:
            checkpoint = None
        if checkpoint is not None:
            global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            pretrained_vars = []
            for var in global_vars:
                if 'global_step' in var.name:
                    pass
                else:
                    pretrained_vars.append(var)
            print('Resume pretrained detector...')
            for i, var in enumerate(pretrained_vars):
                print('#{} {} [{}]'.format(i, var.name, var.shape))
            saver = tf.train.Saver(pretrained_vars)
            saver.restore(sess, checkpoint)
            saver = None
            print('Load pretrained model from {}'.format(checkpoint))
        else:
            raise ValueError('Cannot open checkpoint: {}'.format(checkpoint))

    best_saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
    # latest_saver = tf.train.Saver(max_to_keep=1, save_relative_paths=True)
    latest_saver = tf.train.Saver(max_to_keep=100, save_relative_paths=True) # save everything
 
    latest_checkpoint = tf.train.latest_checkpoint(log_dir)
    best_score_filename = os.path.join(log_dir, 'valid', 'best_score.txt')
    best_score = 0 # larger is better
    curr_epoch = 0
    if latest_checkpoint is not None:
        from parse import parse
        print('Resume the previous model...')
        latest_saver.restore(sess, latest_checkpoint)
        curr_step = sess.run(global_step)
        curr_epoch = curr_step // (tr_loader.total_num_photos // batch_size)
        print('Current step={}, epoch={}'.format(curr_step, curr_epoch))
        if os.path.exists(best_score_filename):
            with open(best_score_filename, 'r') as f:
                dump_res = f.read()
            dump_res = parse('{step:d} {best_score:g}\n', dump_res)
            best_score = dump_res['best_score']
            print('Previous best score = {}'.format(best_score))

    train_writer = tf.summary.FileWriter(
        os.path.join(log_dir, 'train'), graph=sess.graph
    )
    valid_writer = tf.summary.FileWriter(
        os.path.join(log_dir, 'valid'), graph=sess.graph
    )    

    if SAVE_MODEL:
        latest_saver.export_meta_graph(os.path.join(log_dir, "models.meta"))
    # Save config
    with open(os.path.join(log_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)    

    ops = {
        'is_training': is_training_ph,
        'handle': handle,
        'photos1': next_batch[0],
        'photos2': next_batch[1],
        'depths1': next_batch[2],
        'depths2': next_batch[3],
        'valid_masks1': next_batch[4],
        'valid_masks2': next_batch[5],
        'c2Tc1s': next_batch[6],
        'c1Tc2s': next_batch[7],
        'c1Tws': next_batch[8],
        'c2Tws': next_batch[9],
        'Ks1': next_batch[10],
        'Ks2': next_batch[11],
        'loss': desc_loss,
        'loss_det': det_loss,
        'step': global_step,
        'desc_minimize_op': desc_minimize_op,
        'det_minimize_op': det_minimize_op,
        'global_step': global_step,
        'summary': summary,
    }
    for k, v in det_endpoints.items():
        ops['det_'+k] = v
    for k, v in desc_endpoints.items():
        ops['desc_'+k] = v
    for k, v in eval_endpoints.items():
        ops['eval_'+k] = v
    for k, v in sift_endpoints.items():
        ops['sift_'+k] = v

    #----------------------
    # Start Training
    #----------------------

    num_itr_in_epoch = tr_loader.total_num_photos // batch_size
    save_summary_interval = 200
    save_model_interval = 2000
    valid_interval = 1000

    va_params = {
        'batch_size': va_batch_size,
        'log_dir': log_dir,
        'summary_writer': valid_writer,
        'num_kp': config.top_k,
        'best_score': best_score,
        'best_score_filename': best_score_filename,
        'num_photos_per_seq': None,
        'dataset_size': None,
        'handle': None,
        'ev_init_op': None,
        'best_saver': None,
    }

    # init g_sift_metrics
    global g_sift_metrics
    for attr in va_attributes:
        g_sift_metrics[attr['name']] = [None] * 100

    print('Start training.... (1epoch={}itr #size={})'.format(num_itr_in_epoch, tr_loader.total_num_photos))

    def check_counter(counter, interval):
        return (interval > 0 and counter % interval == 0)

    start_itr = sess.run(ops['global_step'])

    for _ in range(start_itr, config.max_itr):

        feed_dict = {
            ops['is_training']: True,
            ops['handle']: tr_handle,
        }


        if config.train_same_time:
            step, _, _ = sess.run([ops['step'], ops['desc_minimize_op'], ops['det_minimize_op']], feed_dict=feed_dict)
        else:
            step, _,  = sess.run([ops['step'], ops['desc_minimize_op']], feed_dict=feed_dict)
            _ = sess.run(ops['det_minimize_op'], feed_dict=feed_dict)

        if check_counter(step, save_summary_interval):
            feed_dict = {
                ops['is_training']: False,
                ops['handle']: tr_handle,
            }
            fetch_dict = {
                'loss': ops['loss'],
                'loss_det': ops['loss_det'],
                'det_loss': ops['det_loss'],
                'desc_loss': ops['desc_loss'],
                'summary': ops['summary'],
                'scale_maps': ops['det_scale_maps'],
            }
            start_time = time.time()
            outputs = sess.run(fetch_dict, feed_dict=feed_dict)
            elapsed_time = time.time() - start_time
            train_writer.add_summary(outputs['summary'], step) # save summary
            # scale_hist = np.histogram(outputs['scale_maps'], bins=config.net_num_scales, range=[config.net_min_scale, config.net_max_scale])
            # print(scale_hist)
            # print(outputs['scale_maps'].min(), outputs['scale_maps'].max())

            summaries = [tf.Summary.Value(tag='sec/step', simple_value=elapsed_time)]
            train_writer.add_summary(tf.Summary(value=summaries), global_step=step)
            train_writer.flush()

            print('[Train] {}step Loss(desc|det): {:g}|{:g} ({:.3f}|{:.3f}) ({:.1f}sec)'.format(
                        step,
                        outputs['loss'], outputs['loss_det'],
                        outputs['det_loss'], outputs['desc_loss'],
                        elapsed_time))
        if check_counter(step, save_model_interval):
            if SAVE_MODEL and latest_saver is not None:
                print('#{}step Save latest model'.format(step))
                latest_saver.save(sess, os.path.join(log_dir, 'models-latest'), global_step=step, write_meta_graph=False)

        if check_counter(step, valid_interval):
            va_mean_match_score = 0
            num_valid_set = 0
            for i, va_dataset in enumerate(va_dataset_list):
                va_params['num_photos_per_seq'] = va_attributes[i]['num_photos_per_seq']
                va_params['dataset_size'] = va_attributes[i]['total_num_photos']
                va_params['handle'] = va_handle_list[i]
                va_params['ev_init_op'] = va_iter_list[i].initializer

                name = va_attributes[i]['name']
                if i == 0:
                    va_params['best_saver'] = best_saver
                else:
                    va_params['best_saver'] = None
                print('Eval {} (#samples={})'.format(name, va_params['dataset_size']))
                match_score =  eval_one_epoch(sess, ops, va_params, name=name)
                if name.startswith('va'):
                    va_mean_match_score += match_score
                    num_valid_set += 1
            if num_valid_set > 0:
                va_mean_match_score /= num_valid_set
            if SAVE_MODEL and va_mean_match_score > best_score and best_saver is not None:
                best_score = va_mean_match_score
                print("Saving best model with valid-score = {}".format(best_score))
                best_saver.save(sess, os.path.join(log_dir, 'models-best'), write_meta_graph=False)
                with open(best_score_filename, 'w') as f:
                    f.write('{} {:g}\n'.format(step, best_score))

def overwrite_config(config):
    if config.pretrain_dir == None or len(config.pretrain_dir) == 0:
        print('Skip overwrite config')
        return config# do nothing to overwrite
    pt_config_path = os.path.join(config.pretrain_dir, 'config.pkl')
    if not os.path.exists(pt_config_path):
        print('[WARNING] Not found pretrained config, {}'.format(pt_config_path))
        return config

    with open(pt_config_path, 'rb') as f:
        pt_config = pickle.load(f)

    overwrite_attrs = [
        # Descriptor Train
        'desc_inputs',
        # Detector CNN
        'detector',
        'activ_fn',
        'leaky_alpha',
        'perform_bn',
        'net_channel',
        'net_block',
        'conv_ksize',
        'sm_ksize',
        'com_strength',
        'train_ori',
        'net_min_scale',
        'net_num_scales',
        # Descriptor CNN
        'descriptor',
        'desc_activ_fn',
        'desc_leaky_alpha',
        'desc_perform_bn',
        'desc_net_channel',
        'desc_net_depth',
        'desc_conv_ksize',
        'desc_norm',
        'desc_dim',
    ]
    check_attrs = [
        'hm_ksize',
        'hm_sigma',
        'nms_thresh',
        'nms_ksize',
        'top_k',
        'crop_radius',
        'patch_size',
    ]

    for attr in overwrite_attrs:
        src_val = getattr(config, attr)
        dst_val = getattr(pt_config, attr)
        if src_val != dst_val:
            print('Overwrite {} : {} --> {}'.format(attr, src_val, dst_val))
            setattr(config, attr, dst_val)
    for attr in check_attrs:
        src_val = getattr(config, attr)
        dst_val = getattr(pt_config, attr)
        if src_val != dst_val:
            print('[WARNING] {} has set different values, pretrain={}, now={}'.format(attr, dst_val, src_val))

    # for key, val in sorted(vars(config).items()):
    #     print(f'{key} : {val}')
    print('Finish overwritting config.')
    return config

if __name__ == '__main__':

    from common.argparse_utils import *
    parser = get_parser()

    general_arg = add_argument_group('General', parser)
    general_arg.add_argument('--num_threads', type=int, default=8,
                            help='the number of threads (for dataset)')

    train_arg = add_argument_group('Train', parser)
    train_arg.add_argument('--log_dir', type=str, default='logs/',
                            help='where to save')
    train_arg.add_argument('--pretrain_dir', type=str, default='',
                            help='pretrain model directory')
    train_arg.add_argument('--clear_logs', action='store_const',
                            const=True, default=False,
                            help='clear logs if it exists')
    train_arg.add_argument('--show_histogram', action='store_const',
                            const=True, default=False,
                            help='show variable / gradient histograms on tensorboard (consume a lot of disk space)')
    train_arg.add_argument('--max_itr', type=int, default=50000,
                            help='max epoch')
    train_arg.add_argument('--batch_size', type=int, default=6,
                            help='batch size')
    train_arg.add_argument('--optim_method', type=str, default='Adam',
                            help='adam, momentum, ftrl, rmsprop')
    train_arg.add_argument('--lr', type=float, default=1e-3,
                            help='learning rate')
    train_arg.add_argument('--lr_decay', type=str2bool, default=False,
                            help='apply lr decay')

    det_train_arg = add_argument_group('Detector Train', parser)
    det_train_arg.add_argument('--det_loss', type=str, default='l2loss',
                            help='l2loss|lift')
    det_train_arg.add_argument('--hm_ksize', type=int, default=15,
                            help='gauss kernel size for heatmaps (odd value)')
    det_train_arg.add_argument('--hm_sigma', type=float, default=0.5,
                            help='gauss kernel sigma for heatmaps')
    det_train_arg.add_argument('--nms_thresh', type=float, default=0.0,
                            help='threshold before non max suppression')
    det_train_arg.add_argument('--nms_ksize', type=int, default=5,
                            help='filter size of non max suppression')
    det_train_arg.add_argument('--top_k', type=int, default=512,
                            help='select top k keypoints')
    det_train_arg.add_argument('--weight_det_loss', type=float, default=0.01,
                            help='L_det = L2-score map + lambda * pairwise-loss')
    # supervised orientation training
    det_train_arg.add_argument('--ori_loss', type=str, default='l2loss',
                            help='orientation loss (l2loss|cosine)')
    det_train_arg.add_argument('--ori_weight', type=float, default=0.1,
                            help='orientation weight (L_det = L_score + L_ori + L_scale + L_pair)')
    det_train_arg.add_argument('--scale_weight', type=float, default=0.1,
                            help='scale weight (L_det = L_score + L_ori + L_scale + L_pair)')
    
    desc_train_arg = add_argument_group('Descriptor Train', parser)
    desc_train_arg.add_argument('--desc_loss', type=str, default='triplet',
                            help='descriptor loss')
    desc_train_arg.add_argument('--desc_margin', type=float, default=1.0,
                            help='triplet margin for descriptor loss')
    desc_train_arg.add_argument('--crop_radius', type=int, default=16, 
                            help='crop radius of region proposal')
    desc_train_arg.add_argument('--patch_size', type=int, default=32, 
                            help='cropped patch size')
    desc_train_arg.add_argument('--mining_type', type=str, default='rand_hard_sch',
                            help='negative mining type (hard|random|hard2geom|rand_hard_sch)')
    desc_train_arg.add_argument('--desc_inputs', type=str, default='photos',
                            help='descriptor inputs type (det_feats|photos|concat)')
    desc_train_arg.add_argument('--desc_train_delay', type=int, default=0,
                            help='starting iteration to train descriptor')

    dataset_arg = add_argument_group('Dataset', parser)
    dataset_arg.add_argument('--dataset', type=str, default='sfm',
                            help='dataset (scenenet|scannet)')
    dataset_arg.add_argument('--sfm_img_dir', type=str, default='./release/outdoor_examples/images',
                            help='sfm image root directory')
    dataset_arg.add_argument('--sfm_dpt_dir', type=str, default='./release/outdoor_examples/depths',
                            help='sfm depth and pose root directory')
    dataset_arg.add_argument('--sfm_seq', type=str, default='sacre_coeur',
                            help='sfm sequence name. concatenate with , if you want to add multiple sequences')
    dataset_arg.add_argument('--rot_aug', type=str2bool, default=True,
                            help='add rotation augmentation')
    dataset_arg.add_argument('--scale_aug', type=str2bool, default=True,
                            help='add rotation augmentation')
    dataset_arg.add_argument('--aug_max_degree', type=int, default=180,
                            help='max degree for rot, min_degree will be decided by -max_degree')
    dataset_arg.add_argument('--aug_max_scale', type=float, default=1.414,
                            help='max scale (in linear space, min_scale and max_scale should be symmetry in log-space)')
    dataset_arg.add_argument('--data_raw_size', type=int, default=362,
                            help='image raw size')
    dataset_arg.add_argument('--data_size', type=int, default=256,
                            help='image size (data_size * sqrt(2) = data_raw_size)')

    dataset_arg.add_argument('--depth_thresh', type=float, default=1.0,
                            help='depth threshold for inverse warping')
    dataset_arg.add_argument('--match_reproj_thresh', type=float, default=5,
                            help='matching reprojection error threshold')

    det_net_arg = add_argument_group('Detector CNN', parser)
    det_net_arg.add_argument('--detector', type=str, default='mso_resnet_detector',
                            help='network model (mso_resnet_detector)')
    det_net_arg.add_argument('--activ_fn', type=str, default='leaky_relu',
                            help='activation function (relu|leaky_relu|tanh)')
    det_net_arg.add_argument('--leaky_alpha', type=float, default=0.2,
                            help='alpha of leaky relu')
    det_net_arg.add_argument('--perform_bn', type=str2bool, default=True,
                            help='use batch normalization')
    det_net_arg.add_argument('--net_channel', type=int, default=16,
                            help='init network channels')
    det_net_arg.add_argument('--net_block', type=int, default=3,
                            help='# residual block (each block has 2 conv)')    
    det_net_arg.add_argument('--conv_ksize', type=int, default=5,
                            help='kernel size of conv layer')
    det_net_arg.add_argument('--ori_ksize', type=int, default=5,
                            help='kernel size of orientation conv layer')
    det_net_arg.add_argument('--sm_ksize', type=int, default=15,
                            help='kernel size of spatial softmax')    
    det_net_arg.add_argument('--com_strength', type=float, default=3.0,
                            help='center of the mass')
    det_net_arg.add_argument('--train_ori', type=str2bool, default=True,
                            help='train ori params')
    det_net_arg.add_argument('--net_min_scale', type=float, default=1.0/np.sqrt(2),
                            help='min scale at pyramid heatmaps')
    det_net_arg.add_argument('--net_max_scale', type=float, default=np.sqrt(2),
                            help='max scale at pyramid heatmaps')
    det_net_arg.add_argument('--net_num_scales', type=int, default=5,
                            help='number of scale maps (e.g. num_scales = (log2(1)-log2(min_scale)) / log2(2**(1/3)) )')

    desc_net_arg = add_argument_group('Descriptor CNN', parser)
    desc_net_arg.add_argument('--descriptor', type=str, default='simple_desc',
                            help='descriptor network model (simple_desc)')

    desc_net_arg.add_argument('--desc_activ_fn', type=str, default='relu',
                            help='activation function (relu|leaky_relu|tanh)')
    desc_net_arg.add_argument('--desc_leaky_alpha', type=float, default=0.2,
                            help='alpha of leaky relu')
    desc_net_arg.add_argument('--desc_perform_bn', type=str2bool, default=True,
                            help='use batch normalization')
    desc_net_arg.add_argument('--desc_net_channel', type=int, default=64,
                            help='init network channels')
    desc_net_arg.add_argument('--desc_net_depth', type=int, default=3,
                            help='# conv layers')    
    desc_net_arg.add_argument('--desc_conv_ksize', type=int, default=3,
                            help='kernel size of conv layer')    
    desc_net_arg.add_argument('--desc_norm', type=str, default='l2norm',
                            help='feature normalization (l2norm|inst|rootsift|non)')    
    desc_net_arg.add_argument('--desc_dim', type=int, default=256,
                            help='descriptor feature dimension')

    misc_arg = add_argument_group('Misc.', parser)
    misc_arg.add_argument('--train_same_time', type=str2bool, default=True,
                            help='train det loss and ori loss at the same time')
    misc_arg.add_argument('--input_inst_norm', type=str2bool, default=True,
                            help='input are normalized with inpstance norm')
    misc_arg.add_argument('--hard_geom_thresh', type=str2bool, default=32,
                            help='x,y coordinate distance threshold')
    misc_arg.add_argument('--init_num_mine', type=int, default=64,
                            help='initial top-k sampling for negative mining')
    desc_train_arg.add_argument('--min_num_pickup', type=int, default=5,
                            help='minimum random pickup')
    desc_train_arg.add_argument('--pickup_delay', type=float, default=0.9,
                            help='decay rate in every 1000 iteration')

    misc_arg.add_argument('--soft_scale', type=str2bool, default=True,
                            help='make scale differentiable')
    misc_arg.add_argument('--soft_kpts', type=str2bool, default=True,
                            help='make delta xy differentiable')
    misc_arg.add_argument('--do_softmax_kp_refine', type=str2bool, default=True,
                            help='do softmax again for kp refinement')
    misc_arg.add_argument('--kp_loc_size', type=int, default=9,
                            help='make scale differentiable')
    misc_arg.add_argument('--score_com_strength', type=float, default=100,
                            help='com strength')
    misc_arg.add_argument('--scale_com_strength', type=float, default=100,
                            help='com strength')
    misc_arg.add_argument('--kp_com_strength', type=float, default=1.0,
                            help='com strength')
    misc_arg.add_argument('--use_nms3d', type=str2bool, default=True,
                            help='use NMS3D to detect keypoints')

    config, unparsed = get_config(parser)

    if len(unparsed) > 0:
        raise ValueError('Miss finding argument: unparsed={}\n'.format(unparsed))

    config = overwrite_config(config)

    if config.aug_max_degree == 0:
        config.rot_aug = False
        config.aug_max_degree = 45
        print('Kill rot_aug because aug_max_degree=0')

    main(config)