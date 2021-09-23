# -*- coding: utf-8 -*-

import math
import numpy as np
import cv2
import scipy

import utils

def compute_various_matching_score(match_dist_all, kpvis2w, dist_thresh_list):
    assert len(match_dist_all) == len(kpvis2w)
    # assert kpvis2w.max() == 1.0
    if kpvis2w.max() != 1.0:
        print('[WARN] kpvis2w = {}'.format(kpvis2w.max()))
    num_thresh = len(dist_thresh_list)
    match_score_list = [0] * num_thresh

    num_vis = max(kpvis2w.sum(), 1.0)

    for i in range(num_thresh):
        dist_th = dist_thresh_list[i]
        is_match = (match_dist_all <= dist_th) * kpvis2w
        match_score = is_match.sum() / num_vis
        match_score_list[i] = match_score

    match_score_list = np.array(match_score_list)

    return match_score_list

def compute_matching_score_numpy(outs, reproj_thresh):
    # kpts1 [N,2], int32
    # kpts2_corr [M,2], int32
    # xy_maps1to2 [H,W,2] float32

    # kpts1
    kpts1 = outs['kpts1']
    kpts2_corr = outs['kpts2_corr']
    xy_maps1to2 = outs['xy_maps1to2'][0]
    visible_masks1 = outs['visible_masks1'][0,...,0]
    N = len(kpts1)

    num_match = 0.0
    num_vis = 0.0
    match_dist = 0.0
    match_dist_all = np.zeros(N, np.float32)
    is_match_all = np.zeros(N, np.float32)
    reproj_thresh = 5
    for n in range(N):
        x1, y1 = kpts1[n]
        x2, y2 = kpts2_corr[n]
        xw, yw = xy_maps1to2[y1,x1]
        vis = visible_masks1[y1, x1]

        dist = np.sqrt((x2-xw)**2 + (y2-yw)**2)
        match_dist_all[n] = dist
        if vis > 0:
            num_vis += 1
            is_match = dist <= reproj_thresh
            is_match_all[n] = float(is_match)
            if is_match:
                num_match += 1
                match_dist += dist

    match_score = num_match / num_vis
    match_dist = match_dist / num_match

    outs = {
        'match_score': match_score,
        'match_dist': match_dist,
        'is_match_all': is_match_all,
        'match_dist_all': match_dist_all,
        'num_vis': num_vis,
        'num_match': num_match,
    }
    return outs
    # print(match_score, match_dist)

    # return match_score, match_dist

def compute_sift(image, num_kp=256, patch_size=32):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_kp, contrastThreshold=1e-5)
    height, width = image.shape[:2]
    kpts_info = sift.detect(image)

    fixed_size = patch_size / 6
    # fixed_size = 10

    # Fix scale and orientation
    for i in range(len(kpts_info)):
        kpts_info[i].angle = 0
        kpts_info[i].size = fixed_size

    kpts_info, feats = sift.compute(image, kpts_info)
    # kpts_info, feats = sift.detectAndCompute(image, None)
    kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts_info])
    kpts = kpts.reshape(-1, 2).astype(np.float32)
    kpts = np.round(kpts).astype(np.int32)
    kpts[:,0] = np.clip(kpts[:,0], 0, width-1)
    kpts[:,1] = np.clip(kpts[:,1], 0, height-1)
    return kpts, feats

def compute_sift_multi_scale(image, num_kp=256):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_kp, contrastThreshold=1e-5)
    height, width = image.shape[:2]
    # kpts_info = sift.detect(image)
    # kpts_info, feats = sift.compute(image, kpts_info)
    kpts_info, feats = sift.detectAndCompute(image, None)

    kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts_info])
    kpts = kpts.reshape(-1, 2).astype(np.float32)
    kpts = np.round(kpts).astype(np.int32)
    kpts[:,0] = np.clip(kpts[:,0], 0, width-1)
    kpts[:,1] = np.clip(kpts[:,1], 0, height-1)
    return kpts, feats

def compute_multi_scale_keypoints(image, num_kp=512, algo='sift'):
    if algo == 'sift':
        competitor = cv2.xfeatures2d.SIFT_create(nfeatures=num_kp, contrastThreshold=1e-5)
    elif algo == 'orb':
        # competitor = cv2.ORB_create(nfeatures=num_kp)
        # competitor = cv2.ORB_create(nfeatures=num_kp, edgeThreshold=11, patchSize=11)
        competitor = cv2.ORB_create(nfeatures=num_kp, edgeThreshold=7, patchSize=7)
    elif algo == 'akaze':
        competitor = cv2.AKAZE_create(threshold=0.000001)
    elif algo == 'kaze':
        competitor = cv2.KAZE_create()
    elif algo == 'surf':
        competitor = cv2.xfeatures2d.SURF_create(hessianThreshold=10)

    height, width = image.shape[:2]
    kpts_info, feats = competitor.detectAndCompute(image, None)
    
    N = len(kpts_info)
    if N > num_kp:
        # sort by response and filter out low score keypoints
        kp_feats = []
        for i in range(N):
            kp_feats.append([kpts_info[i], feats[i]])
        new_kp_feats = sorted(kp_feats, key=lambda x: x[0].response, reverse=True) # sort descending order
        new_kpts = [x[0] for x in new_kp_feats]
        new_feats = [x[1] for x in new_kp_feats]
        kpts_info = new_kpts[:num_kp]
        feats = np.array(new_feats[:num_kp])

    kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts_info])
    kpts = kpts.reshape(-1, 2).astype(np.float32)
    kpts = np.round(kpts).astype(np.int32)
    kpts[:,0] = np.clip(kpts[:,0], 0, width-1)
    kpts[:,1] = np.clip(kpts[:,1], 0, height-1)
    return kpts, feats

def draw_keypoints(img, kpts, valid_mask=None, color_t=(0,0xFF,0), color_f=(0,0,0xFF), radius=2, thickness=-1):
    if valid_mask is None:
        valid_mask = np.ones((len(kpts),), dtype=bool)

    canvas = img.copy()
    for kp, valid in zip(kpts, valid_mask):
        x, y = np.round(kp).astype(np.int)
        if valid:
            color = color_t
        else:
            color = color_f
        cv2.circle(canvas, (x,y), radius, color, thickness)
    return canvas


def draw_match2(img1, img2, kpts1, kpts2, scales1, scales2, oris1, oris2, kpvis2, base_rad=2):
    height, width = img1.shape[:2]
    canvas = np.concatenate([img1, img2], axis=1)
    if canvas.shape[-1] == 1:
        canvas = np.repeat(canvas, 3, -1) # gray to rgb
    for k in range(len(kpts1)):
        x1,y1 = kpts1[k]
        x2,y2 = kpts2[k]
        x1 = int(x1+0.5)
        y1 = int(y1+0.5)
        x2 = int(x2+0.5)
        y2 = int(y2+0.5)
        scl1 = scales1[k]
        scl2 = scales2[k]
        cos1, sin1 = oris1[k]
        cos2, sin2 = oris2[k]

        rad1 = int(scl1 * base_rad+0.5)
        rad2 = int(scl2 * base_rad+0.5)

        color = tuple(np.random.random(3))
        cv2.circle(canvas, (x1,y1), rad1, color)
        x12 = int(rad1 * cos1 + x1 + 0.5)
        y12 = int(rad1 * sin1 + y1 + 0.5)
        cv2.line(canvas, (x1,y1), (x12,y12), color, 1)

        if kpvis2[k] > 0:
            cv2.circle(canvas, (x2+width,y2), rad2, color)
            x22 = int(rad2 * cos2 + x2 + 0.5)
            y22 = int(rad2 * sin2 + y2 + 0.5)
            cv2.line(canvas, (x2+width,y2), (x22+width,y22), color, 1)

    return canvas

def draw_match(img1, img2, kpts1, kpts2_corr, is_match, match_score=None):
    height, width = img1.shape[:2]
    canvas = np.concatenate([img1, img2], axis=1)
    if canvas.shape[-1] == 1:
        canvas = np.repeat(canvas, 3, -1) # gray to rgb
    for k in range(len(kpts1)):
        x1,y1 = kpts1[k]
        x2,y2 = kpts2_corr[k]
        x1 = int(x1+0.5)
        y1 = int(y1+0.5)
        x2 = int(x2+0.5)
        y2 = int(y2+0.5)        
        if is_match[k] == 1:
            color = tuple(np.random.random(3))
            cv2.line(canvas, (x1,y1), (x2+width, y2), color, 1)
            cv2.circle(canvas, (x1,y1), 1, color)
            cv2.circle(canvas, (x2+width,y2), 1, color)
        else:
            cv2.circle(canvas, (x1,y1), 1, (0,0,0))
            cv2.circle(canvas, (x2+width,y2), 1, (0,0,0))

    if match_score is not None:
        num_match = int(np.sum(is_match))
        cv2.putText(canvas,'#{}-{:.1f}%'.format(num_match, match_score*100),(10,20),cv2.FONT_HERSHEY_PLAIN, 1,(0,1,0))

    return canvas



def apply_nms(score, filter_size):
    max_score = scipy.ndimage.filters.maximum_filter(
        score, filter_size, mode='constant', cval=-np.inf
    )
    # second score in region
    second_score = scipy.ndimage.filters.rank_filter(
        score, -2, filter_size, mode='constant', cval=-np.inf
    )
    # min score in region to check infs
    min_score = scipy.ndimage.filters.minimum_filter(
        score, filter_size, mode='constant', cval=np.inf
    )
    nonmax_mask = ((score == max_score) *
                           (max_score > second_score) *
                           np.isfinite(min_score))    
    return nonmax_mask

def compute_reprojection(pts1, depth1, depth2, c2Tc1, fx, fy, u0, v0, depth_thresh=1.0):
    # src_pts.shape = dst_pts.shape = [#points, 2]
    # T21 = 2T1
    height, width = depth1.shape[:2]
    ctrsl = np.array([u0, v0, 0])
    inv_f = np.array([1.0/fx, 1.0/fy, 1.0])
    f = np.array([fx, fy, 1.0])    
    xyz = np.ones((3, len(pts1)), dtype=np.float32)
    xyz[:2,:] = pts1.T
    
    pts1_int = np.round(pts1).astype(np.int32)
    Z = depth1[pts1_int[:,1], pts1_int[:,0]]
    XYZ = inv_f[:,None] * Z[None] * (xyz-ctrsl[:,None])
    
    rXYZ = c2Tc1[:3,:3].dot(XYZ) + c2Tc1[:3,[3]]

    rxyz = f[:,None] * rXYZ / (rXYZ[[2]]+1e-5) + ctrsl[:,None]
    
    camback_mask = rXYZ[2] <= 0
    rxyz[:2, camback_mask] = -1 # set invalid value
    
    rprj1 = rxyz[:2,:].T # [2,#points] --> [#points,2]
    
    valid_mask = np.logical_and(
                np.logical_and(0<=rprj1[:,0], rprj1[:,0]<width-0.5),
                np.logical_and(0<=rprj1[:,1], rprj1[:,1]<height-0.5) # avoid round
    )

    # handle occlusion problem by using depth consistency
    rxyz_valid = rxyz[:,valid_mask]
    rxy_int = np.round(rxyz_valid[:2]).astype(np.int32)
    valid_idx = np.where(valid_mask)[0]
    z1 = rXYZ[2,valid_mask]
    z2 = depth2[rxy_int[1,:], rxy_int[0,:]]
    far_depth = np.abs(z1-z2) > depth_thresh
    far_depth_idx = valid_idx[far_depth]
    valid_mask[far_depth_idx] = False    
    
    return rprj1, valid_mask

def compute_distance(pts1, pts2):
    x1mat = np.repeat(pts1[:, 0][..., None], len(pts2), axis=-1)
    y1mat = np.repeat(pts1[:, 1][..., None], len(pts2), axis=-1)
    x2mat = np.repeat(pts2[:, 0][None], len(pts1), axis=0)
    y2mat = np.repeat(pts2[:, 1][None], len(pts1), axis=0)
    
    distance = (x1mat-x2mat)**2 + (y1mat-y2mat)**2 # [#pair1,#pair2] matrix

    min_dists_1to2 = distance.min(axis=1)
    min_args_1to2 = distance.argmin(axis=1)
    min_dists_2to1 = distance.min(axis=0)
    min_args_2to1 = distance.argmin(axis=0)
    
    return min_dists_1to2, min_args_1to2, min_dists_2to1, min_args_2to1
