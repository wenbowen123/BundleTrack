#
# Authors: Bowen Wen
# Contact: wenbowenxjtu@gmail.com
# Created in 2021

# Copyright (c) Rutgers University, 2021 All rights reserved.

# Bowen Wen and Kostas Bekris. "BundleTrack: 6D Pose Tracking for Novel Objects
#  without Instance or Category-Level 3D Models."
#  In IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). 2021.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Bowen Wen, Kostas Bekris, Rutgers University,
#       nor the names of its contributors may be used to
#       endorse or promote products derived from this software without
#       specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 'AS IS' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#


import os,sys
code_path = os.path.dirname(os.path.realpath(__file__))
import glob,argparse
import numpy as np
import math
import _pickle as cPickle

'''The experiments that reproduce paper result
https://github.com/j96w/6-PACK/issues/19
'''



synset_names = ['BG',
                'bottle',
                'bowl',
                'camera',
                'can',
                'laptop',
                'mug'
                ]


def normalizeRotation(pose):
    for i in range(3):
        norm = np.linalg.norm(pose[:,i])
        pose[:,i] /= norm
    return pose

def compute_3d_iou_new(RT_1, RT_2, noc_cube_1, noc_cube_2, handle_visibility, class_name_1, class_name_2):
    '''Computes IoU overlaps between two 3d bboxes.
       bbox_3d_1, bbox_3d_1: [3, 8]
    '''
    def asymmetric_3d_iou(RT_1, RT_2, noc_cube_1, noc_cube_2):
        bbox_3d_1 = transform_coordinates_3d(noc_cube_1, RT_1)

        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, RT_2)


        bbox_1_max = np.amax(bbox_3d_1, axis=0)
        bbox_1_min = np.amin(bbox_3d_1, axis=0)
        bbox_2_max = np.amax(bbox_3d_2, axis=0)
        bbox_2_min = np.amin(bbox_3d_2, axis=0)

        overlap_min = np.maximum(bbox_1_min, bbox_2_min)
        overlap_max = np.minimum(bbox_1_max, bbox_2_max)

        if np.amin(overlap_max - overlap_min) <0:
            intersections = 0
        else:
            intersections = np.prod(overlap_max - overlap_min)
        union = np.prod(bbox_1_max - bbox_1_min) + np.prod(bbox_2_max - bbox_2_min) - intersections
        overlaps = intersections / union
        return overlaps

    symmetry_flag = False
    if (class_name_1 in ['bottle', 'bowl', 'can'] and class_name_1 == class_name_2) or (class_name_1 == 'mug' and class_name_1 == class_name_2 and handle_visibility==0):

        bbox_3d_2 = transform_coordinates_3d(noc_cube_2, RT_2)

        def y_rotation_matrix(theta):
            return np.array([[np.cos(theta), 0, np.sin(theta), 0],
                             [0, 1, 0 , 0],
                             [-np.sin(theta), 0, np.cos(theta), 0],
                             [0, 0, 0 , 1]])

        n = 20
        max_iou = 0
        for i in range(n):
            rotated_RT_1 = RT_1@y_rotation_matrix(2*math.pi*i/float(n))
            max_iou = max(max_iou,
                          asymmetric_3d_iou(rotated_RT_1, RT_2, noc_cube_1, noc_cube_2))
    else:
        max_iou = asymmetric_3d_iou(RT_1, RT_2, noc_cube_1, noc_cube_2)

    return max_iou

def transform_coordinates_3d(coordinates, RT):
    assert coordinates.shape[0] == 3
    coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
    new_coordinates = RT @ coordinates
    new_coordinates = new_coordinates[:3, :]/new_coordinates[3, :]
    return new_coordinates

def compute_RT_degree_cm_symmetry(RT_1, RT_2, class_id, handle_visibility, synset_names):
    if RT_1 is None or RT_2 is None:
        return 10000,10000
    try:
        assert np.array_equal(RT_1[3, :], RT_2[3, :])
        assert np.array_equal(RT_1[3, :], np.array([0, 0, 0, 1]))
    except AssertionError:
        return 10000,10000

    R1 = RT_1[:3, :3] / np.cbrt(np.linalg.det(RT_1[:3, :3]))
    T1 = RT_1[:3, 3]

    R2 = RT_2[:3, :3] / np.cbrt(np.linalg.det(RT_2[:3, :3]))
    T2 = RT_2[:3, 3]

    if synset_names[class_id] in ['bottle', 'can', 'bowl']:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif synset_names[class_id] == 'mug' and handle_visibility==0:
        y = np.array([0, 1, 0])
        y1 = R1 @ y
        y2 = R2 @ y
        theta = np.arccos(y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))
    elif synset_names[class_id] in ['phone', 'eggbox', 'glue']:
        y_180_RT = np.diag([-1.0, 1.0, -1.0])
        R = R1 @ R2.transpose()
        R_rot = R1 @ y_180_RT @ R2.transpose()
        theta = min(np.arccos((np.trace(R) - 1) / 2),
                    np.arccos((np.trace(R_rot) - 1) / 2))
    else:
        R = R1 @ R2.transpose()
        theta = np.arccos((np.trace(R) - 1) / 2)

    theta *= 180 / np.pi
    shift = np.linalg.norm(T1 - T2)
    result = np.array([theta, shift])

    return result

score_dict = {}

def main(args):
    verbose = args.verbose
    pred_data = args.pred_data
    pred_list = np.array(list(map(int,args.pred_list.split(','))))

    all_score = []
    all_score25 = []
    all_rot_err = []
    all_trans_err = []
    for exp in pred_list:
        print('pred_list exp=',exp)
        all_score.append([])
        all_score25.append([])
        all_rot_err.append([])
        all_trans_err.append([])
        pred_data_dir = pred_data+'{}'.format(exp)
        score = 0
        score_25 = 0
        rot_err = 0
        trans_err = 0
        for cls_idx in [1,2,3,4,5,6]:
            cls_num = 0
            cls_test_num = 0
            cls_in_5_5 = 0
            cls_iou_25 = 0
            cls_path = os.path.join(args.nocs_dir,'NOCS-REAL275-additional', "data_list/real_val", str(cls_idx))
            cls_rot = []
            cls_trans = []
            model_list = glob.glob(os.path.join(cls_path, "*"))
            assert len(model_list)>0
            for model_path in model_list:
                scene_his = ""
                model_num = 0
                scene_num = 0
                scene_in = 0
                model_name = model_path.split("/")[-1]
                if not model_name in score_dict:
                    score_dict[model_name] = {}
                list_path = os.path.join(model_path, "list.txt")
                print("list_path: ",list_path)
                with open(list_path, 'r') as list_file:
                    for img_path in list_file:
                        img_path = img_path.replace('real_val','real_test')
                        img_path = os.path.join(args.nocs_dir, img_path)
                        img_path = img_path.replace("\n", "")
                        scene = img_path.split("/")[-2]
                        frame_id = int(os.path.basename(img_path))
                        nocs_gt_path = os.path.join(args.nocs_dir, "gts", "real_test", "results_real_test_" + scene + "_%04d.pkl"%(frame_id))
                        if not os.path.exists(nocs_gt_path):
                            print('ERROR nocs_gt_path not found:',nocs_gt_path)
                            continue
                        cls_num = cls_num + 1
                        if scene != scene_his:
                            if scene_his != "":
                                if not scene_his in score_dict[model_name]:
                                    score_dict[model_name][scene_his] = {'temp':0,'score':0}
                                if scene_in/scene_num > score_dict[model_name][scene_his]['score']:
                                    score_dict[model_name][scene_his]['score'] = scene_in/scene_num
                                    score_dict[model_name][scene_his]['temp'] = exp
                            scene_his = scene
                            scene_num = 0
                            scene_in = 0
                        pred_path = os.path.join(pred_data_dir, "temp" + str(cls_idx), model_name + "_" + scene + "_"+str(model_num) + "_pose.txt")
                        model_num = model_num + 1
                        scene_num = scene_num + 1
                        if not os.path.exists(pred_path):
                            print('ERROR {} not found'.format(pred_path))
                            continue
                        obj_path = img_path+'_meta.txt'
                        ins_id = -1
                        num_idx = 0
                        with open(obj_path, "r") as obj_f:
                            for line in obj_f:
                                if int(line.split(" ")[1]) == cls_idx and line.split(" ")[-1].replace("\n","") == model_name:
                                    ins_id = int(line.split(" ")[0])
                                    break
                                num_idx = num_idx + 1
                        if ins_id == -1:
                            continue
                        with open(nocs_gt_path, 'rb') as f:
                            result = cPickle.load(f)
                            try:
                                gt_pose = result['gt_RTs'][num_idx]
                            except Exception as e:
                                print('ERROR loading gt from {}:'.format(nocs_gt_path),e)
                                gt_pose = np.eye(4)
                        gt_pose[:3,3] = gt_pose[:3,3] * 1000
                        with open(pred_path, "r") as pred_f:
                            pred_pose = []
                            for i in range(3):
                                pred_pose.append([])
                                new_line = pred_f.readline()
                                for j in range(3):
                                    pred_pose[i].append(float(new_line.split(" ")[j]))
                            new_line = pred_f.readline()
                            for i in range(3):
                                pred_pose[i].append(float(new_line.split(" ")[i]))
                            pred_pose.append([0.0,0.0,0.0,1.0])

                        z_180_RT = np.zeros((4, 4), dtype=np.float32)
                        z_180_RT[:3, :3] = np.diag([-1, -1, 1])
                        z_180_RT[3, 3] = 1
                        pred_pose = z_180_RT @ pred_pose
                        gt_pose = np.array(gt_pose)
                        pred_pose = np.array(pred_pose)
                        pred_pose = normalizeRotation(pred_pose)
                        gt_pose = normalizeRotation(gt_pose)
                        result = compute_RT_degree_cm_symmetry(pred_pose, gt_pose, cls_idx, 1, synset_names)
                        bbox = np.loadtxt(args.nocs_dir + "/NOCS-REAL275-additional/model_scales/" + model_name + ".txt").transpose()
                        miou = compute_3d_iou_new(gt_pose, pred_pose, bbox, bbox, 1, synset_names[cls_idx], synset_names[cls_idx])
                        if verbose:
                            print('model_name={} scene={} frame_id={} deg/mm={}'.format(model_name,scene,frame_id,result))
                        cls_test_num = cls_test_num + 1
                        if miou > 0.25 and result[0] < 360:
                            cls_rot.append(result[0])
                        if miou > 0.25:
                            cls_trans.append(result[1])
                        if miou > 0.25:
                            cls_iou_25 = cls_iou_25 + 1
                        if result[0] < 5 and result[1] < 50:
                            scene_in = scene_in + 1
                            cls_in_5_5 = cls_in_5_5 + 1

                if not scene_his in score_dict[model_name]:
                    score_dict[model_name][scene_his] = {'temp':0,'score':0}
                if scene_in/scene_num > score_dict[model_name][scene_his]['score']:
                    score_dict[model_name][scene_his]['score'] = scene_in/scene_num
                    score_dict[model_name][scene_his]['temp'] = exp
                print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
            all_score[-1].append(cls_in_5_5/cls_num)
            all_score25[-1].append(cls_iou_25/cls_num)
            all_rot_err[-1].append(np.mean(cls_rot))
            all_trans_err[-1].append(np.mean(cls_trans))
            score = score + (cls_in_5_5/cls_num)/6
            score_25 = score_25 + (cls_iou_25/cls_num)/6
            rot_err = rot_err +  np.mean(cls_rot)/6
            trans_err = trans_err +  np.mean(cls_trans)/6
            print("********************************************************")
            print('class={}'.format(synset_names[cls_idx]))
            print("5cm 5degree:",cls_in_5_5/cls_num*100)
            print("IoU 25:     ",cls_iou_25/cls_num*100)
            print("rot error:  ",np.mean(cls_rot))
            print("tran error: ",np.mean(cls_trans)/10)
        print("********************************************************")
        print("5cm 5degree:",score*100)
        print("IoU 25:     ",score_25*100)
        print("rot error:  ",rot_err)
        print("tran error: ",trans_err/10)
        all_score[-1].append(score)
        all_score25[-1].append(score_25)
        all_rot_err[-1].append(rot_err)
        all_trans_err[-1].append(trans_err)
    print("********************************************************")
    print("Mean 5cm 5degree:",np.mean(np.array(all_score)*100,0)[-1])
    print("Mean IoU 25:     ",np.mean(np.array(all_score25)*100,0)[-1])
    print("Mean rot error:  ",np.mean(np.array(all_rot_err),0)[-1])
    print("Mean tran error: ",np.mean(np.array(all_trans_err)/10,0)[-1])
    print("********************************************************")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--nocs_dir', type=str, required=True)
    parser.add_argument('--pred_data', type=str, default="/home/bowen/debug/nocs_res_ours/TEMP_")
    parser.add_argument('--pred_list', type=str, default="50")  # split by comma, eval_id list, if you want to test the mean score of TEST_1 and TEST_2, change to "1,2"
    args = parser.parse_args()

    main(args)
