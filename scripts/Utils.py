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


import os, sys, time
import open3d as o3d
import cv2
from PIL import Image, ImageDraw
import numpy as np
import multiprocessing as mp
sys.path.append('../')
import math,glob,re,copy
from scipy.spatial import ConvexHull, distance_matrix
import scipy.spatial as spatial
import transformations

COLOR_MAP=np.array([[0, 0, 0], #Ignore
					[128,0,0], #Background
					[0,128,0], #Wall
					[128,128,0], #Floor
					[0,0,128], #Ceiling
					[128,0,128], #Table
					[0,128,128], #Chair
					[128,128,128], #Window
					[64,0,0], #Door
					[192,0,0], #Monitor
					[64, 128, 0],     # 11th
					[192, 0, 128],
					[64, 128, 128],
					[192, 128, 128],
					[0, 64, 0],
					[128, 64, 0],
					[0, 192, 0],
					[128, 192, 0], # defined for 18 classes currently
					])


def add(pred,gt,model):
	"""
	Average Distance of Model Points for objects with no indistinguishable views
	- by Hinterstoisser et al. (ACCV 2012).
	"""
	pred_model = copy.deepcopy(model)
	gt_model = copy.deepcopy(model)
	pred_model.transform(pred)
	gt_model.transform(gt)
	e = np.linalg.norm(np.asarray(pred_model.points) - np.asarray(gt_model.points), axis=1).mean()
	return e

def adi(pred,gt,model):
	"""
	@pred: 4x4 mat
	@gt:
	@model: open3d pcd model
	"""
	pred_model = copy.deepcopy(model)
	gt_model = copy.deepcopy(model)
	pred_model.transform(pred)
	gt_model.transform(gt)

	nn_index = spatial.cKDTree(np.asarray(pred_model.points).copy())
	nn_dists, _ = nn_index.query(np.asarray(gt_model.points).copy(), k=1, n_jobs=10)
	e = nn_dists.mean()
	return e

class AverageMeter(object):
	"""
	Compute and store running average values.
	"""

	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count



def toOpen3dCloud(points,colors):
	import open3d as o3d
	cloud = o3d.geometry.PointCloud()
	cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
	cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64)/255.0)
	return cloud



def normalizeRotation(pose):
	for i in range(3):
		norm = np.linalg.norm(pose[:3,i])
		pose[:3,i] /= norm
	return pose

def random_direction():
	theta = np.random.uniform(0, 1) * np.pi * 2
	phi = np.arccos((2 * (np.random.uniform(0, 1))) - 1)

	def sph2cart(phi, theta, r):
		points = np.zeros(3)
		points[0] = r * math.sin(phi) * math.cos(theta)
		points[1] = r * math.sin(phi) * math.sin(theta)
		points[2] = r * math.cos(phi)
		return points

	return sph2cart(phi, theta, 1)


