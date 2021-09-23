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


import argparse,glob,cv2,re,os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
import time

import torch
import torch.nn as nn
import numpy as np
from dataset.transforms import *
from multiprocessing import Pool
import multiprocessing
from functools import partial
from itertools import repeat

import dataset
import modeling
from lib.utils import AverageMeter, save_prediction, idx2onehot
from lib.predict import predict, prepare_first_frame,get_spatial_weight
from PIL import Image



parser = argparse.ArgumentParser()
parser.add_argument('--ref_num', '-n', type=int, default=9,
					help='number of reference frames for inference')
parser.add_argument('--resume', '-r', type=str,
					help='path to the resumed checkpoint', default='{}/pretrained/davis_trainval.pth.tar'.format(code_dir))
parser.add_argument('--model', type=str, default='resnet50',
					help='network architecture, resnet18, resnet50 or resnet101')
parser.add_argument('--sigma1', type=float, default=8.0,
					help='smaller sigma in the motion model for dense spatial weight')
parser.add_argument('--sigma2', type=float, default=21.0,
					help='bigger sigma in the motion model for sparse spatial weight')
parser.add_argument('--range', type=int, default=40,
					help='range of frames for inference')
parser.add_argument('--temperature', '-t', type=float, default=1.0, help='temperature parameter')
parser.add_argument('--img_dir', type=str, required=True)
parser.add_argument('--init_mask_file', type=str, required=True)
parser.add_argument('--mask_save_dir', type=str, required=True)



def run_one_video(model,args):
	color_files = sorted(glob.glob(f'{args.img_dir}/*'))
	batch_time = AverageMeter()

	rgb_normalize = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

	def prepare_first_frame(init_mask,sigma1=8,sigma2=21):
		first_annotation = Image.fromarray(init_mask)
		(H, W) = np.asarray(first_annotation).shape
		H_d = int(np.ceil(H / 8))
		W_d = int(np.ceil(W / 8))
		palette = first_annotation.getpalette()
		label = np.asarray(first_annotation)
		d = np.max(label) + 1
		label = torch.Tensor(label).long().cuda()
		label_1hot = idx2onehot(label.view(-1), d).reshape(1, d, H, W)
		label_1hot = torch.nn.functional.interpolate(label_1hot,size=(H_d, W_d),mode='bilinear',align_corners=False)
		label_1hot = label_1hot.reshape(d, -1).unsqueeze(1)
		weight_dense = get_spatial_weight((H_d, W_d), sigma1)
		weight_sparse = get_spatial_weight((H_d, W_d), sigma2)
		return label_1hot, d, palette, weight_dense, weight_sparse

	def write_mask(color,color_file,fg_mask):
		out_dir = args.mask_save_dir
		print(f"Saving mask to {out_dir}")
		name = os.path.basename(color_file).split('.')[0]
		out_file = f'{out_dir}/{name}.png'
		os.system('mkdir -p {}'.format(os.path.dirname(out_file)))
		fg_mask[fg_mask>0] = 255
		cv2.imwrite(out_file, fg_mask.astype(np.uint8))

	frame_idx = 0
	with torch.no_grad():
		for i, color_file in enumerate(color_files):
			frame_id = int(os.path.basename(color_file)[:4])
			color = cv2.imread(color_file)
			input = cv2.cvtColor(color,cv2.COLOR_BGR2RGB)
			input = rgb_normalize(input).unsqueeze(0)
			if frame_idx == 0:
				input = input.cuda()
				with torch.no_grad():
					feats_history = model(input)
				init_file = args.init_mask_file
				print("init_file: ",init_file)
				init_mask = cv2.imread(init_file,-1)
				if len(init_mask.shape)==3:
					init_mask = init_mask[...,0]

				label_history, d, palette, weight_dense, weight_sparse = prepare_first_frame(init_mask,args.sigma1,args.sigma2)
				frame_idx += 1
				write_mask(color,color_file,init_mask)
				continue

			(batch_size, num_channels, H, W) = input.shape
			input = input.cuda()

			start = time.time()
			features = model(input)
			(_, feature_dim, H_d, W_d) = features.shape
			prediction = predict(feats_history,
								 features[0],
								 label_history,
								 weight_dense,
								 weight_sparse,
								 frame_idx,
								 args
								 )

			new_label = idx2onehot(torch.argmax(prediction, 0), d).unsqueeze(1)
			label_history = torch.cat((label_history, new_label), 1)
			feats_history = torch.cat((feats_history, features), 0)

			frame_idx += 1

			prediction = torch.nn.functional.interpolate(prediction.view(1, d, H_d, W_d),
														 size=(H, W),
														 mode='bilinear',
														 align_corners=False)
			prediction = torch.argmax(prediction[0], 0)  # (H, W)
			batch_time.update(time.time() - start)
			prediction = prediction.data.cpu().numpy()
			write_mask(color,color_file,prediction)

	print('Finished inference, avg time={}'.format(batch_time.avg))


def main():
	global args
	args = parser.parse_args()

	model = modeling.VOSNet(model=args.model)
	model = nn.DataParallel(model)
	model.cuda()

	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			model.load_state_dict(checkpoint['state_dict'])
			print("=> loaded checkpoint '{}'"
					.format(args.resume))
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))
	model.eval()

	run_one_video(model,args)



if __name__ == '__main__':
	main()
