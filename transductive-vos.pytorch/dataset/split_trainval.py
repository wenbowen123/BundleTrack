import os
import argparse
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, required=True,
                    help='path to DAVIS set')
args = parser.parse_args()

trainval_path = args.i

train_list = os.path.join(trainval_path, 'ImageSets/2017/train.txt')
train_list = open(train_list).readlines()
for i in range(len(train_list)):
    train_list[i] = train_list[i].strip()

val_list = os.path.join(trainval_path, 'ImageSets/2017/val.txt')
val_list = open(val_list).readlines()
for i in range(len(val_list)):
    val_list[i] = val_list[i].strip()

full_img_path = os.path.join(trainval_path, 'JPEGImages/480p')
full_annotation_path = os.path.join(trainval_path, 'Annotations/480p/')
full_video_list = os.listdir(full_img_path)

train_img = './DAVIS_train/JPEGImages/480p'
train_annotations = './DAVIS_train/Annotations/480p'
val_img = './DAVIS_val/JPEGImages/480p'
val_annotations = './DAVIS_val/Annotations/480p'
l = [train_img, train_annotations, val_img, val_annotations]
for p in l:
    if not os.path.exists(p):
        os.makedirs(p)

for video in full_video_list:
    src1 = os.path.join(full_annotation_path, video)
    src2 = os.path.join(full_img_path, video)
    if video in train_list:
        dest1 = train_annotations
        dest2 = train_img
    else:
        dest1 = val_annotations
        dest2 = val_img
    if not os.path.exists(dest1):
        os.makedirs(dest1)
    if not os.path.exists(dest2):
        os.makedirs(dest2)
    shutil.move(src1, dest1)
    shutil.move(src2, dest2)
print('success')
