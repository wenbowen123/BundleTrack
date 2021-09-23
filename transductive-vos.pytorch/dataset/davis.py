from io import BytesIO
from PIL import Image

import torch
import numpy as np
from torchvision import datasets

from dataset.transforms import *


class DavisTrain(datasets.ImageFolder):
    def __init__(self,
                 img_root,
                 annotation_root,
                 cropping=256,
                 frame_num=10,
                 transform=None,
                 target_transform=None,
                 color_jitter=False):
        super(DavisTrain, self).__init__(img_root,
                                         transform=transform,
                                         target_transform=target_transform)
        # img root and annotation root should have the same class_to_idx
        self.annotations = make_dataset(annotation_root, self.class_to_idx)
        self.cropping = cropping
        self.frame_num = frame_num
        self.color_jitter = color_jitter
        self.rgb_normalize = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
        # read all jpgs and annotations into mem to speed up training
        self.img_bytes = []
        self.annotation_bytes = []
        idx = 0
        for path, _ in self.imgs:
            with open(path, 'rb') as f:
                self.img_bytes.append(f.read())
                idx += 1
            if idx % 500 == 0:
                print("%d images loaded." % idx)
        print("JPEGImages loaded: ", len(self.img_bytes))
        idx = 0
        for path, _ in self.annotations:
            with open(path, 'rb') as f:
                self.annotation_bytes.append(f.read())
                idx += 1
            if idx % 500 == 0:
                print("%d annotations loaded." % idx)
        print("Annotations loaded: ", len(self.annotation_bytes))

    def __getitem__(self, index):
        img_output = []
        annotation_output = []

        # if index reaches end of dataset, get the last frames
        if index + self.frame_num > len(self.imgs):
            index = len(self.imgs) - self.frame_num
        while not self.__is_from_same_video__(index):
            index -= 1
        # get transform params
        if self.color_jitter:
            color_transform = FixedColorJitter(brightness=0.4, contrast=0.4,
                                               saturation=0.4, hue=0.4)
        crop_i, crop_j, th, tw = 0, 0, 0, 0
        h_flip = True if random.random() < 0.5 else False
        v_flip = True if random.random() < 0.5 else False
        for i in range(self.frame_num):
            path, video_index = self.imgs[index + i]
            img = Image.open(BytesIO(self.img_bytes[index + i]))
            img = img.convert('RGB')
            annotation = Image.open(BytesIO(self.annotation_bytes[index + i]))  # (W, H), -P mode
            annotation = annotation.convert('RGB')

            if h_flip:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                annotation = annotation.transpose(Image.FLIP_LEFT_RIGHT)
            if v_flip:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                annotation = annotation.transpose(Image.FLIP_TOP_BOTTOM)
            if i == 0:
                W, H = img.size
                crop_i, crop_j, th, tw = get_crop_params((W, H), self.cropping)

            # all images and annotations should cropped in the same way
            img_cropped = crop(img, crop_i, crop_j, th, tw)
            annotation_cropped = crop(annotation, crop_i, crop_j, th, tw)
            if self.color_jitter:
                img_cropped = color_transform(img_cropped)

            img_cropped = self.rgb_normalize(img_cropped).numpy()
            annotation_cropped = np.asarray(annotation_cropped).transpose((2, 0, 1))
            img_output.append(img_cropped)
            annotation_output.append(annotation_cropped)

        img_output = torch.from_numpy(np.asarray(img_output)).float()
        annotation_output = torch.from_numpy(np.asarray(annotation_output)).float()
        return img_output, annotation_output, video_index

    def __is_from_same_video__(self, index):
        _, indexStart = self.imgs[index]
        _, indexEnd = self.imgs[index + self.frame_num - 1]
        return indexStart == indexEnd


class DavisInference(datasets.ImageFolder):
    """
        Load one frame at a time.
        Used for inference.
    """

    def __init__(self,
                 root,
                 transform=None,
                 target_transform=None):
        super(DavisInference, self).__init__(root,
                                             transform=transform,
                                             target_transform=target_transform)
        self.img_bytes = []
        self.rgb_normalize = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(
                                                     mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
        for path, _ in self.imgs:
            with open(path, 'rb') as f:
                self.img_bytes.append(f.read())
        print("Tracking folder JPEGImages loaded: ", len(self.img_bytes))

    def __getitem__(self, index):
        path, video_index = self.imgs[index]
        img = Image.open(BytesIO(self.img_bytes[index]))
        img = img.convert('RGB')

        img_original = np.asarray(img)

        output = self.rgb_normalize(img_original)
        return output, video_index, img_original

    def __len__(self):
        return len(self.imgs)
