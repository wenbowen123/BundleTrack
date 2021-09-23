import os
import random
import numbers

from torchvision import transforms


def get_crop_params(img_size, output_size):
    """ input:
        - img_size : tuple of (w, h), original image size
        - output_size: desired output size, one int or tuple
        return:
        - i
        - j
        - th
        - tw
    """
    w, h = img_size
    if isinstance(output_size, numbers.Number):
        th, tw = (output_size, output_size)
    else:
        th, tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w

    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw


def crop(img, i, j, h, w):
    """Crop the given PIL Image.
    Args:
        img (PIL Image): Image to be cropped.
        i: Upper pixel coordinate.
        j: Left pixel coordinate.
        h: Height of the cropped image.
        w: Width of the cropped image.
    Returns:
        PIL Image: Cropped image.
    """
    return img.crop((j, i, j + w, i + h))


class FixedColorJitter(transforms.ColorJitter):
    """
        Same ColorJitter class, only fixes the transform params once instantiated.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super(FixedColorJitter, self).__init__(brightness, contrast, saturation, hue)
        self.transform = self.get_params(self.brightness, self.contrast,
                                         self.saturation, self.hue)

    def __call__(self, img):
        return self.transform(img)


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                images.append(item)
    return images
