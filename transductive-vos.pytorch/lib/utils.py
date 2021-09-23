import os
import sys
import functools
import logging

import torch

from PIL import Image

# so that calling setup_logger multiple times won't add many handlers
@functools.lru_cache()
def setup_logger(
    output=None, distributed_rank=0, *, color=True, name=None, abbrev_name=None
):
    """
    Initialize the detectron2 logger and set its verbosity level to "INFO".

    Args:
        output (str): a file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name (str): the root module name of this logger

    Returns:
        logging.Logger: a logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if abbrev_name is None:
        abbrev_name = name

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    # stdout logging: master only
    if distributed_rank == 0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        formatter = plain_formatter
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # file logging: all workers
    if output is not None:
        if output.endswith(".txt") or output.endswith(".log"):
            filename = output
        else:
            filename = os.path.join(output, "log.txt")
        if distributed_rank > 0:
            filename = filename + f".rank{distributed_rank}"
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)

    return logger

# cache the opened file object, so that different calls to `setup_logger`
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    return open(filename, "a")
class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


def save_prediction(prediction, palette, save_path, save_name, video_name):
    img = Image.fromarray(prediction)
    img = img.convert('L')
    img.putpalette(palette)
    img = img.convert('P')
    video_path = os.path.join(save_path, video_name)
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    img.save('{}/{}.png'.format(video_path, save_name))


def rgb2class(img, centroids):
    """
    Change rgb image array into class index.
    :param img: (batch_size, C, H, W)
    :param centroids:
    :return: (batch_size, H, W)
    """
    (batch_size, C, H, W) = img.shape
    img = img.permute(0, 2, 3, 1).reshape(-1, C)
    class_idx = torch.argmin(torch.sqrt(torch.sum((img.unsqueeze(1) - centroids) ** 2, 2)), 1)
    class_idx = torch.reshape(class_idx, (batch_size, H, W))
    return class_idx


def idx2onehot(idx, d):
    """ input:
        - idx: (H*W)
        return:
        - one_hot: (d, H*W)
    """
    n = idx.shape[0]
    one_hot = torch.zeros(d, n, device=torch.device("cuda")).scatter_(0, idx.view(1, -1), 1)

    return one_hot
