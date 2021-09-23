# -*- coding: utf-8 -*-

import argparse

def add_argument_group(name, parser):
    arg = parser.add_argument_group(name)
    return arg

def str2bool(v):
    return v.lower() in ('true', '1')

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    return parser

def get_config(parser):
    config, unparsed = parser.parse_known_args()
    return config, unparsed

### Examples
'''
parser.add_argument('--image', action='store_const',
                    const=True, default=False,
                    help='create preprocessed image')

#--------------------------
# Training
train_arg = add_argument_group("Train")
train_arg.add_argument("--logdir", type=str, default="logs",
                        help="where to save")
train_arg.add_argument('--clear_logs', type=str2bool, default=False,
                        help='whether to clear existing log directories before training')
train_arg.add_argument("--model", type=str, default="basic_pointnet",
                        help='model name')
train_arg.add_argument("--max_epoch", type=int, default=1,
                        help='the max number of epoch')
train_arg.add_argument('--batch_size', type=int, default=1,
                        help='batch size')
train_arg.add_argument('--normalize_input', type=str2bool, default=True,
                        help='whether to use input normalization')

#--------------------------
# Dataset
dataset_arg = add_argument_group("Dataset")
dataset_arg.add_argument('--train_data', type=str, default='./dataset/toy3d_train.npz',
                        help='where training dataset are placed')
dataset_arg.add_argument('--valid_data', type=str, default='./dataset/toy3d_valid.npz',
                        help='where training dataset are placed')
dataset_arg.add_argument('--config_data', type=str, default='./dataset/toy3d_config.pkl',
                        help='where config data are placed')

'''