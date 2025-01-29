from __future__ import print_function
import torch

import argparse
import os
import random
import sys
import datetime
import dateutil
import dateutil.tz
import shutil

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from config import cfg, cfg_from_file
from utils import mkdir_p
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', help='optional config file', default='..\\cfg\\clevr_train_mac.yml', type=str)
    parser.add_argument('--gpu',  dest='gpu', type=str)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='D:\\University\\Project\\CLEVR_v1.0')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def set_traindir():
    logdir = os.path.join(os.pardir,"log")
    mkdir_p(logdir)

    num = len(os.listdir(logdir))
    traindir = os.path.join(logdir, f"train_{num+1}")
    print("Saving output to: {}".format(traindir))

    return traindir


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.gpu is not None :
        cfg.GPU_ID = args.gpu
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    if cfg.TRAIN.FLAG:
        traindir = set_traindir()
        trainer = Trainer(traindir, cfg)
        trainer.train()
    else:
        raise NotImplementedError

