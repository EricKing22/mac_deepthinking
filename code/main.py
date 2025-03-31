from __future__ import print_function
import torch

import argparse
import os
import random
import sys

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

from config import cfg, cfg_from_file
from utils import mkdir_p
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='optional config file', default='..\\cfg\\clevr_train_mac.yml', type=str)
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--data_dir', type=str, default='D:\\University\\Project\\CLEVR_v1.0')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--num_workers', type=int, help='number of workers')
    parser.add_argument('--batch_size', type=int, help='batch size')
    parser.add_argument('--epochs', type=int, help='number of epochs',default=50)
    parser.add_argument('--train_steps', type=int, help='number of training steps',default=4)
    parser.add_argument('--name', type=str, help='name of the training directory',default='50_epochs_4_steps_DTL_specific')
    args = parser.parse_args()
    return args


def set_traindir():
    logdir = os.path.join(os.pardir,"log")
    mkdir_p(logdir)

    num = 1
    traindir = os.path.join(logdir, f"train_{num}")

    while os.path.exists(traindir):
        num += 1
        traindir = os.path.join(logdir, f"train_{num}")

    print("Saving output to: {}".format(traindir))

    return traindir


if __name__ == "__main__":
    args = parse_args()
    if args.cfg is not None:
        cfg_from_file(args.cfg)
    if args.gpu is not None :
        cfg.GPU_ID = args.gpu
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    if args.num_workers is not None:
        cfg.WORKERS = args.num_workers
    if args.batch_size is not None:
        cfg.TRAIN.BATCH_SIZE = args.batch_size
    if args.epochs is not None:
        cfg.TRAIN.MAX_EPOCHS = args.epochs
    if args.train_steps is not None:
        cfg.TRAIN.MAX_STEPS = args.train_steps

    random.seed(args.manualSeed)
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU_ID
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    if cfg.TRAIN.FLAG:
        if args.name is not None:
            traindir = os.path.join(os.pardir, "log", args.name)
            if os.path.exists(traindir):
                raise RuntimeError("Warning: {} already exists".format(traindir))
        else:
            traindir = set_traindir()
        trainer = Trainer(traindir, cfg)
        trainer.train()
    else:
        raise NotImplementedError

