# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse, os, json
import h5py
import numpy as np
from PIL import Image

import torch
from torchvision import models
import torchvision
import torch.nn as nn
from numpy import dtype
from tqdm import tqdm
from datasets import PreprocessDataset
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument('--input_image_dir', default='D:\\University\\Project\\CLEVR_v1.0')
parser.add_argument('--max_images', default=None, type=int)
parser.add_argument('--output_h5_dir', default='D:\\University\\Project\\mac\\data')

parser.add_argument('--image_height', default=224, type=int)
parser.add_argument('--image_width', default=224, type=int)

parser.add_argument('--model', default='resnet101')
parser.add_argument('--model_stage', default=3, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--gpus', default = None, type=str)

parser.add_argument('--target', default='val', type=str)




class Model(nn.Module):
    def __init__(self, args):
        if not hasattr(torchvision.models, args.model):
          raise ValueError('Invalid model "%s"' % args.model)
        if not 'resnet' in args.model:
          raise ValueError('Feature extraction only supports ResNets')
        super(Model, self).__init__()
        self.args = args
        weights = models.ResNet101_Weights.DEFAULT
        self.cnn = getattr(models, args.model)(weights=weights)
        self.layers = [
            self.cnn.conv1,
            self.cnn.bn1,
            self.cnn.relu,
            self.cnn.maxpool,
        ]
        for i in range(args.model_stage):
            name = 'layer%d' % (i + 1)
            self.layers.append(getattr(self.cnn, name))
        self.model = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.model(x)
        return x



def main(args):
  img_num = len(os.listdir(f'{args.input_image_dir}/images/{args.target}'))

  device = 'cuda' if torch.cuda.is_available() else 'cpu'

  model = Model(args).to(device).eval()

  if torch.cuda.device_count() > 1 and args.gpus is not None:
      available_gpus = [int(x) for x in args.gpus.split(',')]
      print("Using", torch.cuda.device_count(), "GPUs!")
      model = torch.nn.DataParallel(model, available_gpus)

  img_size = (args.image_height, args.image_width)

  dataset = PreprocessDataset(data_dir=args.input_image_dir, target=args.target, img_size=img_size)
  dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16, pin_memory=True)

  output_file_path = os.path.join(args.output_h5_dir, f'{args.target}.h5')

  with h5py.File(output_file_path, 'w') as f:
    feat_dset = None
    i0 = 0

    for i, img in tqdm(enumerate(dataloader), desc='Extracting image features', total=len(dataloader)):
      img = img.to(device)

      with torch.no_grad():
        feats = model(img)
        feats = feats.data.cpu().clone().numpy()

      if feat_dset is None:
        N = img_num

        _, C, H, W = feats.shape
        feat_dset = f.create_dataset('features', (N, C, H, W),
                                     dtype=np.float32)
      i1 = i0 + len(img)
      feat_dset[i0:i1] = feats
      i0 = i1


if __name__ == '__main__':
  args = parser.parse_args()
  main(args)