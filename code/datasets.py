from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torchvision.transforms as transforms
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import glob
from pathlib import Path
import json
import h5py
import re
import random

from torch.utils.data import Dataset

from config import cfg


class ClevrDataset(Dataset):
    # this dataset assumes h5 feature file and pickle question file stores img information in the same order
    def __init__(self, data_dir, split='train'):

        with open(os.path.join(data_dir, '{}.pkl'.format(split)), 'rb') as f:
            self.data = pickle.load(f)
        self.imgs = h5py.File(os.path.join(data_dir, '{}.h5'.format(split)), 'r')['features']

    def __getitem__(self, index):
        imgfile, question, answer = self.data[index]
        # [:-4] removes the .png extension
        img_id = int(imgfile.rsplit('_', 1)[1][:-4])

        # create img tensor
        img = torch.from_numpy(self.imgs[img_id])

        return img, question, len(question), answer

    def __len__(self):
        return len(self.data)


def collate_fn(batch):
    images, lengths, answers, _ = [], [], [], []
    batch_size = len(batch)

    max_len = max(map(lambda x: len(x[1]), batch))

    questions = np.zeros((batch_size, max_len), dtype=np.int64)
    sort_by_len = sorted(batch, key=lambda x: len(x[1]), reverse=True)

    for i, b in enumerate(sort_by_len):
        image, question, length, answer = b
        images.append(image)
        length = len(question)
        questions[i, :length] = question
        lengths.append(length)
        answers.append(answer)

    return {'image': torch.stack(images), 'question': torch.from_numpy(questions),
            'answer': torch.LongTensor(answers), 'question_length': lengths}


class PreprocessDataset(Dataset):
    def __init__(self, data_dir, img_size, target):
        super(PreprocessDataset, self).__init__()
        self.img_dir = os.path.join(data_dir, 'images', target)
        self.img_files = []
        # sort files in numerical order
        for fn in os.listdir(self.img_dir):
            if not fn.endswith('.png'): continue
            self.img_files.append((os.path.join(self.img_dir, fn), int(os.path.splitext(fn)[0].split('_')[-1])))

        self.img_files.sort(key=lambda x: x[1])
        self.img_files = [f[0] for f in self.img_files]



        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def __getitem__(self, index):
        img_path = self.img_files[index]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img


    def __len__(self):
        return len(self.img_files)
