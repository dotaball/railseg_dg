#coding=utf-8

import os
import cv2
import numpy as np
import torch
import transform

from torch.utils.data import Dataset


mean_rgb = np.array([[[0.551, 0.619, 0.532]]])*255
std_rgb = np.array([[[0.241, 0.236, 0.244]]])*255


class Data(Dataset):
    def __init__(self, root, cla, mode='train'):
        self.samples = []
        lines = os.listdir(os.path.join(root, 'GT'))
        self.mode = mode
        for line in lines:
            rgbpath = os.path.join(root, 'RGB', line[:-4]+'.jpg')
            maskpath = os.path.join(root, 'GT', line)
            self.samples.append([rgbpath, maskpath])
        self.cla = cla

        if mode == 'train':
            self.transform = transform.Compose( transform.Normalize(mean1=mean_rgb, std1=std_rgb),
                                                transform.Resize(384, 384),
                                                transform.RandomHorizontalFlip(),
                                                transform.ToTensor())

        elif mode == 'test':
            self.transform = transform.Compose( transform.Normalize(mean1=mean_rgb, std1=std_rgb),
                                                transform.Resize(384, 384),
                                                transform.ToTensor())
        else:
            raise ValueError

    def __getitem__(self, idx):
        rgbpath, maskpath = self.samples[idx]
        rgb = cv2.imread(rgbpath).astype(np.float32)
        mask = cv2.imread(maskpath).astype(np.float32)
        H, W, C = mask.shape
        cla = self.cla
        rgb, mask = self.transform(rgb, mask)

        return rgb, mask, cla, (H, W), maskpath.split('/')[-1]

    def __len__(self):
        return len(self.samples)
