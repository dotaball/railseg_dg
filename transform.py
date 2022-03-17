import cv2
import torch
import numpy as np

class Compose(object):
    def __init__(self, *ops):
        self.ops = ops

    def __call__(self, rgb, mask):
        for op in self.ops:
            rgb, mask = op(rgb, mask)
        return rgb, mask



class Normalize(object):
    def __init__(self, mean1, std1):
        self.mean1 = mean1
        self.std1  = std1

    def __call__(self, rgb, mask):
        rgb = (rgb - self.mean1)/self.std1
        mask /= 255
        return rgb, mask

class Minusmean(object):
    def __init__(self, mean1):
        self.mean1 = mean1

    def __call__(self,rgb, mask):
        rgb = rgb - self.mean1
        mask /= 255
        return rgb, mask


class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, rgb, mask):
        rgb = cv2.resize(rgb, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize( mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return rgb, mask

class RandomCrop(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, rgb, mask):
        H,W,_ = rgb.shape
        xmin = np.random.randint(W-self.W+1)
        ymin = np.random.randint(H-self.H+1)
        rgb = rgb[ymin:ymin+self.H, xmin:xmin+self.W, :]
        mask = mask[ymin:ymin+self.H, xmin:xmin+self.W, :]
        return rgb, mask

class RandomHorizontalFlip(object):
    def __call__(self, rgb, mask):
        if np.random.randint(2)==1:
            rgb = rgb[:,::-1,:].copy()

            mask = mask[:,::-1,:].copy()
        return rgb, mask

class ToTensor(object):
    def __call__(self, rgb, mask):
        rgb = torch.from_numpy(rgb)
        rgb = rgb.permute(2, 0, 1)

        mask  = torch.from_numpy(mask)
        mask  = mask.permute(2, 0, 1)
        return rgb, mask.mean(dim=0, keepdim=True)