import os
from torch.utils.data import DataLoader
from dataset import Data
import torch.nn.functional as F
import torch
import cv2
import time
from CFM import CFMnet
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == '__main__':
    model_path = './model/final.pth'
    out_path = './output'
    data = Data(root='./dataset_root', mode='test')

    loader = DataLoader(data, batch_size=1, shuffle=False)
    net = CFMnet().cuda()
    net.load_state_dict(torch.load(model_path))
    if not os.path.exists(out_path): os.mkdir(out_path)

    img_num = len(loader)
    net.eval()

    with torch.no_grad():
        for rgb, _, (H, W), name in loader:
            print(name[0])

            score1, score2, score3, score4 = net(rgb.cuda().float())
            score = F.interpolate(score1, size=(H, W), mode='bilinear', align_corners=True)
            pred = np.squeeze(torch.sigmoid(score).cpu().data.numpy())

            pred[pred > 0.5] = 1
            pred[pred < 1] = 0
            cv2.imwrite(os.path.join(out_path, name[0][:-4] + '.png'), pred)




