import torch.nn as nn
import torch.nn.functional as F

class CM(nn.Module):
    def __init__(self):
        super(CM, self).__init__()

        self.cnn = nn.Sequential(nn.Conv2d(3, 64, 7, stride=2, padding=3),  # 128 * 256
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 64 * 128
                                 nn.BatchNorm2d(128),
                                 nn.ReLU(),
                                 nn.Conv2d(128, 256, 3, stride=2, padding=1),  # 32 * 64
                                 nn.BatchNorm2d(256),
                                 nn.ReLU(),
                                 nn.Conv2d(256, 256, 3, stride=2, padding=1),  # 16 * 32
                                 nn.BatchNorm2d(256),
                                 nn.ReLU(),
                                 nn.Conv2d(256, 256, 3, stride=2, padding=1),  # 8 * 16
                                 nn.BatchNorm2d(256),
                                 nn.ReLU())
        self.aap = nn.AdaptiveAvgPool2d((1, 1))
        self.weight = nn.Conv2d(256, 1, 1, 1, 0)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        bs = x.size(0)
        x = self.cnn(x)
        x = self.aap(x)
        w = self.weight(x)
        w = self.sig(w)
        w = w.squeeze()

        return w