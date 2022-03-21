import torch.nn as nn
import torch.nn.functional as F


class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(32 ndf, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=3, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf * 8, 5, kernel_size=3, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.max_pool(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.max_pool(x)
        x = self.gap(x)
        x = self.classifier(x)
        x = x.squeeze()

        return x
