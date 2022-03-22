import torch
from torch import nn
import torch.nn.functional as F
from itertools import chain
import mix_transformer


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CA(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CA, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class CF(nn.Module):
    def __init__(self, in_channel):
        super(CF, self).__init__()

        self.relu = nn.ReLU(True)
        self.rgb_ = BasicConv2d(in_channel, in_channel / 4, 1, padding=0, dilation=1)
        self.srgb_ = BasicConv2d(in_channel, in_channel / 4, 1, padding=0, dilation=1)
        self.re_rgb_ = BasicConv2d(in_channel / 4, in_channel, 1, padding=0, dilation=1)
        self.re_srgb_ = BasicConv2d(in_channel / 4, in_channel, 1, padding=0, dilation=1)

        self.rgb_sa = SA()
        self.rgb_ca = CA(in_channel / 4)

        self.srgb_sa = SA()
        self.srgb_ca = CA(in_channel / 4)

        self.ca = CA(in_channel)

    def forward(self, rgb, srgb):

        rgb_size = rgb.size()[2:]
        srgb = F.interpolate(srgb, rgb_size, mode='bilinear', align_corners=True)

        rgb_ = self.rgb_(rgb)
        srgb_ = self.srgb_(srgb)

        rgb_sa = rgb_.mul(self.srgb_sa(srgb_)) + rgb_
        srgb_sa = srgb_.mul(self.rgb_sa(rgb_)) + srgb_

        rgb_sa = self.re_rgb_(rgb_sa)
        srgb_sa = self.re_srgb_(srgb_sa)

        fus = rgb_sa + srgb_sa
        fus = self.ca(fus)

        return fus


class DEC(nn.Module):
    def __init__(self, ch_1, ch_2):  # ch_1:previous, ch_2:current/output
        super(DEC, self).__init__()

        self.ch2 = ch_2
        self.cf = CF(ch_2)
        self.conv_pre = BasicConv2d(ch_1, ch_2, 3, 1, 1)


    def forward(self, rgb, srgb, pre):
        cur_size = rgb.size()[2:]

        cur = self.cf(rgb, srgb)
        pre = self.conv_pre(F.interpolate(pre, cur_size, mode='bilinear', align_corners=True))
        fus = pre + cur

        return fus


class DEC4(nn.Module):
    def __init__(self, ch_1, ch_2):  # ch_1:previous, ch_2:current/output
        super(DEC4, self).__init__()

        self.ch2 = ch_2
        self.cf = CF(ch_2)

    def forward(self, rgb, srgb):

        cur = self.cf(rgb, srgb)

        return fus


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.de4 = DEC4(512, 512)
        self.de3 = DEC(512, 320)
        self.de2 = DEC(320, 128)
        self.de1 = DEC(128, 64)

        self.score_1 = nn.Conv2d(64, 1, 1, 1, 0)
        self.score_2 = nn.Conv2d(128, 1, 1, 1, 0)
        self.score_3 = nn.Conv2d(320, 1, 1, 1, 0)
        self.score_4 = nn.Conv2d(512, 1, 1, 1, 0)
        self.dis4 = nn.Conv2d(512, 32, 1, 1, 0)

    def forward(self, rgb, srgb):

        de4 = self.de4(rgb[3], srgb[3])
        de3 = self.de3(rgb[2], srgb[2], d4)
        de2 = self.de2(rgb[1], srgb[1], d3)
        de1 = self.de1(rgb[0], srgb[0], d2)

        score1 = self.score_1(de1)
        score2 = self.score_2(de2)
        score3 = self.score_3(de3)
        score4 = self.score_4(de4)
        dis4 = self.dis4(de4)

        return score1, score2, score3, score4, dis4


class Segformer(nn.Module):
    def __init__(self, backbone, pretrained=None):
        super().__init__()

        self.encoder = getattr(mix_transformer, backbone)()
        if pretrained:
            state_dict = torch.load(backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

    def forward(self):
        model = Segformer('mit_b3', pretrained=True)
        return model


class CFMnet(nn.Module):
    def __init__(self, backbone="mit_b3", pretrained=True):
        super(Mnet, self).__init__()

        rgb_net = Segformer(backbone, pretrained)
        srgb_net = Segformer(backbone, pretrained)
        self.rgb_encoder = rgb_net.encoder
        self.srgb_encoder = srgb_net.encoder

        self.decoder = Decoder()
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb):
        # rgb
        B = rgb.shape[0]
        rgb_f = []

        # stage 1
        x, H, W = self.rgb_encoder.patch_embed1(rgb)
        for i, blk in enumerate(self.rgb_encoder.block1):
            x = blk(x, H, W)
        x = self.rgb_encoder.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rgb_f.append(x)

        # stage 2
        x, H, W = self.rgb_encoder.patch_embed2(x)
        for i, blk in enumerate(self.rgb_encoder.block2):
            x = blk(x, H, W)
        x = self.rgb_encoder.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rgb_f.append(x)

        # stage 3
        x, H, W = self.rgb_encoder.patch_embed3(x)
        for i, blk in enumerate(self.rgb_encoder.block3):
            x = blk(x, H, W)
        x = self.rgb_encoder.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rgb_f.append(x)

        # stage 4
        x, H, W = self.rgb_encoder.patch_embed4(x)
        for i, blk in enumerate(self.rgb_encoder.block4):
            x = blk(x, H, W)
        x = self.rgb_encoder.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        rgb_f.append(x)

        # srgb
        srgb = F.interpolate(rgb, (192, 192), mode='bilinear', align_corners=True)
        srgb_f = []
        # stage 1
        x, H, W = self.srgb_encoder.patch_embed1(srgb)
        for i, blk in enumerate(self.srgb_encoder.block1):
            x = blk(x, H, W)
        x = self.srgb_encoder.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        srgb_f.append(x)

        # stage 2
        x, H, W = self.srgb_encoder.patch_embed2(x)
        for i, blk in enumerate(self.srgb_encoder.block2):
            x = blk(x, H, W)
        x = self.srgb_encoder.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        srgb_f.append(x)

        # stage 3
        x, H, W = self.srgb_encoder.patch_embed3(x)
        for i, blk in enumerate(self.srgb_encoder.block3):
            x = blk(x, H, W)
        x = self.srgb_encoder.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        srgb_f.append(x)

        # stage 4
        x, H, W = self.srgb_encoder.patch_embed4(x)
        for i, blk in enumerate(self.srgb_encoder.block4):
            x = blk(x, H, W)
        x = self.srgb_encoder.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        srgb_f.append(x)

        score1, score2, score3, score4, dis4 = self.decoder(rgb_f, srgb_f)
        return score1, score2, score3, score4, dis4
