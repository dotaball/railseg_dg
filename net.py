import torch
from torch import nn
import torch.nn.functional as F
from itertools import chain
import mix_transformer
from torchvision.models import resnet50


def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )


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


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class CFM(nn.Module):
    def __init__(self, in_channel):
        super(CFM, self).__init__()

        self.relu = nn.ReLU(True)
        self.rgb_ = BasicConv2d(in_channel, in_channel, 3, padding=1, dilation=1)
        self.srgb_ = BasicConv2d(in_channel, in_channel, 3, padding=1, dilation=1)

        self.rgb_sa = SpatialAttention()
        self.rgb_ca = ChannelAttention(in_channel)

        self.srgb_sa = SpatialAttention()
        self.srgb_ca = ChannelAttention(in_channel)

        self.ca = ChannelAttention(in_channel)

    def forward(self, rgb, srgb):

        rgb_size = rgb.size()[2:]
        srgb = F.interpolate(srgb, rgb_size, mode='bilinear', align_corners=True)

        rgb_ = self.rgb_(rgb)
        srgb_ = self.srgb_(srgb)

        rgb_sa = rgb_.mul(self.srgb_sa(srgb_)) + rgb_
        srgb_sa = srgb_.mul(self.rgb_sa(rgb_)) + srgb_

        fus = rgb_sa + srgb_sa
        fus = fus.mul(self.ca(fus))
        fus = fus + rgb

        return fus


class GAM(nn.Module):
    def __init__(self, ch_1, ch_2):  # ch_1:previous, ch_2:current/output
        super(GAM, self).__init__()
        self.ca1 = CA(ch_1)
        self.ch2 = ch_2
        self.cfm = CFM(ch_2)
        self.conv_pre = convblock(ch_1, ch_2, 3, 1, 1)

        self.conv_fuse = nn.Sequential(
            nn.Conv2d(ch_2, ch_2 // 4, 3, 1, 1),
            nn.BatchNorm2d(ch_2 // 4),
            nn.ReLU(),
            nn.Conv2d(ch_2 // 4, ch_2, 3, 1, 1),
            nn.BatchNorm2d(ch_2),
            nn.ReLU()
        )

    def forward(self, rgb, srgb, pre):
        cur_size = rgb.size()[2:]

        cur = self.cfm(rgb, srgb)

        pre = self.ca1(pre)
        pre = self.conv_pre(F.interpolate(pre, cur_size, mode='bilinear', align_corners=True))

        fus = pre + cur
        fus = pre + self.conv_fuse(fus)

        return fus


class GAM5(nn.Module):
    def __init__(self, ch_1, ch_2):  # ch_1:previous, ch_2:current/output
        super(GAM5, self).__init__()
        self.ca1 = CA(ch_1)
        self.ch2 = ch_2
        self.cfm = CFM(ch_2)
        self.conv_pre = convblock(ch_1, ch_2, 3, 1, 1)

        self.conv_fuse = nn.Sequential(
            nn.Conv2d(ch_2, ch_2 // 4, 3, 1, 1),
            nn.BatchNorm2d(ch_2 // 4),
            nn.ReLU(),
            nn.Conv2d(ch_2 // 4, ch_2, 3, 1, 1),
            nn.BatchNorm2d(ch_2),
            nn.ReLU()
        )

    def forward(self, rgb, srgb):
        cur_size = rgb.size()[2:]

        cur = self.cfm(rgb, srgb)
        fus = self.conv_fuse(cur)

        return fus


class CA(nn.Module):
    def __init__(self, in_ch):
        super(CA, self).__init__()
        self.avg_weight = nn.AdaptiveAvgPool2d(1)
        self.max_weight = nn.AdaptiveMaxPool2d(1)
        self.fus = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(in_ch // 2, in_ch, 1, 1, 0),
        )
        self.c_mask = nn.Sigmoid()

    def forward(self, x):
        avg_map_c = self.avg_weight(x)
        max_map_c = self.max_weight(x)
        c_mask = self.c_mask(torch.add(self.fus(avg_map_c), self.fus(max_map_c)))
        return torch.mul(x, c_mask)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.d4 = GAM5(512, 512)
        self.d3 = GAM(512, 320)
        self.d2 = GAM(320, 128)
        self.d1 = GAM(128, 64)

        self.score_1 = nn.Conv2d(64, 1, 1, 1, 0)
        self.score_2 = nn.Conv2d(128, 1, 1, 1, 0)
        self.score_3 = nn.Conv2d(320, 1, 1, 1, 0)
        self.score_4 = nn.Conv2d(512, 1, 1, 1, 0)

    def forward(self, rgb, srgb):

        d4 = self.d4(rgb[3], srgb[3])
        d3 = self.d3(rgb[2], srgb[2], d4)
        d2 = self.d2(rgb[1], srgb[1], d3)
        d1 = self.d1(rgb[0], srgb[0], d2)

        score1 = self.score_1(d1)
        score2 = self.score_2(d2)
        score3 = self.score_3(d3)
        score4 = self.score_4(d4)

        return score1, score2, score3, score4


class Segformer(nn.Module):
    def __init__(self, backbone, pretrained=None):
        super().__init__()

        self.encoder = getattr(mix_transformer, backbone)()
        ## initilize encoder
        if pretrained:
            state_dict = torch.load(backbone + '.pth')
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            self.encoder.load_state_dict(state_dict, )

    def forward(self):
        model = Segformer('mit_b3', pretrained=True)
        return model


class Mnet(nn.Module):
    def __init__(self, backbone="mit_b3", pretrained=True):
        super(Mnet, self).__init__()

        net = Segformer(backbone, pretrained)
        self.rgb_encoder = net.encoder
        self.srgb_encoder = net.encoder
        self.decoder = Decoder()
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb):
        # rgb
        B = rgb.shape[0]
        Hig = rgb.shape[2]
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
        srgb = F.interpolate(rgb, (160, 160), mode='bilinear', align_corners=True)
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

        score1, score2, score3, score4 = self.decoder(rgb_f, srgb_f)
        return score1, score2, score3, score4
