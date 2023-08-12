#!/usr/bin/env python3

import torch
from torch import nn

from det_sdk.backbones.backbone_mgr import GlobalBackbones


class VGG_FT_HEAD(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(VGG_FT_HEAD, self).__init__(*args, **kwargs)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.layers = [
            self.pool, self.conv,
            nn.ReLU(inplace=True), self.conv2, nn.ReLU(inplace=True)
        ]
        

class SSD_VGG_300(nn.Module):
    __BACKBONE__ = "vgg"
    def __init__(self, num_classes, pretrained = False):
        super(SSD_VGG_300, self).__init__()
        self.num_classes = num_classes
        self.vgg_ft  = GlobalBackbones().get_backbone("vgg16_official", pretrained).features
        self.ft_head = VGG_FT_HEAD()
        # self.extra = add_head(2014)
        # self.L2Norm = L2Norm(512, 20)
        # mbox = [4, 6, 6, 6, 4, 4]

        # loc_layers = []
        # conf_layers = []
        # backbone_source = [21, -2]

        # for k,v in enumerate(backbone_source):
        #     loc_layers += [nn.Conv2d(self.vgg[v].out_channels,mbox[k] * 4, kernel_size=3, padding=1)]
        #     conf_layers += [nn.Conv2d(self.vgg[v].out_channels,mbox[k] * num_classes, kernel_size=3, padding=1)]



def main():
    sdd_vgg = SSD_VGG_300(10, pretrained=True)

if __name__ == "__main__":
    main()