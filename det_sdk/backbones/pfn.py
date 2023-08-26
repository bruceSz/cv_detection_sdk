#!/usr/bin/env python3

from torch import nn
from torch.functional import F

from backbones.backbone_mgr import GlobalBackbones

@GlobalBackbones.RegisterBackbone
class PyramidFeatures3(nn.Module):
    __BACKBONE__ = "PyramidFeatures3RetinaNet"
    WEIGHT_URL = ""

    def __init__(self, ft1_size, ft2_size, ft3_size, ft_size=256):
        super(PyramidFeatures3, self).__init__()

        self.p3_1 = nn.Conv2d(ft3_size, ft_size, kernel_size=1, stride=1, padding=0)
        self.p3_2 = nn.Conv2d(ft_size, ft_size, kernel_size=3, stride=1, padding=1)

        self.p2_1 = nn.Conv2d(ft2_size, ft_size, kernel_size=1, stride=1, padding=0)
        self.p2_2 = nn.Conv2d(ft_size, ft_size, kernel_size=3, stride=1, padding=1)

        self.p1_1 = nn.Conv2d(ft1_size, ft_size, kernel_size=1, stride=1, padding=0)
        self.p1_2 = nn.Conv2d(ft_size, ft_size, kernel_size=3, stride=1, padding=1)

        self.p4 = nn.Conv2d(ft3_size, ft_size, kernel_size=3, stride=2, padding=1)

        self.p5_1 = nn.ReLU(inplace=True)
        self.p5_2 = nn.Conv2d(ft_size, ft_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        ft1, ft2, ft3 = inputs

        _, _, h1, w1 = ft1.size()
        _, _, h2, w2 = ft2.size()

        # convert to same feature num.
        # 75, 75, 512 -> 75, 75, 256
        p1_x = self.p1_1(ft1)
        # 38, 38, 512 -> 38, 38, 256
        p2_x = self.p2_1(ft2)
        # 19, 19, 512 -> 19, 19, 256
        p3_x = self.p3_1(ft3)

        # upsample and interpolate.
        p3_up_x = F.interpolate(p3_x, size= (h2, w2))
        p2_x = p3_up_x + p2_x
        p2_up_x = F.interpolate(p2_x, size=(h1, w1))
        p1_x = p1_x + p2_up_x

        # do another round conv to reduce upsample effect.
        p3_x = self.p3_2(p3_x)
        p2_x = self.p2_2(p2_x)
        p1_x = self.p1_2(p1_x)

        p4_x = self.p4(ft3)
        p5_x = self.p5_1(p4_x)
        p5_x = self.p5_2(p5_x)

        return [p1_x, p2_x, p3_x, p4_x, p5_x]






