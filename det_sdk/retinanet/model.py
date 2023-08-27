#!/usr/bin/env python3

import numpy as np
import torch
from torch import nn
from backbones.backbone_mgr import GlobalBackbones
from common.utils import modules_init
import colorsys

from anchors import Anchors

class RegressionHead(nn.Module):
    def __init__(self, ft_size, n_anchors = 9, out_ft=256) -> None:
        super(RegressionHead, self).__init__()
        self.n_anchors = n_anchors
        self.ft_size = ft_size
        self.ft_channels = out_ft
        # default stride == 1
        self.conv1 = nn.Conv2d(self.ft_size, self.ft_channels, kernel_size=3,  padding=1)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(self.ft_channels, self.ft_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(self.ft_channels, self.ft_channels, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(self.ft_channels, self.ft_channels, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(self.ft_channels, self.n_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.output(out)
        # nchw -> nhwc
        out = out.permute(0, 2, 3, 1)
        #make memory format contiguous and view as (n, -1, 4), n is batch_size.
        return out.contigous().view(out.shape[0], -1, 4)
        

class ClassificationHead(nn.Module):
    def __init__(self, ft_size, num_anchors = 9, n_class=80, out_ft = 256):
        self.num_classes = n_class
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(ft_size, out_ft, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_ft, out_ft, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(out_ft, out_ft, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(out_ft, out_ft, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.out = nn.Conv2d(out_ft, self.num_anchors * self.num_classes, kernel_size=3, padding=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        out = self.conv3(out)
        out = self.act3(out)
        out = self.conv4(out)
        out = self.act4(out)
        out = self.out(out)
        out = self.out_act(out)

        # same as RegressionHead, change to nhwc.
        # c is num_classes * num_anchors
        out1 = out.permute(0, 2, 3, 1)
        
        bs, h, w, c = out1.shape
        out2 = out1.view(bs, h, w, self.num_anchors, self.num_classes)
        # view as (n, -1, num_classes)
        return out2.contiguous().view(out2.shape[0], -1, self.num_classes)


class RetinaAnchors(nn.Module):
    def __init__(self, anchor_scale = 4., pyramid_level = [3,4,5,6,7]):
        super(RetinaAnchors, self).__init__()
        self.anchor_scale = anchor_scale
        self.pyramid_level = pyramid_level
        # strides: [ 8, 16, 32, 64, 128 ]
        self.strides = [2 ** x for x in self.pyramid_level]
        self.scales = np.array([2**(x/3.) for x in range(3)])
        self.ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]

class Retinanet(nn.Module):
    def __init__(self, n_classes, backbone_name='resnet18', pretrained=False) -> None:
        super(Retinanet, self).__init__()
        self.n_classes = n_classes
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.fpn_bb_name = "PyramidFeatures3RetinaNet"
        self.backbone = GlobalBackbones.get_backbone(self.backbone_name, pretrained=self.pretrained)
        #   用于选择所使用的模型的版本
        #   0、1、2、3、4
        #   resnet18, resnet34, resnet50, resnet101, resnet152
        self.fpn_sizes = {
            "resnet18": [128, 128, 512],
            "resnet34": [128, 256, 512],
            "resnet50": [512, 1024, 2048],
            "resnet101": [512, 1024, 2048],
            "resnet152": [512, 1024, 2048],
        }[self.backbone_name]

        self.ft_size = 256

        # default feature pyramid network output channel num is 256
        self.fpn= GlobalBackbones.get_backbone(self.fpn_bb_name, pretrained=False)

        self.reg_head = RegressionHead(self.ft_size)
        self.cls_head = ClassificationHead(self.ft_size, self.n_classes)
        self.anchors = Anchors()
        self.init_weight()

    def init_weight(self, fp16=False):
        if not self.pretrained:
            modules_init(self.modules())
        prior = 0.01
        self.cls_head.output.weight.data.fill_(0)
        if fp16:
            self.cls_head.output.bias.data.fill_(-2.9)
        else:
            self.cls_head.output.bias.data.fill_(-np.log((1.0 - prior) / prior).astype(np.float16))

        self.reg_head.output.weight.data.fill_(0)
        self.reg_head.output.bias.data.fill_(0)


    def forward(self, inputs):

        ft1, ft2, ft3 = self.backbone(inputs)

        det_fts = self.fpn([ft1, ft2, ft3])

        # result shape: (n, -1, 4)
        reg = torch.cat([self.reg_head(x) for x in det_fts], dim=1)
        # result shape: (n, -1, num_classes)
        clsses = torch.cat([self.cls_head(x) for x in det_fts], dim=1)

        anchors = self.anchors(det_fts)
        return det_fts, reg, clsses, anchors
        

            


class RetinanetConfig(object):
    """
        [resnet18, resnet34, resnet50, resnet101, resnet152]
    """
    def __init__(self, config, pretrained=False) -> None:
        h, w = config.input_shape
        self.input_shape = [h, w]
        self.n_classes = config.num_class
        self.class_names = config.class_names
        # 0.3
        self.num_iou = config.num_iou
        self.backbone_name = config.backbone_name
        self.conf = config.confidence_threshold

        self.hsv_list = [(x/self.n_classes, 1, 1) for x in range(self.n_classes)]
        self.colors = list(map(lambda x:colorsys.hsv_to_rgb(*x), self.hsv_list))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))



