#!/usr/bin/env python

import torch
from torch import nn
from backbones.blocks import CBR6Block, CBR6DWBlock #CBRBlock, CBlock, CLRBlock

# reference: https://github.com/mnmjh1215/mobilenet-pytorch/blob/master/models/mobilenetv2.py
class MobileNetV1(nn.Module):
    """
        * MobileNetV1 introduced in paper: 
            MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
        * It is a lightweight network for mobile vision applications.
        * Compared with VGG16, it is 32 times smaller and 27 times faster while still achieving a slightly higher accuracy.
        * It's main feature is depthwise separable convolution.
    """
    def __init__(self, n_classes,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.num_classes = n_classes

        self.modle = nn.Sequential(
            # kernel == 3, stride == 2
            CBR6Block(3, 32, 3, 2, 1),

            CBR6DWBlock(32, 64, 3, 1, 1),
            CBR6DWBlock(64, 128, 3, 2, 1),
            CBR6DWBlock(128, 128, 3, 1, 1),
            CBR6DWBlock(128, 256, 3, 2, 1),
            CBR6DWBlock(256, 256, 3, 1, 1),
            CBR6DWBlock(256, 512, 3, 2, 1),

            CBR6DWBlock(512, 512, 3, 1, 1),
            CBR6DWBlock(512, 512, 3, 1, 1),
            CBR6DWBlock(512, 512, 3, 1, 1),
            CBR6DWBlock(512, 512, 3, 1, 1),
            CBR6DWBlock(512, 512, 3, 1, 1),

            CBR6DWBlock(512, 1024, 3, 2, 1),
            CBR6DWBlock(1024, 1024, 3, 1, 1)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.modle(x)
        x = self.avg_pool(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
    

class MobileNetV2Block(nn.Module):
    """
        1 .expantion layer: 1x1 conv with out_channel = in_channel * expantion_ratio
         this layer will make channel into higher dimension, 
         (relu will loss less infomarmation for higher dimension)
        2. depthwise layer: depthwise conv
        3. 1 * 1 conv to reduce dimension to out_channel(outc).
    """
    def __init__(self, inc, outc, stride, expand_ratio,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.res_connection = (stride==1 and inc == outc)
        self.layers = nn.Sequential(
            nn.Conv2d(inc, inc*expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inc*expand_ratio),
            nn.ReLU6(inplace=True),

            nn.Conv2d(inc*expand_ratio, inc*expand_ratio, 3, stride, 1, groups=inc*expand_ratio, bias=False),
            nn.BatchNorm2d(inc*expand_ratio),
            nn.ReLU6(inplace=True),

            nn.Conv2d(inc*expand_ratio, outc, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outc)
        )

    def forward(self, x):
        s = self.layers(x)
        if self.res_connection:
            out = x + s
        else:
            out = x
        return out

class MobileNetV2(nn.Module):
    def __init__(self, n_class, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_classes = n_class

        self.layers = nn.Sequential(
            CBR6Block(3, 32, 3, 2, 1),
            MobileNetV2Block(32, 16, 1, 1),
            
            MobileNetV2Block(16, 24, 2, 6),
            MobileNetV2Block(24, 24, 1, 6),

            MobileNetV2Block(24, 32, 2, 6),
            MobileNetV2Block(32, 32, 1, 6),
            MobileNetV2Block(32, 32, 1, 6),

            MobileNetV2Block(32, 64, 2, 6),
            MobileNetV2Block(64, 64, 1, 6),
            MobileNetV2Block(64, 64, 1, 6),
            MobileNetV2Block(64, 64, 1, 6),

            MobileNetV2Block(64, 96, 1, 6),
            MobileNetV2Block(96, 96, 1, 6), 
            MobileNetV2Block(96, 96, 1, 6),

            MobileNetV2Block(96, 160, 2, 6),
            MobileNetV2Block(160, 160, 1, 6),
            MobileNetV2Block(160, 160, 1, 6),

            MobileNetV2Block(160, 320, 1, 6),

            CBR6Block(320, 1280, 1, 1, 0)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1280, n_class, kernel_size =1, stride=1, padding=0, bias=False)


    def forward(self, x):
        x = self.layers(x)
        # make output shape (batch_size, 1280, 1, 1)
        x = self.avg_pool(x)
        x = self.fc(x)
        out = x.view(-1, self.num_classes)
        return out


