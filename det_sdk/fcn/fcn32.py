#!/usr/bin/env python3

import torch
from torch import nn

from backbones.blocks import CRBlock
class FCN32(nn.Module):

    def __init__(self, n_class = 21) -> None:
        print("init fcn with class num: {}".format(n_class))
        super(FCN32, self).__init__()

        self.cr1_1 = CRBlock(3, 64, 3, 1, 100)
        self.cr1_2 = CRBlock(64, 64, 3, 1, 1)
        # downsample 2:1 -> 1/2
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.cr2_1 = CRBlock(64, 128, 3, 1, 1)
        self.cr2_2 = CRBlock(128, 128, 3, 1, 1)

        # downsample 2:1 -> 1/4
        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.cr3_1 = CRBlock(128, 256, 3, 1, 1)
        self.cr3_2 = CRBlock(256, 256, 3, 1, 1)
        self.cr3_3 = CRBlock(256, 256, 3, 1, 1)
        # downsample 2:1 -> 1/8
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.cr4_1 = CRBlock(256, 512, 3, 1, 1)
        self.cr4_2 = CRBlock(512, 512, 3, 1, 1)
        self.cr4_3 = CRBlock(512, 512, 3, 1, 1)
        # downsample 2:1 -> 1/16
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)

        self.cr5_1 = CRBlock(512, 512, 3, 1, 1)
        self.cr5_2 = CRBlock(512, 512, 3, 1, 1)
        self.cr5_3 = CRBlock(512, 512, 3, 1, 1)
        # downsample 2:1 -> 1/32
        self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True)

        # fc6
        self.cr6_1 = CRBlock(512, 4096, 7, 1, 0)
        #self.fc6 = nn.Conv2d(512, 4096, 7, 1, 0)
        #self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.cr7_1 = CRBlock(4096, 4096, 1, 1, 0)
        #self.fc7 = nn.Conv2d(4096, 4096, 1, 1, 0)
        #self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr= nn.Conv2d(4096, n_class, 1, 1, 0)
        self.upscore = nn.ConvTranspose2d(n_class, n_class, 64, 32, 0, bias=False)


    def get_eval_cb(self):
        return None
    
    