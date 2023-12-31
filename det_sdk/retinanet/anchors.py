#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import itertools

class Anchors(nn.Module):
    def __init__(self, anchor_scale = 4., pyramid_levels = [3,4,5,6,7]):
        super(Anchors, self).__init__()
        self.anchor_scale = anchor_scale
        self.pyramid_levels = pyramid_levels

        self.strides = [2 **x for x in self.pyramid_levels]
        self.scales = np.array([2 ** 0, 2**(1/3.), 2 ** (2/3.)])
        self.ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]

    def forward(self, features):
        hs  = []
        ws = []
        #####
        #### record h and w for each feature map
        #### h [75, 38, 19, 10, 5]
        #### w [75, 38, 19, 10, 5]
        ######

        for ft in features:
            _, _, h, w = ft.size()
            hs.append(h)
            ws.append(w)
        boxes_all = []

        for i, stride in enumerate(self.strides):
            boxes_level = []
            for scale, ratio in itertools.product(self.scales, self.ratios):

                base_anchor_size = self.anchor_scale * stride * scale
                anchor_size_x_2 = base_anchor_size * ratio[0] / 2.0
                anchor_size_y_2 = base_anchor_size * ratio[1] / 2.0

                # compute center of feature map

                x = np.arange(0, ws[i]) * stride + stride /2
                y = np.arange(0, hs[i]) * stride + stride /2

                xv, yv = np.meshgrid(x, y)
                xv = xv.reshape(-1)
                yv = yv.reshape(-1)

                boxes = np.vstack(yv - anchor_size_y_2, xv - anchor_size_x_2,
                                  yv + anchor_size_y_2, xv + anchor_size_x_2)
                
                boxes = np.swapaxes(boxes, 0, 1)
                boxes_level.append(np.expand_dims(boxes, axis = 1))
            
            boxes_level = np.concatenate(boxes_level, axis = 1)
            boxes_all.append(boxes_level.reshape([-1, 4]))
        anchor_boxes = np.vstack(boxes_all)
        anchor_boxes = torch.from_numpy(anchor_boxes).to(features[0].device)
        anchor_boxes = anchor_boxes.unsqueeze(0)

        return anchor_boxes

