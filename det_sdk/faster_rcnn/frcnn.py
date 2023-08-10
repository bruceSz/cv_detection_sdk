#!/usr/bin/env python3

import torch
import torch.nn as nn
import faster_rcnn.resnet50_adaptor as resnet50_adaptor
from rpn import RegionProposalNetwork
from faster_rcnn.resnet50_adaptor import ResNet50ROIHead

class FasterRCNN(nn.Module):
    def __init__(self, n_classes,
                mode = 'train',
                feat_stride = 16,
                anchor_scales = [8, 16, 32],
                ratios = [0.5, 1, 2],
                backbone = 'resnet50',
                pretrained=False):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride

        # only one backbone: resnet50

        self.extractor, classifier = resnet50_adaptor.resnet50(pretrained)

        self.rpn = RegionProposalNetwork(1024, 512,
        anchor_scales = anchor_scales,
        ratios = ratios,
        feat_stride = feat_stride)

        self.head = ResNet50ROIHead(
            n_class = n_classes + 1,
            roi_size = 14, 
            spatial_scale = 1,
            classifier = classifier
        )
    

    def forward(self, x, scale=1.0, mode=  "forward"):

        if mode == "forward":
            print("calling forward.")
            # x is n,c,h,w
            img_size = x.shape[2:]

            ft_map = self.extractor.forward(x)
            # rpn_locs, rpn_scores, rois, rois_indices(batch index), anchors
            # rois here is first stage result (pass nms check, top_n check, roi size check.)
            _,_, rois, rois_indices, _ = self.rpn.forward(ft_map, img_size, scale)

            roi_cls_locs, roi_scores = self.head.forward(ft_map, rois, rois_indices,img_size)

            return roi_cls_locs, roi_scores,rois, rois_indices
        
        elif mode == "extract":
            ft_map = self.extractor.forward(x)
            return ft_map
        elif mode == "rpn":
            assert(type(x) == type([]))
            #ft_map = self.extractor(x)
            ft_map, img_size = x

            print("calling rpn.")
            rpn_locs, rpn_scores, rois, rois_indices, anchor = self.rpn.forward(ft_map, img_size, scale)
            return rpn_locs, rpn_scores, rois, rois_indices, anchor

        elif mode == "head":
            ft_map, rois, rois_indices, img_size = x
            roi_cls_locs, roi_scores = self.head.forward(ft_map, rois, rois_indices,img_size)

            return roi_cls_locs, roi_scores
        else:
            raise NotImplementedError
        

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()






