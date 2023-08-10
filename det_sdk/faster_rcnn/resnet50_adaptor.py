#!/usr/bin/env python3

import math
import os
import torch
import torch.nn as nn

from torchvision.ops import RoIPool

from utils import normal_init


from backbones.resnet50 import ResNet, BottleNeck

            

GLOBAL_RESNET_PATH = "/disk2/TensorD/samples/pytorch/model_checkpoints/resnet.pth"

def resnet50(pretrained=False):
    model = ResNet(BottleNeck, [3, 4, 6, 3])
    if pretrained:
        assert(os.path.exists(GLOBAL_RESNET_PATH))
        model.load_state_dict(torch.load(GLOBAL_RESNET_PATH))
        #TODO read from local path
        #model.load_state_dict((model_urls['resnet50']))
        
    
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    classifier = list([model.layer4, model.avgpool])
    features = nn.Sequential(*features)

    classifier = nn.Sequential(*classifier)
    return features, classifier

class ResNet50ROIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(ResNet50ROIHead, self).__init__()
        self.classifier = classifier

        self.cls_loc = nn.Linear(2048, n_class * 4)
        self.score = nn.Linear(2048, n_class)

        normal_init(self.cls_loc, 0, 0.01)
        normal_init(self.score, 0, 0.01)

        self.roi_ = RoIPool((roi_size, roi_size), spatial_scale)
    
    def forward(self, x, rois, roi_indices, img_size):
        """ 
            x: feat_map from backbone(resnet50 layer3 output 1024 channels)
                n,c,h,w
            rois: rois from rpn network(treated as first stage of filter out possible roi based on anchors.)
            roi_indices:  batch index.
            img_size: original image size.
        """
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()

        rois = torch.flatten(rois, 0, 1)
        rois_indices = torch.flatten(roi_indices, 0, 1)
        rois_ft_map = torch.zeros_like(rois)

        # compute roi on ft_map scale. see:
        # w_scale = img_size[1] / x.size()[3]
        # roi_ft_map =  roi / w_scale
        rois_ft_map[:,[0,2]] = rois[:,[0,2]]/ img_size[1] * x.size()[3]
        rois_ft_map[:,[1,3]] = rois[:,[1,3]]/ img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([rois_indices[:,None], rois_ft_map], dim=1)
        # roi-pooling for feature map: x and rois.
        pool = self.roi_(x, indices_and_rois)

        #layer4 + avgpool
        fc7 = self.classifier(pool)
        # x * 2048
        fc7 = fc7.view(fc7.size(0), -1)

        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores = roi_scores.view(n , -1, roi_scores.size(1))

        return roi_cls_locs, roi_scores



        
    
if __name__ == '__main__':
    ft, clss = resnet50()
    print(ft)
