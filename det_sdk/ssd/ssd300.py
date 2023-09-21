#!/usr/bin/env python3

import torch
from torch import nn
import torch.functional as F

import torch.nn.init as init

from det_sdk.backbones.backbone_mgr import GlobalBackbones


class VGG_FT_HEAD(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(VGG_FT_HEAD, self).__init__(*args, **kwargs)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1)

        layers_ = [
            self.pool, self.conv,
            nn.ReLU(inplace=True), self.conv2, nn.ReLU(inplace=True)
        ]

        


        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def  vgg_det_head(in_ch):
    layers = []
    # block 1?
    #channel changes: 1024 -> 256 -> 512
    layers += [nn.Conv2d(in_ch, 256, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]

    # block 2?
    #channel changes: 512 -> 128 -> 256
    layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

    # block 3?
    #channel changes: 256 -> 128 -> 256
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]

    # block 4?
    #channel changes: 256 -> 128 -> 256
    layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
    layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]

    return nn.ModuleList(layers)


class L2Norm(nn.Module):
    def __init__(self, n_chan, scale,  *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_channels = n_chan
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()


    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)


    def forward(self, x):
        # n, c, h, w = x.size()
        # norm on channels level
        # so-called instance norm?
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out



class SSD_VGG_300(nn.Module):
    __BACKBONE__ = "vgg"
    def __init__(self, num_classes, pretrained = False):
        super(SSD_VGG_300, self).__init__()
        self.num_classes = num_classes
        self.vgg_ft  = GlobalBackbones().get_backbone("vgg16_official", pretrained).features
        self.ft_head = VGG_FT_HEAD()
        # output of ft_head is 1024 channels
        self.extra = vgg_det_head(1024)

        self.L2Norm = L2Norm(512, 20)
      
        mbox = [4, 6, 6, 6, 4, 4]

        loc_layers = []
        conf_layers = []
        #backbone_source = [21, -2]

        # loc and reg layers is 
        # 1. layer 4_3 of vgg_ft layer.  ()
        # 2. conv7  of vgg_ft_head layer.

        loc_layers += [nn.Conv2d(512, mbox[0] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(512, mbox[0] * num_classes, kernel_size=3, padding=1)]


        loc_layers += [nn.Conv2d(1024, mbox[1] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(1024, mbox[1] * num_classes, kernel_size=3, padding=1)]

        # block1 for extra detection layer.

        loc_layers += [nn.Conv2d(512, mbox[2] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(512, mbox[2] * num_classes, kernel_size=3, padding=1)]

        # block2 for extra detection layer.
        loc_layers += [nn.Conv2d(256, mbox[3] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, mbox[3] * num_classes, kernel_size=3, padding=1)]

        # block3 for extra detection layer.
        loc_layers += [nn.Conv2d(256, mbox[4] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, mbox[4] * num_classes, kernel_size=3, padding=1)]

        # block4 for extra detection layer.
        loc_layers += [nn.Conv2d(256, mbox[5] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(256, mbox[5] * num_classes, kernel_size=3, padding=1)]


        self.loc = nn.ModuleList(loc_layers)
        self.conf = nn.ModuleList(conf_layers)

        self.backbonename = "vgg16_official"



    def forward(self, x):
        sources = []
        loc = []
        conf = []



        x = self.vgg_ft[:23](x)

        #do l2 norm on channel for 4_3 layer(norm is large in this layer)
        s = self.L2Norm(x)
        sources.append(s)


        x = self.ft_head(x)
        sources.append(x)

        # forward through extra detection layer
        for k, v in enumerate(self.extra):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        assert(len(sources) == 6)
        assert(len(self.loc) == 6)
        assert(len(self.conf) == 6)

        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # shape of o?
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        # loc: batch_size, num_anchors, 4
        # conf: batch_size, num_anchors, num_classes
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
        )

        return output




      

        # for k,v in enumerate(backbone_source):
        #     loc_layers += [nn.Conv2d(self.vgg[v].out_channels,mbox[k] * 4, kernel_size=3, padding=1)]
        #     conf_layers += [nn.Conv2d(self.vgg[v].out_channels,mbox[k] * num_classes, kernel_size=3, padding=1)]



def main():
    sdd_vgg = SSD_VGG_300(10, pretrained=True)

if __name__ == "__main__":
    main()