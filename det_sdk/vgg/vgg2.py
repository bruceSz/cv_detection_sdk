#!/usr/bin/env python3

import torch
from torch import nn
from torch.hub import load_state_dict_from_url

out_chan_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512]

url = "https://download.pytorch.org/models/vgg16-397923af.pth"

def build_vgg_v2(base, pretrained=False):
    layers = []
    in_channels = 3

    for v in base:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)

    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [
        pool5, conv6,
        nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)
    ]

    
    

    model = nn.ModuleList(layers)

    if pretrained:
        #state_dict = 
        state_dict = load_state_dict_from_url(url, model_dir = 'model_data/')
        
        #state_dict = {k.replace("features.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    return model


if __name__ == "__main__":
    model = build_vgg_v2(out_chan_cfg, pretrained=True)
    for i, layer in enumerate(model):
        print(i, layer)


