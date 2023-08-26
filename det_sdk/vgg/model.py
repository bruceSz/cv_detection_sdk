#!/usr/bin/env python

from typing import Any
import torch

from torch import nn

from det_sdk.vgg.dataset import get_loaders
from det_sdk.framework.train_config import CIFARConfig as Config
from det_sdk.backbones.blocks import CBRBlock
from det_sdk.backbones.backbone_mgr import GlobalBackbones

# differences between VGG16 and VGG19: 16 hidden layers vs 19 hidden layers, 
# prediction heads are same: 3 layer fc.


from torch.hub import load_state_dict_from_url


model_urls = {
    "vgg16" : "https://download.pytorch.org/models/vgg16-397923af.pth",
}




class VGG_HEAD(nn.Module):
    def __init__(self, in_channel ,n_classes, *args, **kwargs) -> None:
        super(VGG_HEAD, self).__init__(*args, **kwargs)
        self.inc = in_channel
        self.fc = nn.Sequential(
            nn.Linear(in_channel , in_channel),
            nn.ReLU(inplace=True),
            # default it 0.5
            nn.Dropout(0.5),

            nn.Linear(in_channel, in_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(in_channel, n_classes)
        )

    def forward(self, x):
        x = x.view(-1, self.inc)
        return self.fc(x)
    



class VGG19(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layer1 = nn.Sequential(
            CBRBlock(3, 64, 3, 1, 1),
            CBRBlock(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )
        self.layer2 = nn.Sequential(
            CBRBlock(64, 128, 3, 1, 1),
            CBRBlock(128, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )
        self.layer3 = nn.Sequential(
            CBRBlock(128, 256, 3, 1, 1),
            CBRBlock(256, 256, 3, 1, 1),
            CBRBlock(256, 256, 3, 1, 1),
            CBRBlock(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )

        self.layer4 = nn.Sequential(
            CBRBlock(256, 512, 3, 1, 1),
            CBRBlock(512, 512, 3, 1, 1),
            CBRBlock(512, 512, 3, 1, 1),
            CBRBlock(512, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )
        self.layer5 = nn.Sequential(
            CBRBlock(512, 512, 3, 1, 1),
            CBRBlock(512, 512, 3, 1, 1),
            CBRBlock(512, 512, 3, 1, 1),
            CBRBlock(512, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )
        self.m_ = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )
    def forward(self, x):
        return self.m_(x)

vgg_cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
 

class VGG16_FT(nn.Module):
    base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512]
    def __init__(self, *args: Any, **kwds: Any) -> Any:
        super(VGG16_FT, self).__init__(*args, **kwds)
        
        self.layer1 = nn.Sequential(
            # in channes: 3, out channels: 64,  kernel size: 3,stride: 1, padding: 1
            CBRBlock(3, 64, 3, 1, 1),
            # in channes: 64, out channels: 64,kernel size: 3
            # stride: 1
            # padding: 1
            CBRBlock(64, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )

        self.layer2 = nn.Sequential(
            # in channes: 64, out channels: 128, kernel size: 3, stride: 1, padding: 1
            CBRBlock(64, 128, 3, 1, 1),
            # in channes: 128, out channels: 128, kernel size: 3, stride: 1, padding: 1
            CBRBlock(128, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )

        self.layer3 = nn.Sequential(
            # in channes: 128, out channels: 256, kernel size: 3, stride: 1, padding: 1
            CBRBlock(128, 256, 3, 1, 1),
            CBRBlock(256, 256, 3, 1, 1),
            CBRBlock(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )

        self.layer4 = nn.Sequential(
            # in channes: 256, out channels: 512, kernel size: 3, stride: 1, padding: 1
            CBRBlock(256, 512, 3, 1, 1),
            CBRBlock(512, 512, 3, 1, 1),
            CBRBlock(512, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )
        self.layer5 = nn.Sequential(
            # in channes: 512, out channels: 512, kernel size: 3, stride: 1, padding: 1
            CBRBlock(512, 512, 3, 1, 1),
            CBRBlock(512, 512, 3, 1, 1),
            CBRBlock(512, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2, ceil_mode=True)
        )

        self.conv = nn.Sequential(
            self.layer1, 
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )
    def forward(self, x):
        return self.conv(x)

@GlobalBackbones().RegisterBackbone
class VGG16_CF(nn.Module):
    __BACKBONE__ = "vgg16"
    
    WEIGHT_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"
    

    
    def __init__(self,  *args, **kwargs) -> None:
        super(VGG16_CF, self).__init__(*args, **kwargs)
        self.feature = VGG16_FT()
        self.head = VGG_HEAD(512, 10)

    def forward(self, x):
        x = self.ft_layer(x)
        print("ft shape:",x.shape)
        x = self.head(x)
        print("head shape:",x.shape)
        return x
    
@GlobalBackbones().RegisterBackbone
class VGG_OFFICIAL(nn.Module):
    __BACKBONE__ = "vgg16_official"
    WEIGHT_URL = "https://download.pytorch.org/models/vgg16-397923af.pth"
    #WEIGHT_URL =  'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth'
    
    def __init__(self,  num_classes = 1000, init_weights=True):
        super(VGG_OFFICIAL, self).__init__()

        self.features = self._make_layers("D", batch_norm=False)
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            # inc is 512
            nn.Linear(512 * 7 * 7, 4096), # 512 * 7 * 7
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096), # 4096
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes), # 4096
        )
    def _make_layers(self, type, batch_norm=False):
        layers = []
        in_channels = 3

        for v in vgg_cfgs[type]:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == "C":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        print("ft shape:",x.shape)
        x = self.avgpool(x)
        print("avg shape:",x.shape)
        x = torch.flatten(x, 1)
        print("flatten shape:",x.shape)
        x = self.classifier(x)
        print("classifier shape:",x.shape)
        return x


def print_model(model):
    print(model)

def mock_infer(model):
    
    conf = Config()
    _, val_loader = get_loaders(conf)
    
    for inputs, labels in val_loader:
        print(inputs.shape)
        #output = model(inputs)
        print(labels.shape)
        break
    input = torch.randn(1, 3, 32, 32)
    print("input is :", input.shape)
    output = model(input)    
    print(output.shape)

def vgg16(pretrained = False):
    model = VGG16_CF()
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['vgg16'], model_dir = 'model_data/')
        model.load_state_dict(state_dict)
    #----------------------------------------------------------#
    #   获取特征提取部分
    #----------------------------------------------------------#
    # features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4])
    # features = nn.Sequential(*features)
    # return features

    return model

if __name__ == "__main__":
    m = GlobalBackbones().get_backbone("vgg16_official", pretrained=True)
    mat = torch.randn(1, 3, 300, 300)
    #print(m)
    # no.22 should be layer 4_3: 38 * 38 * 512
    ft = m.features[:23]
    print("total feature length:", len(ft))
    
    out =  ft(mat)
    print("feature out shape")
    print(out.shape)
    #model = VGG16_CF()
    #print_model(model)
    