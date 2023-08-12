#!/usr/bin/env python3


from torch import nn
import math
from torch.hub import load_state_dict_from_url

model_urls = {
    'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
    'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
    'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1,stride=stride , bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            res = self.downsample(x)
        out += res
        out = self.relu(out)

        return out
    

#-----------------------------------------------------------------#
#   使用Renset50作为主干特征提取网络，最终会获得一个
#   16x16x2048的有效特征层
#-----------------------------------------------------------------#
class ResNet(nn.Module):
    """
        refer: https://blog.csdn.net/zjc910997316/article/details/102912175
    """
    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 64
        super(ResNet, self).__init__()
        # h, w, c-> h/2, w/2, 64, etc:
        # 512,512,3 -> 256,256,64
        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # h, w, 64 -> h/2, w/2, 64, etc:
        # 256x256x64 -> 128x128x64,
        # 300* 300 * 64 -> 150 * 150 * 64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change

        # expand channel by 4 times:
        # 128x128x64 -> 128x128x256,
        # 150* 150 * 64 -> 150 * 150 * 256(64 * 4)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # h, w , c -> h/2, w/2, c*2
        # 150* 150 * 256 -> 75 * 75 * 512 (128 * 4)
        # 128x128x256 -> 64x64x512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # h, w , c -> h/2, w/2, c*2
        # 64x64x512 -> 32x32x1024
        # 75* 75 * 512 -> 38 * 38 * 1024 (256 * 4)
        # this is a shared feature map layer.
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  

        # h, w, c -> h/2, w/2, c*2
        # 38 * 38 * 1024 -> 19 * 19 * 2048
        # 32x32x1024 -> 16x16x2048 [centernet]
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool =  nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)


        for m in self.modules():
            # init with MSRC initializer/kaiming
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet50(pretrained = False):
    model = ResNet(BottleNeck, [3, 4, 6, 3])
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['resnet50'], model_dir = 'model_data/')
        model.load_state_dict(state_dict)
    #----------------------------------------------------------#
    #   获取特征提取部分
    #----------------------------------------------------------#
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4])
    features = nn.Sequential(*features)
    return features
