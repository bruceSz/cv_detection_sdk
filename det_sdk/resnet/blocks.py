#!/usr/bin/env python3


import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary


class BN_Conv2d(nn.Module):
    """
    Batch Normalization + Conv2d
    """

    def __init__(self, in_channels: object, out_channels: object, kernel_size: object, stride: object, 
                 padding: object,dilation =1, groups: object = 1, bias: object = False, act=True):
        super(BN_Conv2d, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias),
                            nn.BatchNorm2d(out_channels)]
        
        if act:
            layers.append(nn.ReLU(inplace=True))
        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        return self.seq(x)

class BasicBlock(nn.Module):
    """
    basic building block for resnet-18, restnet-34
    """

    message = "basic"

    def __init__(self, in_channels: object, out_channels: object, 
                 strides, is_se=False):
        
        super(BasicBlock, self).__init__()

        self.is_se = is_se
        self.conv1 = BN_Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1)
        self.conv2 = BN_Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.is_se:
            self.se = SE(out_channels, 16)

        self.short_cut = nn.Sequential()
        if not strides == 1:
            self.short_cut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, padding=0, bias=False),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.is_se:
            coeff = self.se(out)
            out = out *  coeff
        out = out + self.short_cut(x)
        return F.relu(out)
    

class BottleNeck(nn.Module):
    """
        BottleNeck for resnet-50, ResNet-101, ResNet-152
    """
    message = "bottleneck"

    def __init__(self, in_channels: object, out_channels: object, strides, is_se = False):
        super(BottleNeck, self).__init__()
        self.is_se = is_se
        self.conv1 = BN_Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = BN_Conv2d(out_channels, out_channels, kernel_size=3, stride=strides, padding=1, bias=False)
        self.conv3 = BN_Conv2d(out_channels, out_channels* 4, kernel_size=1, stride=1, padding=0, bias=False, act=False)
        if self.is_se:
            self.se = SE(out_channels * 4, 16)

        self.short_cut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels*4, kernel_size=1, stride=strides, padding=0, bias=False),
            nn.BatchNorm2d(out_channels*4)
        )


    def forward(self, x):

        out = self.conv1(x)
        print("conv1 input shape: ", x.shape)
        print("conv1 output shape: ", out.shape)
        out = self.conv2(out)
        print("conv2 output shape: ", out.shape)
        out = self.conv3(out)
        print("conv3 output shape: ", out.shape)
        if self.is_se:
            coeff = self.se(out)
            out = out *  coeff
        out = out + self.short_cut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, groups, num_classes=1000) -> object:
        super(ResNet, self).__init__()
        self.channels = 64
        self.block = block

        self.conv1 = nn.Conv2d(3, self.channels, kernel_size=7,  stride=2, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(self.channels)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # output shape: batch * 64 * 56 * 56

        self.conv2_x = self._make_conv_x(channels=64, blocks=groups[0], strides=1,index=2)
        self.conv3_x = self._make_conv_x(channels=128, blocks=groups[1], strides=2, index=3)
        self.conv4_x = self._make_conv_x(channels=256, blocks=groups[2], strides=2, index=4)
        self.conv5_x = self._make_conv_x(channels=512, blocks=groups[3], strides=2, index=5)

        self.pool2 = nn.AvgPool2d(7)

        patches = 512 if self.block.message == "basic"  else 512 * 4

        self.fc = nn.Linear(patches, num_classes)


    def _make_conv_x(self, channels, blocks, strides, index):
        list_strides = [strides] + [1] * (blocks-1)
        conv_x = nn.Sequential()
        for i in range(len(list_strides)):
            layer_name = str('block_%d_%d'%(index, i))
            conv_x.add_module(layer_name, self.block(self.channels, channels, list_strides[i]))
            self.channels = channels if self.block.message == 'basic' else channels * 4
        return conv_x
    
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn(out))
        out = self.pool1(out)

        print("shape of pool out: ", out.shape)
        # out.shape[0] is batch size.

        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.pool2(out)
        out = out.view(out.size(0), -1)
        out = F.softmax(self.fc(out))
        return out
        

        

def ResNet18(num_c=100):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_c)

def ResNet34(num_c=100):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_c)

def ResNet50(num_c=100):
    return ResNet(BottleNeck, [3,4,6,3], num_classes=num_c)

def ResNet101(num_c=100):
    return ResNet(BottleNeck, [3,4,23,3], num_classes=num_c)

def ResNet152(num_c=100):
    return ResNet(block=BottleNeck, groups=[3,8,36,3], num_classes=num_c)


def test():
    net = ResNet152()
    #print(net)
    net.cuda()
    summary(net, (3,224, 224))

test()
