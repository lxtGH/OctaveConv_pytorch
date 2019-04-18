#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiangtai Li(lxtpku@pku.edu.cn)
# Pytorch Implementation of Adaptive Conv
# This is un-offical implementation of Adaptive Conv
# PixelAwareAdaptiveBottleneck: (finished)
# DataSetAwareAdaptiveBottleneck: depends on input size (finished) which is position sensitive
# (data sensitive with learnable weights)

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 conv with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 conv"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,1), stride=stride, bias=False, padding=0)


class PixelAwareAdaptiveBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, input_size=None):
        super(PixelAwareAdaptiveBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2_3x3 = conv3x3(width, width, stride, groups)
        self.bn2 = norm_layer(width)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(width,width,kernel_size=1)
        self.fc2 = nn.Conv2d(width,width,kernel_size=1)

        self.fusion_conv1 = nn.Conv2d(width*2,width,1)
        self.fusion_conv2 = nn.Conv2d(width,width,1)


        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # conv
        out_conv3x3 = self.conv2_3x3(out)

        # gap
        size = out_conv3x3.size()[2:]
        gap = self.gap(out_conv3x3)
        gap = self.relu(self.fc1(gap))
        gap = self.fc2(gap)
        gap = F.upsample(gap, size=size,mode="bilinear", align_corners=True)

        # concat
        out_concat = torch.cat([gap,out_conv3x3],dim=1)
        out_concat = self.fusion_conv1(out_concat)
        # out_concat = self.bn_fusion1(out_concat)
        out_concat = self.relu(out_concat)
        out_concat = self.fusion_conv2(out_concat)
        # out_concat = self.bn_fusion2(out_concat)
        out_concat = self.sigmod(out_concat)

        out = out_conv3x3 + gap * out_concat

        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DataSetAwareAdaptiveBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, norm_layer=None, input_size=(224, 224)):
        super(DataSetAwareAdaptiveBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        H, W = input_size
        self.H, self.W = H, W
        if stride == 1:
            self.H, self.W = H, W
        else:
            self.H, self.W = H//2, W//2
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        # which is a little different
        self.conv1 = conv1x1(inplanes, width, stride=stride)
        self.bn1 = norm_layer(width)
        self.conv2 = AdaptiveConv(width, width, stride=1, groups=groups,size=(self.H, self.W))
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        _,_,h,w = out.size()
        assert self.H == h and self.W == w

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AdaptiveConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, size=(224, 224)):
        super(AdaptiveConv, self).__init__()

        self.conv3x3 = nn.Conv2d(in_channels, out_channels,3, stride, padding=1, dilation=dilation,groups=groups, bias=bias)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels,1, stride, padding=0, dilation=dilation,groups=groups, bias=bias)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.size = size
        self.w = nn.Parameter(torch.ones(3, 1, self.size[0], self.size[1]))
        self.softmax = nn.Softmax()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        _, _, h, w = x.size()

        weight = self.softmax(self.w)
        w1 = weight[0, :, :, :]
        w2 = weight[1, :, :, :]
        w3 = weight[2, :, :, :]

        x1 = self.conv3x3(x) # con3x3
        x2 = self.conv1x1(x) # self

        size = x1.size()[2:] # gap
        gap = self.gap(x1)
        gap = self.relu(self.fc1(gap))
        gap = self.fc2(gap)
        # gap = F.upsample(gap, size=size, mode="bilinear", align_corners=True)
        gap = F.upsample(gap,size=size,mode="nearest")

        x = w1 * x1 + w2 * x2 + w3 * gap

        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None, input_size=(224,224)):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, input_size=(input_size[0]//4,
                                                                                                 input_size[1]//4,
                                                                                                ))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer, input_size=(input_size[0]//4,
                                                                                                            input_size[1]//4
                                                                                                            ))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer, input_size=(input_size[0]//8,
                                                                                                             input_size[1]//8
                                                                                                             ))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, input_size=(input_size[0]//16,
                                                                                                             input_size[1]//16
                                                                                                             ))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, PixelAwareAdaptiveBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None, input_size=(224, 224)):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride,),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, norm_layer, input_size))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            if stride !=1:
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer,
                                    input_size=(input_size[0]//2, input_size[1]//2)))
            else:
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer, input_size=input_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(x.size())
        x = self.layer1(x)
        # print(x.size())
        x = self.layer2(x)
        # print(x.size())
        x = self.layer3(x)
        # print(x.size())
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



def PixelAwareResnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(PixelAwareAdaptiveBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def PixelAwareResnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(PixelAwareAdaptiveBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def PixelAwareResnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(PixelAwareAdaptiveBottleneck, [3, 8, 36, 3], **kwargs)
    return model


def DataSetAwareResnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(DataSetAwareAdaptiveBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def DataSetAwareResnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(DataSetAwareAdaptiveBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def DataSetAwareResnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(DataSetAwareAdaptiveBottleneck, [3, 4, 23, 3], **kwargs)
    return model


if __name__ == '__main__':
    model = PixelAwareResnet152().cuda()
    i = torch.Tensor(2, 3, 224, 224).cuda()
    y = model(i)
    print(y.size())