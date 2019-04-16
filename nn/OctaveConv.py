#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Xiangtai(lxtpku@pku.edu.cn)
# Pytorch Implementation of Octave Conv Operation

import torch
import torch.nn as nn
import torch.nn.functional as F

up_kwargs = {'mode': 'bilinear', 'align_corners': True}


class OctaveConv(nn.Module):
    def __init__(self, kernel, in_channels, out_channels, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = up_kwargs):
        super(OctaveConv, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel[0], kernel[1]))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = torch.zeros(out_channels).cuda()
        self.up_kwargs = up_kwargs
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out

    def forward(self, x):
        X_h, X_l = x

        if self.stride ==2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)
        X_l2h = F.upsample(X_l,scale_factor=2, **self.up_kwargs)

        end_h_x = int(self.in_channels*(1- self.alpha_in))
        end_h_y = int(self.out_channels*(1- self.alpha_out))


        X_h2h = F.conv2d(X_h, self.weights[0:end_h_y, 0:end_h_x, :,:], self.bias[0:end_h_y], 1,
                        self.padding, self.dilation, self.groups)

        X_l2l = F.conv2d(X_l, self.weights[end_h_y:, end_h_x:, :,:], self.bias[0:end_h_y], 1,
                        self.padding, self.dilation, self.groups)

        X_h2l = F.conv2d(X_h2l, self.weights[end_h_y:, 0: end_h_x, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)

        X_l2h = F.conv2d(X_l2h, self.weights[0:end_h_y, end_h_x:, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)

        X_h = X_h2h + X_l2h
        X_l = X_l2l + X_h2l

        return X_h, X_l



class FirstOctaveConv(nn.Module):
    def __init__(self, kernel, in_channels, out_channels, alpha_in=0, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = up_kwargs):
        super(FirstOctaveConv, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel[0], kernel[1]))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = torch.zeros(out_channels).cuda()
        self.up_kwargs = up_kwargs
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out

    def forward(self, x):

        if self.stride ==2:
            x = self.h2g_pool(x)

        X_h2l = self.h2g_pool(x)
        X_h = x

        end_h_x = int(self.in_channels*(1- self.alpha_in))
        end_h_y = int(self.out_channels*(1- self.alpha_out))

        X_h2h = F.conv2d(X_h, self.weights[0:end_h_y, 0:end_h_x, :,:], self.bias[0:end_h_y], 1,
                        self.padding, self.dilation, self.groups)

        X_h2l = F.conv2d(X_h2l, self.weights[end_h_y:, 0: end_h_x, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)

        X_h = X_h2h
        X_l = X_h2l

        return X_h, X_l


class LastOctaveConv(nn.Module):
    def __init__(self, kernel, in_channels, out_channels, alpha_in=0.5, alpha_out=0, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = up_kwargs):
        super(LastOctaveConv, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel[0], kernel[1]))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = torch.zeros(out_channels).cuda()
        self.up_kwargs = up_kwargs
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out

    def forward(self, x):
        X_h, X_l = x

        if self.stride ==2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_l2h = F.upsample(X_l,scale_factor=2, **self.up_kwargs)

        end_h_x = int(self.in_channels*(1- self.alpha_in))
        end_h_y = int(self.out_channels*(1- self.alpha_out))

        X_h2h = F.conv2d(X_h, self.weights[0:end_h_y, 0:end_h_x, :,:], self.bias[:], 1,
                        self.padding, self.dilation, self.groups)

        X_l2h = F.conv2d(X_l2h, self.weights[0:end_h_y, end_h_x:, :,:], self.bias[:], 1,
                        self.padding, self.dilation, self.groups)

        X_h = X_h2h + X_l2h

        return X_h

if __name__ == '__main__':
    # nn.Conv2d
    low = torch.Tensor(1,128, 16, 16).cuda()
    high = torch.Tensor(1,128, 32, 32).cuda()
    OCconv = OctaveConv(kernel=(3,3),in_channels=256,out_channels=256,bias=False,stride=2).cuda()
    i = high,low
    x_out,y_out = OCconv(i)
    print(x_out.size())
    print(y_out.size())

    i = torch.Tensor(1,3,512,512).cuda()
    FOCconv = FirstOctaveConv(kernel=(3,3), in_channels=3, out_channels=128).cuda()
    x_out, y_out = FOCconv(i)

    LOCconv = LastOctaveConv(kernel=(3,3), in_channels=256, out_channels=128).cuda()
    i = high, low
    out = LOCconv(i)
    print(out.size())