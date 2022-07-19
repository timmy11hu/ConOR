#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 16:37
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : resnet.py
"""


import torch
import torch.nn as nn
from torch.nn import BatchNorm2d

affine_par = True


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, multi_grid=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid, dilation=dilation * multi_grid, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu_inplace(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, deep_stem=False):
        super(ResNet, self).__init__()
        self.deep_stem = deep_stem
        if deep_stem:
            self.inplanes = 128
            self.conv1 = nn.Sequential(
                conv3x3(3, 64, stride=2),
                BatchNorm2d(64),
                nn.ReLU(inplace=True),
                conv3x3(64, 64),
                BatchNorm2d(64),
                nn.ReLU(inplace=True),
                conv3x3(64, 128),
                BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )
        else:
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(self.inplanes)
        self.relu1 = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4, multi_grid=(1, 1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(planes * block.expansion, affine=affine_par))

        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, dilation=dilation, multi_grid=generate_multi_grid(i, multi_grid)))

        return nn.Sequential(*layers)

    def forward(self, x):
        x0 = self.relu1(self.bn1(self.conv1(x)))
        x1 = self.maxpool(x0)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        # feats = dict(conv_feat=x0, pool_feat=x1, layer1_feat=x2, layer2_feat=x3, layer3_feat=x4, layer4_feat=x5)
        # return feats
        return x5

    def freeze(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class ResNetBackbone(nn.Module):

    def __init__(self, batch_norm=False, pretrained=True, fix_param=True, deep_stem=True):
        super().__init__()

        self.backbone = ResNet(Bottleneck, [3, 4, 23, 3], deep_stem=deep_stem)

        if pretrained:
            ckpt_name = 'resnet101_v1c.pth' if deep_stem else 'resnet101-63fe2227.pth'
            saved_state_dict = torch.load('./dp/models/backbones/pretrained_models/'+ckpt_name, map_location="cpu")
            new_params = self.backbone.state_dict().copy()
            for i in saved_state_dict:
                i_parts = i.split('.')
                if not i_parts[0] == 'fc':
                    new_params['.'.join(i_parts[0:])] = saved_state_dict[i]

            self.backbone.load_state_dict(new_params)

            if fix_param:
                for name, param in self.backbone.named_parameters():
                    split_name = name.split('.')
                    if split_name[0] in ('conv1', 'layer1'):
                        param.requires_grad = False
                    if set(split_name).intersection(('bn1', 'bn2', 'bn3')):
                        if not batch_norm:
                            param.requires_grad = False


    def forward(self, input):
        return self.backbone(input)
