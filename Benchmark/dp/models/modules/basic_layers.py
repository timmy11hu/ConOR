#!/usr/bin/python3
# -*- coding: utf-8 -*-


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _triple


def consistent_padding_with_dilation(padding, dilation, dim=2):
    assert dim == 2 or dim == 3, 'Convolution layer only support 2D and 3D'
    if dim == 2:
        padding = _pair(padding)
        dilation = _pair(dilation)
    else:  # dim == 3
        padding = _triple(padding)
        dilation = _triple(dilation)

    padding = list(padding)
    for d in range(dim):
        padding[d] = dilation[d] if dilation[d] > 1 else padding[d]
    padding = tuple(padding)

    return padding, dilation


def conv_bn_relu(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
    padding, dilation = consistent_padding_with_dilation(padding, dilation, dim=2)
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )
    else:
        return nn.Sequential(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias),
            nn.ReLU(inplace=True),
        )
