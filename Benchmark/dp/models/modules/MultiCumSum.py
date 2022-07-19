#!/usr/bin/python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxLayer(nn.Module):
    def __init__(self):
        super(SoftmaxLayer, self).__init__()

    def forward(self, x):
        """
        :param x: NxCxHxW, N is batch_size, C is channels of features
        :return: ord_label is ordinal outputs for each spatial locations , N x 1 x H x W
                 ord prob is the probability of each label, N x OrdNum x H x W
        """
        N, C, H, W = x.size()
        ord_num = C
        pdf = F.softmax(x, dim=1).view(N, C, H, W)
        return pdf


