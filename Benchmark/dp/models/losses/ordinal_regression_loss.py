#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 2020-05-02 18:17
@Author  : Wang Xin
@Email   : wangxin_buaa@163.com
@File    : ordinal_regression_loss.py
"""

import numpy as np
import torch

import torch.nn.functional as F


class OrdinalRegressionLoss(object):

    def __init__(self, ord_num, alpha, beta, gamma, discretization, ctpts):
        assert ctpts.shape[0] == ord_num + 1
        self.ord_num = ord_num
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.discretization = discretization
        self.ctpts = ctpts

    def _create_ord_label(self, gt):
        N, _, H, W = gt.shape
        # print("gt shape:", gt.shape)
        ord_c0 = torch.ones(N, self.ord_num, H, W).to(gt.device)
        if self.discretization == "SID":
            label = self.ord_num * (torch.log(gt+self.gamma)-np.log(self.alpha+self.gamma)) / (np.log(self.beta+self.gamma)-np.log(self.alpha+self.gamma))
        else:
            label = self.ord_num * (gt - self.alpha) / (self.beta - self.alpha)
        label = label.long()
        mask = torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False) \
            .view(1, self.ord_num, 1, 1).to(gt.device)
        mask = mask.repeat(N, 1, H, W).contiguous().long()
        mask = (mask > label)
        ord_c0[mask] = 0
        ord_c1 = 1 - ord_c0
        ord_label = torch.cat((ord_c0, ord_c1), dim=1)

        # ord_c0 = torch.zeros(N, self.ord_num, H, W).to(gt.device)
        # if self.discretization == "SID":
        #     label = self.ord_num * (torch.log(gt+self.gamma)-np.log(self.alpha+self.gamma)) / np.log(self.beta-self.alpha)
        # else:
        #     label = self.ord_num * (gt - self.alpha) / (self.beta - self.alpha)
        # label = label.long()
        # mask = torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False) \
        #     .view(1, self.ord_num, 1, 1).to(gt.device)
        # mask = mask.repeat(N, 1, H, W).contiguous().long()
        # mask = (mask < label)
        # ord_c0[mask] = 1
        # ord_c1 = 1 - ord_c0
        # ord_label = torch.cat((ord_c0, ord_c1), dim=1)
        return ord_label, mask

    def __call__(self, prob, gt, weight=None):
        valid_mask = gt > 0.
        gt = torch.unsqueeze(gt, dim=1)
        ord_label, mask = self._create_ord_label(gt)
        # print("prob shape: {}, ord label shape: {}".format(prob.shape, ord_label.shape))
        entropy = -prob * ord_label
        loss = torch.sum(entropy, dim=1)[valid_mask]
        return loss.mean()

        # N, p2, ord_num, H, W = cdf_binary.shape
        # assert p2 == 2
        # C = int(2*ord_num)
        # cdf_binary = cdf_binary.view(N, C, H, W)
        # valid_mask = gt > 0.
        # gt = torch.unsqueeze(gt, dim=1)
        # ord_label, mask = self._create_ord_label(gt)
        # clipped_probs = torch.clamp(input=cdf_binary, min=1e-7, max=1 - 1e-7)
        # entropy = -torch.log(clipped_probs) * ord_label
        # loss = torch.sum(entropy, dim=1)[valid_mask]
        # return loss.mean()

