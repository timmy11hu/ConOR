#!/usr/bin/python3
# -*- coding: utf-8 -*-


import numpy as np
import torch

import torch.nn.functional as F


class BinaryClsLoss(object):
    # KITTI SIGMA2=26
    def __init__(self, ord_num, t0s, t1s, use_smooth=False, alpha=25.0):
        self.ord_num = ord_num + 1
        self.t0s = t0s
        self.t1s = t1s
        self.bin_values = (t0s+t1s)/2
        self.smooth = use_smooth
        self.alpha = alpha

    def _create_class_label(self, gt):
        N, _, H, W = gt.shape
        if self.smooth:
            bv = self.bin_values[None, :, None, None].repeat(N, 1, H, W).to(gt.device)
            gt[gt < 0.] = 1.0
            delta = torch.log(gt) - torch.log(bv)
            q = torch.exp(-self.alpha*torch.abs(delta))
        else:
            t0s = self.t0s[None, :, None, None].repeat(N, 1, H, W).to(gt.device)
            t1s = self.t1s[None, :, None, None].repeat(N, 1, H, W).to(gt.device)
            q = torch.where(torch.logical_and(gt>=t0s, gt<=t1s), 1, 0)

        return q.float()

    def __call__(self, logits, gt, weight=None):
        """
        :param prob: ordinal regression probability, N x 2*Ord Num x H x W, torch.Tensor
        :param gt: depth ground truth, NXHxW, torch.Tensor
        :return: loss: loss value, torch.float
        """
        # if prob.shape != gt.shape:
        #     prob = F.interpolate(prob, size=gt.shape[-2:], mode="bilinear", align_corners=True)

        N, C, H, W = logits.shape
        prob = F.sigmoid(logits).view(N, C, H, W)

        gt = torch.unsqueeze(gt, dim=1)
        valid_mask = (gt > 0.).repeat(1, self.ord_num, 1, 1)

        class_label = self._create_class_label(gt)

        if weight is not None:
            weight = torch.unsqueeze(weight, dim=1)
            weight = weight.repeat(1, self.ord_num, 1, 1)
            loss = F.binary_cross_entropy(prob[valid_mask], class_label[valid_mask],
                                          weight=weight[valid_mask])
        else:
            loss = F.binary_cross_entropy(prob[valid_mask], class_label[valid_mask])
        return loss

