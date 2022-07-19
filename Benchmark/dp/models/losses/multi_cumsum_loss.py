#!/usr/bin/python3
# -*- coding: utf-8 -*-

import numpy as np
import torch

import torch.nn.functional as F


class MultiCumsumLoss(object):

    def __init__(self, ord_num, alpha, beta, gamma, discretization, t0s, ctpts):
        assert ctpts.shape[0] == ord_num + 1
        self.ord_num = ord_num + 1
        # self.alpha = alpha
        # self.beta = beta
        # self.gamma = gamma
        # self.discretization = discretization
        self.bin_values = (t0s + ctpts) / 2
        self.ctpts = ctpts


    def _create_ord_label(self, gt):
        N, _, H, W = gt.shape

        ctpts = self.ctpts[None, :, None, None].repeat(N, 1, H, W).to(gt.device)
        ord_label = torch.zeros(N, self.ord_num, H, W).to(gt.device)
        ord_label[gt < ctpts] = 1

        return ord_label

    def __call__(self, pdf, gt, weight=None):
        """
        :param prob: ordinal regression probability, N x 2*Ord Num x H x W, torch.Tensor
        :param gt: depth ground truth, NXHxW, torch.Tensor
        :return: loss: loss value, torch.float
        """
        # if pdf.shape != gt.shape:
        #     pdf = F.interpolate(pdf, size=gt.shape[-2:], mode="bilinear", align_corners=True)

        N, C, H, W = pdf.shape

        gt = torch.unsqueeze(gt, dim=1)
        valid_mask = (gt > 0.).repeat(1, self.ord_num, 1, 1)
        ordinal_labels = self._create_ord_label(gt)
        cdf = torch.cumsum(pdf, dim=1)
        clipped_probs = torch.clamp(input=cdf, min=1e-7, max=1 - 1e-7)

        if weight is not None:
            weight = torch.unsqueeze(weight, dim=1)
            weight = weight.repeat(1, self.ord_num, 1, 1)
            loss = F.binary_cross_entropy(clipped_probs[valid_mask], ordinal_labels[valid_mask], weight=weight[valid_mask])
        else:
            loss = F.binary_cross_entropy(clipped_probs[valid_mask], ordinal_labels[valid_mask])

        # N, C, H, W = pdf.shape
        bin_values = self.bin_values[None, :, None, None].repeat(N, 1, H, W).to(pdf.device)
        mu = torch.sum(bin_values * pdf, dim=1).view(N, -1, H, W)
        w_variance = torch.sum(torch.pow(mu[:, :, :, :-1] - mu[:, :, :, 1:], 2))
        h_variance = torch.sum(torch.pow(mu[:, :, :-1, :] - mu[:, :, 1:, :], 2))
        tv_loss = 0.1 * (h_variance + w_variance)
        loss += tv_loss

        return loss



