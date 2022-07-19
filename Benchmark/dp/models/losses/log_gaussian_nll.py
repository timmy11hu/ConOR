#!/usr/bin/python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F

class LogGaussianNLL(object):
    def __init__(self):
        self.eps = 1e-6
        self.loss_fnc = nn.GaussianNLLLoss(eps=self.eps)
    def __call__(self, mle, gt, weight=None):
        N, C, H, W = mle.size()
        assert C == 2
        valid_mask = (gt > 0.)
        # mle = torch.log(1 + torch.exp(mle))
        mu_z = torch.clamp(mle[:, 0, :, :], min=0)
        sigma2_z = torch.clamp(mle[:, 1, :, :], min=0)

        if weight is not None:
            var = torch.clamp(sigma2_z, min=self.eps)
            loss = torch.mean(
                weight[valid_mask] * (0.5 *
                          (torch.log(var[valid_mask])
                           + torch.square(mu_z[valid_mask] - torch.log(gt[valid_mask]+1)) / var[valid_mask]))
            )
        else:
            loss = self.loss_fnc(mu_z[valid_mask], torch.log(gt[valid_mask]+1), sigma2_z[valid_mask])
        return loss