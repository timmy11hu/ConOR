import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultinomialLoss(object):
    def __init__(self, ord_num, t0s, t1s):
        self.ord_num = ord_num + 1
        self.t0s = t0s
        self.t1s = t1s

    def _create_class_label(self, gt, onehot=False):
        N, _, H, W = gt.shape
        t0s = self.t0s[None, :, None, None].repeat(N, 1, H, W).to(gt.device)
        t1s = self.t1s[None, :, None, None].repeat(N, 1, H, W).to(gt.device)
        y_label = torch.where(torch.logical_and(gt>=t0s, gt<=t1s), 1, 0)
        if not onehot:
            y_label = torch.argmax(y_label, dim=1)
            y_label = torch.unsqueeze(y_label, dim=1).view(N, 1, H, W)
        return y_label

    def __call__(self, prob, gt, weight=None):
        """
        :param prob: ordinal regression probability, N x 2*Ord Num x H x W, torch.Tensor
        :param gt: depth ground truth, NXHxW, torch.Tensor
        :return: loss: loss value, torch.float
        """
        if prob.shape != gt.shape:
            prob = F.interpolate(prob, size=gt.shape[-2:], mode="bilinear", align_corners=True)

        N, C, H, W = prob.shape

        gt = torch.unsqueeze(gt, dim=1)
        valid_mask_label = (gt > 0.)
        valid_mask_probs = (gt > 0.).repeat(1, self.ord_num, 1, 1)

        x = torch.clamp(input=prob[valid_mask_probs], min=1e-7, max=1 - 1e-7)
        z = torch.t(x.view(self.ord_num, -1))

        if weight is None:
            class_label = self._create_class_label(gt, onehot=False)
            y = class_label[valid_mask_label]
            loss = F.nll_loss(input=torch.log(z), target=y)
        else:
            class_label = self._create_class_label(gt, onehot=True)
            y = class_label[valid_mask_probs]
            y = y.view(-1, self.ord_num)
            weight = torch.unsqueeze(weight, dim=1)
            w = weight[valid_mask_label]
            loss = -torch.mean(torch.sum(torch.log(z) * y, dim=1) * w)
        return loss

