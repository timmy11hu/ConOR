#!/usr/bin/python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalRegressionLayer(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        """
        :param x: NxCxHxW, N is batch_size, C is channels of features
        :return: ord_label is ordinal outputs for each spatial locations , N x 1 x H x W
                 ord prob is the probability of each label, N x OrdNum x H x W
        """
        N, C, H, W = x.size()
        ord_num = C // 2

        # implementation according to the paper
        # A = x[:, ::2, :, :]
        # B = x[:, 1::2, :, :]
        #
        # # A = A.reshape(N, 1, ord_num * H * W)
        # # B = B.reshape(N, 1, ord_num * H * W)
        # A = A.unsqueeze(dim=1)
        # B = B.unsqueeze(dim=1)
        # concat_feats = torch.cat((A, B), dim=1)
        #
        # if self.training:
        #     prob = F.log_softmax(concat_feats, dim=1)
        #     ord_prob = x.clone()
        #     ord_prob[:, 0::2, :, :] = prob[:, 0, :, :, :]
        #     ord_prob[:, 1::2, :, :] = prob[:, 1, :, :, :]
        #     return ord_prob
        #
        # ord_prob = F.softmax(concat_feats, dim=1)[:, 0, ::]
        # ord_label = torch.sum((ord_prob > 0.5), dim=1).reshape((N, 1, H, W))
        # return ord_prob, ord_label

        # reimplementation for fast speed.

        x = x.view(-1, 2, ord_num, H, W)
        if self.training:
            prob = F.log_softmax(x, dim=1).view(N, C, H, W)
            return prob

        # ord_prob = F.softmax(x, dim=1)[:, 0, :, :, :]
        # ord_label = torch.sum((ord_prob > 0.5), dim=1)
        # return ord_label
        pdf = torch.diff(F.softmax(x, dim=1)[:, 1, :, :, :], dim=1)
        # if torch.min(pdf) < -0.1:
        #     print("so neg")
        return pdf
        # print(torch.min(pdf))
        # if torch.min(pdf) < -0.1:
        #     print("so neg")
        # pdf = pdf.numpy()

        # prob, label = torch.max(pdf, dim=1)
        # uncertainty = 1-prob

        # prob = F.softmax(x, dim=1)
        # return prob

