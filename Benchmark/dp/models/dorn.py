#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch.nn as nn

from dp.models.backbones.resnet import ResNetBackbone
from dp.models.backbones.vgg import vgg16
from dp.models.modules.SceneUnderstandingModule import SceneUnderstandingModule
from dp.models.modules.OrdinalRegression import OrdinalRegressionLayer
from dp.models.modules.MultiCumSum import SoftmaxLayer
from dp.models.modules.BinaryClassification import BinaryClsLayer


class DepthPredModel(nn.Module):

    def __init__(self, loss_type="multi_cumsum", ord_num=80, alpha=1.0, gamma=0.0, beta=80.0,
                 input_size=(385, 513), kernel_size=4, pyramid=[6, 12, 18], batch_norm=False,
                 discretization="SID", backbone='ResNet101', pretrained=True, fix_param=True,
                 dropout_prob=0.5):
        super().__init__()
        assert len(input_size) == 2
        assert isinstance(kernel_size, int)
        self.loss_types = loss_type
        self.ord_num = ord_num
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.discretization = discretization

        if backbone == 'ResNet101':
            self.backbone = ResNetBackbone(batch_norm=batch_norm, pretrained=pretrained, fix_param=fix_param)
        elif backbone == 'VGG16':
            self.backbone = vgg16(pretrained=pretrained)
        else:
            raise NotImplementedError

        if loss_type == "conor" or loss_type == 'mcc':
            last_layer_width = int(ord_num * 1 + 1)
            last_layer = SoftmaxLayer()
        elif loss_type == "bc":
            last_layer_width = int(ord_num * 1 + 1)
            last_layer = nn.Sequential()
        elif loss_type == "gl" or loss_type == "lgl":
            last_layer_width = 2
            last_layer = nn.Sequential()
        elif loss_type == 'or':
            last_layer_width = int((ord_num+1) * 2)
            last_layer = OrdinalRegressionLayer()
        else:
            raise NotImplementedError

        self.SceneUnderstandingModule = SceneUnderstandingModule(backbone=backbone,
                                                                 ord_num=ord_num,
                                                                 size=input_size,
                                                                 kernel_size=kernel_size,
                                                                 pyramid=pyramid,
                                                                 batch_norm=batch_norm,
                                                                 last_layer_width=last_layer_width,
                                                                 dropout_prob=dropout_prob)
        self.regression_layer = last_layer

    def optimizer_params(self):
        group_params = [{"params": filter(lambda p: p.requires_grad, self.backbone.parameters()), "lr": 1.0},
                        {"params": filter(lambda p: p.requires_grad, self.SceneUnderstandingModule.parameters()),
                         "lr": 1.0}]
        return group_params

    def forward(self, image):
        """
        :param image: RGB image, torch.Tensor, Nx3xHxW
        :param target: ground truth depth, torch.Tensor, NxHxW
        :return: output: if training, return loss, torch.Float,
                         else return {"target": depth, "prob": prob, "label": label},
                         depth: predicted depth, torch.Tensor, NxHxW
                         prob: probability of each label, torch.Tensor, NxCxHxW, C is number of label
                         label: predicted label, torch.Tensor, NxHxW
        """
        feat = self.backbone(image)
        feat = self.SceneUnderstandingModule(feat)
        return self.regression_layer(feat)
