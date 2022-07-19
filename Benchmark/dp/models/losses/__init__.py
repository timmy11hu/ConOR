#!/usr/bin/python3
# -*- coding: utf-8 -*-
from dp.models.losses.multi_cumsum_loss import MultiCumsumLoss
from dp.models.losses.gaussian_nll import GaussianNLL
from dp.models.losses.log_gaussian_nll import LogGaussianNLL
from dp.models.losses.binary_cls_loss import BinaryClsLoss
from dp.models.losses.cross_entropy_loss import MultinomialLoss
from dp.models.losses.ordinal_regression_loss import OrdinalRegressionLoss

def _get_loss_func(loss_type,ord_num, alpha, beta, gamma, discretization, t0s, t1s):
    if loss_type == "conor":
        return MultiCumsumLoss(ord_num, alpha, beta, gamma, discretization, t0s, t1s)
    elif loss_type == "bc":
        return BinaryClsLoss(ord_num=ord_num, t0s=t0s, t1s=t1s)
    elif loss_type == "gl":
        return GaussianNLL()
    elif loss_type == "lgl":
        return LogGaussianNLL()
    elif loss_type == "mcc":
        return MultinomialLoss(ord_num=ord_num, t0s=t0s, t1s=t1s)
    elif loss_type == "or":
        return OrdinalRegressionLoss(ord_num, alpha, beta, gamma, discretization, t1s)
    else:
        raise NotImplementedError
