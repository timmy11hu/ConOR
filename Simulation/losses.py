import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from evaluation import evaluate_monotonicity, cal_var
from utils import find_cutpoints, create_nominal_label, create_ordinal_label


class MultinomialLoss(object):
    def __init__(self, num_bin, alpha, beta, discretization):
        self.num_bin = num_bin
        self.alpha = alpha
        self.beta = beta
        self.discretization = discretization

    def __call__(self, probs, gts):
        # probs = probs.permute(0, 3, 1, 2)
        clipped_probs = torch.clamp(input=probs, min=1e-7, max=1 - 1e-7)
        log_probs = torch.log(clipped_probs)

        # labels: k class label
        nominal_labels = create_nominal_label(gts, self.discretization, self.num_bin, self.alpha, self.beta)
        loss = F.nll_loss(log_probs, nominal_labels)

        return loss


class OrdinalRegressionLoss(object):
    def __init__(self, num_bin, alpha, beta, discretization):
        self.num_bin = num_bin
        self.alpha = alpha
        self.beta = beta
        self.gamma = 1.0
        self.discretization = discretization
        self.cutpoints = find_cutpoints(alpha=self.alpha, beta=self.beta, num_bin=self.num_bin,
                                        discretization=self.discretization)

    def __call__(self, cdfs, gt, weights=None):
        N, _, ord_num = cdfs.shape
        cdfs = cdfs.view(N, 2*ord_num)
        cdfs = torch.clamp(input=cdfs, min=1e-7, max=1 - 1e-7)
        ord_c1 = create_ordinal_label(gt, self.discretization, self.num_bin, self.alpha, self.beta)
        ord_c0 = 1 - ord_c1
        ord_label = torch.cat((ord_c0, ord_c1), dim=1)

        entropy = -torch.log(cdfs) * ord_label
        loss = torch.sum(entropy, dim=1)
        return loss.mean()


class MultiBinaryLoss(object):
    def __init__(self, num_bin, alpha, beta, discretization):
        self.num_bin = num_bin
        self.alpha = alpha
        self.beta = beta
        self.gamma = 1.0
        self.discretization = discretization
        self.cutpoints = find_cutpoints(alpha=self.alpha, beta=self.beta, num_bin=self.num_bin,
                                        discretization=self.discretization)

    def __call__(self, pdf, gt, weights=None):
        cdfs = torch.cumsum(pdf, dim=1)[:, :-1]
        ordinal_labels = create_ordinal_label(gt, self.discretization, self.num_bin, self.alpha, self.beta)
        loss = 0
        clipped_probs = torch.clamp(input=cdfs, min=1e-7, max=1-1e-7)
        if weights is not None:
            loss += -torch.mean(torch.log(clipped_probs) * ordinal_labels * torch.t(weights.repeat(self.num_bin, 1)))
            loss += -torch.mean(torch.log(1 - clipped_probs) * (1 - ordinal_labels) * torch.t(weights.repeat(self.num_bin, 1)))
        else:
            loss += -torch.mean(torch.log(clipped_probs) * ordinal_labels)
            loss += -torch.mean(torch.log(1 - clipped_probs) * (1 - ordinal_labels))
        return loss


class BinaryClsLoss(object):
    def __init__(self, num_bin, alpha, beta, discretization):
        self.num_bin = num_bin + 1
        self.alpha = alpha
        self.beta = beta
        self.gamma = 0.0
        self.discretization = discretization
        self.cutpoints = find_cutpoints(alpha=alpha, beta=beta, num_bin=num_bin,
                                        discretization=discretization)
        self.t0s = torch.cat((torch.tensor(alpha).view(-1), self.cutpoints), dim=0)
        self.t1s = torch.cat((self.cutpoints, torch.tensor(beta).view(-1)), dim=0)
        self.bin_values = (self.t0s + self.t1s) / 2

    def __call__(self, logit, gt, weights=None):
        N = len(gt)
        t0s = self.t0s[None, :].repeat(N, 1).to(gt.device)
        t1s = self.t1s[None, :].repeat(N, 1).to(gt.device)
        prob = torch.sigmoid(logit)
        # bin_labels = create_nominal_label(gt, self.discretization, self.num_bin, self.alpha, self.beta)
        gt = torch.unsqueeze(gt, dim=1)
        binary_labels = torch.where(torch.logical_and(gt >= t0s, gt <= t1s), 1, 0).float()

        if weights is not None:
            weight = torch.unsqueeze(weights, dim=1)
            weight = weight.repeat(1, self.num_bin)
            loss = F.binary_cross_entropy(prob, binary_labels, weight=weight)
        else:
            loss = F.binary_cross_entropy(prob, binary_labels)
        return loss


class GaussianLoss(object):
    def __init__(self):
        self.eps = 1e-6
        self.loss = nn.GaussianNLLLoss(eps=self.eps)
    def __call__(self, mle, gt, weights=None):
        mle = torch.log(1 + torch.exp(mle))
        mu = torch.clamp(mle[:, 0], min=0)
        var = torch.clamp(mle[:, 1], min=0)

        if weights is not None:
            var = torch.clamp(var, min=self.eps)
            loss = torch.mean(
                weights*(0.5*(torch.log(var)+torch.square(mu-gt)/var))
            )
        else:
            loss = self.loss(mu, gt, var)
        return loss

class LogGaussianLoss(object):
    def __init__(self):
        self.eps = 1e-6
        self.loss = nn.GaussianNLLLoss(eps=self.eps)
    def __call__(self, mle, gt, weights=None):
        mle = torch.log(1 + torch.exp(mle))
        mu = torch.clamp(mle[:, 0], min=0)
        var = torch.clamp(mle[:, 1], min=0)

        if weights is not None:
            var = torch.clamp(var, min=self.eps)
            loss = torch.mean(
                weights*(0.5*(torch.log(var)+torch.square(mu-torch.log(gt))/var))
            )
        else:
            loss = self.loss(mu, torch.log(gt), var)
        return loss
