import torch
import numpy as np
from scipy.stats import pearsonr
from utils import find_cutpoints
import torch.nn.functional as F


class GaussinaEvaluator(object):
    def __init__(self):
        pass
    def __call__(self, output, gt, var_method='definition', smooth=False, TrainX=None, TestX=None, residual=None):
        mle = torch.log(1 + torch.exp(output))
        mle = mle.cpu()
        gt = gt.cpu()
        mu = mle[:, 0].detach().numpy()
        sigma2 = mle[:, 1].detach().numpy()
        mu, sigma2 = np.clip(mu, a_min=0, a_max=100), np.clip(sigma2, a_min=0, a_max=100)
        ent = 0.5 * (1 + np.log(2 * sigma2 * np.pi))
        se = np.square(gt.numpy() - mu)
        rmse = np.sqrt(np.mean(se))
        metrics = {
            'rmse': rmse,
            'mu_array': mu,
            'var_array': sigma2,
            'ent_array': ent,
        }
        return metrics


class LogGaussinaEvaluator(object):
    def __init__(self):
        pass
    def __call__(self, output, gt, var_method='definition', smooth=False, TrainX=None, TestX=None, residual=None):
        mle = torch.log(1 + torch.exp(output))
        mle = mle.cpu()
        gt = gt.cpu()
        mu_z = mle[:, 0].detach().numpy()
        sigma2_z = mle[:, 1].detach().numpy()
        mu_z, sigma2_z = np.clip(mu_z, a_min=0, a_max=5), np.clip(sigma2_z, a_min=0, a_max=1)
        mu = np.exp(mu_z+sigma2_z/2)
        sigma2 = (np.exp(sigma2_z) - 1)*np.exp(2*mu_z+sigma2_z)
        ent = 0.5 * (1 + np.log(2 * sigma2 * np.pi))
        se = np.square(gt.numpy() - mu)
        rmse = np.sqrt(np.mean(se))
        metrics = {
            'rmse': rmse,
            'mu_array': mu,
            'var_array': sigma2,
            'ent_array': ent,
        }
        return metrics


class PdfEvaluator(object):
    def __init__(self, num_bin, alpha, beta, discretization):
        self.num_bin = num_bin
        self.alpha = alpha
        self.beta = beta
        self.gamma = 1.0
        self.discretization = discretization
        self.cutpoints = find_cutpoints(alpha=self.alpha, beta=self.beta, num_bin=self.num_bin,
                                        discretization=self.discretization)
        self.t0s = torch.cat((torch.tensor(alpha).view(-1), self.cutpoints), dim=0)
        self.t1s = torch.cat((self.cutpoints, torch.tensor(beta).view(-1)), dim=0)
        self.bin_values = (self.t0s + self.t1s) / 2


    def __call__(self, output, gt, var_method='definition', smooth=False, TrainX=None, TestX=None, residual=None):
        pdf = output.cpu()
        gt = gt.cpu()
        pdf[pdf < 0] = 0
        mu, sigma2 = cal_var_by_def(pdf=pdf, bin_values=self.bin_values, cutpoints=self.t1s, smooth=smooth)
        clipped_probs = torch.clamp(input=pdf, min=1e-7, max=1 - 1e-7)
        ent = -torch.sum(clipped_probs * torch.log(clipped_probs), dim=1).detach().numpy()
        se = np.square(gt.numpy() - mu)
        rmse = np.sqrt(np.mean(se))

        metrics = {
            'rmse': rmse,
            'mu_array': mu,
            'var_array': sigma2,
            'ent_array': ent,
        }
        return metrics


class CdfEvaluator(object):
    def __init__(self, num_bin, alpha, beta, discretization):
        self.num_bin = num_bin
        self.alpha = alpha
        self.beta = beta
        self.gamma = 1.0
        self.discretization = discretization
        self.cutpoints = find_cutpoints(alpha=self.alpha, beta=self.beta, num_bin=self.num_bin,
                                        discretization=self.discretization)
        self.t0s = torch.cat((torch.tensor(alpha).view(-1), self.cutpoints), dim=0)
        self.t1s = torch.cat((self.cutpoints, torch.tensor(beta).view(-1)), dim=0)
        self.bin_values = (self.t0s + self.t1s) / 2

    def __call__(self, output, gt, var_method='definition', smooth=False, TrainX=None, TestX=None, residual=None):
        cdf = output[:, 1, :].cpu()
        gt = gt.cpu()
        cdf[cdf < 0] = 0
        # print(cdf)
        pdf = torch.diff(cdf, dim=1)
        mu, sigma2 = cal_var_by_def(pdf=pdf, bin_values=self.bin_values[1:-1], cutpoints=self.t1s, smooth=smooth)


        clipped_probs = torch.clamp(input=pdf, min=1e-7, max=1 - 1e-7)
        ent = -torch.sum(clipped_probs * torch.log(clipped_probs), dim=1).detach().numpy()

        se = np.square(gt.numpy() - mu)
        rmse = np.sqrt(np.mean(se))

        metrics = {
            'rmse': rmse,
            'mu_array': mu,
            'var_array': sigma2,
            'ent_array': ent,
        }

        return metrics


class BinaryEvaluator(object):
    def __init__(self, num_bin, alpha, beta, discretization):
        self.num_bin = num_bin
        self.alpha = alpha
        self.beta = beta
        self.gamma = 1.0
        self.discretization = discretization
        self.cutpoints = find_cutpoints(alpha=self.alpha, beta=self.beta, num_bin=self.num_bin,
                                        discretization=self.discretization)
        self.t0s = torch.cat((torch.tensor(alpha).view(-1), self.cutpoints), dim=0)
        self.t1s = torch.cat((self.cutpoints, torch.tensor(beta).view(-1)), dim=0)
        self.bin_values = (self.t0s + self.t1s) / 2

    def __call__(self, output, gt, var_method='definition', smooth=False, TrainX=None, TestX=None, residual=None):
        logits = output.cpu()
        gt = gt.cpu()
        pdf = F.softmax(logits, dim=1)
        pdf[pdf < 0] = 0

        mu, sigma2 = cal_var_by_def(pdf=pdf, bin_values=self.bin_values, cutpoints=self.t1s, smooth=smooth)

        clipped_probs = torch.clamp(input=pdf, min=1e-7, max=1 - 1e-7)
        ent = -torch.sum(clipped_probs*torch.log(clipped_probs), dim=1).detach().numpy()

        se = np.square(gt.numpy() - mu)
        rmse = np.sqrt(np.mean(se))

        metrics = {
            'rmse': rmse,
            'mu_array': mu,
            'se_array': se,
            'var_array': sigma2,
            'ent_array': ent,
        }

        return metrics

def cal_var_by_def(pdf, bin_values, cutpoints, smooth):
    N = pdf.shape[0]
    mu = torch.sum(bin_values * pdf, dim=1).view(N, -1)
    var = torch.sum(torch.square(bin_values - mu)*pdf, dim=1)
    return mu.view(-1).detach().numpy(), var.detach().numpy()


def evaluate_monotonicity(cdf, cutpoints=None, return_crossing_freq=True):
    nobs = cdf.shape[0]
    monotonic = []
    # eliminate some value like 1.000001
    cdf[cdf > 1.0] = 1.0

    if return_crossing_freq:
        diff_matrix = np.diff(cdf)
        return np.sum(diff_matrix < 0) / np.prod(diff_matrix.shape)
    else:
        for i in range(nobs):
            num_cor = pearsonr(cdf[i, :], cutpoints)[0]
            denom_cor = pearsonr(np.sort(cdf[i, :]), np.sort(cutpoints))[0]
            if num_cor != 0:
                mono = num_cor / denom_cor
            else:
                mono = 0
            monotonic.append(mono)

        return np.mean(monotonic)
