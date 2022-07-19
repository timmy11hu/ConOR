import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
# from simclr.modules import LARS
import copy
import scipy.stats
import seaborn as sns


def cal_Sparsification(order, y_hat, gt, fraction, step):
    normalizer = np.sqrt(np.mean(np.square(y_hat - gt)))
    y_hat = y_hat[order]
    gt = gt[order]
    x_axis = []
    y_axis = []
    for i in range(fraction):
        y = y_hat[i * step:]
        target = gt[i * step:]
        x_axis.append(1/fraction * i)
        y_axis.append(np.sqrt(np.mean(np.square(y - target)))/normalizer)
    return np.array(x_axis), np.array(y_axis)


def cal_AUSE(x, y_hat, y_var, gt, config, epoch, model_var=None, fraction=100):
    n = len(gt)
    step = int(n/fraction)

    if model_var is not None:
        y_var += model_var

    # use uncertainty
    var_order = np.argsort(-y_var)
    var_xaxis, var_yaxis = cal_Sparsification(var_order, y_hat, gt, fraction, step)

    # oracle
    abs_error = np.abs(y_hat-gt)
    oracle_order = np.argsort(-abs_error)
    oracle_xaxis, oracle_yaxis = cal_Sparsification(oracle_order, y_hat, gt, fraction, step)

    fig, ax = plt.subplots()
    spar_error = var_yaxis - oracle_yaxis
    sns.lineplot(x=var_xaxis, y=spar_error)
    # fig.suptitle('Sparsification Error curve')
    # ax.plot(var_xaxis, var_yaxis, '-')
    # ax.plot(oracle_xaxis, oracle_yaxis, '--')
    # ax.plot(var_xaxis, var_yaxis-oracle_yaxis, '-.' )

    # sns.lineplot(x=var_xaxis, y=var_yaxis)
    # sns.lineplot(x=oracle_xaxis, y=oracle_yaxis)
    plt.xlabel('Fraction of sample removed')
    plt.ylabel('Sparsification Error')
    plt.title('Mu hat vs. gt')

    fig.savefig(config.savingroot + '/AUSE_epoch_' + str(epoch) + '.png')
    plt.cla()
    plt.clf()
    plt.close()

    ause = np.mean(np.abs(spar_error))
    return ause, var_xaxis, spar_error


def cal_AUCE(x, y_hat, y_var, gt, config, epoch, model_var=None, fraction=100):
    if model_var is not None:
        y_var += model_var

    confidence = []
    cover_rate = []
    for i in range(fraction):
        alpha = i * 1/fraction
        z = np.abs(scipy.stats.norm.ppf((1-alpha)/2))
        y_upper = y_hat + np.sqrt(y_var) * z
        y_lower = y_hat - np.sqrt(y_var) * z
        is_covered = np.logical_and(gt >= y_lower, gt <= y_upper)
        p = np.mean(is_covered)
        confidence.append(alpha)
        cover_rate.append(p)

    confidence = np.array(confidence)
    cover_rate = np.array(cover_rate)

    # fig, ax = plt.subplots()
    # fig.suptitle('Calibration Error curve')
    # ax.plot(confidence, cover_rate, '-')
    # ax.plot(confidence, confidence, '--')
    # fig.savefig(config.savingroot + '/AUCE_epoch_' + str(epoch) + '.png')
    # plt.cla()
    # plt.clf()
    # plt.close()

    auce = np.mean(np.abs(confidence - cover_rate))
    return auce


def cal_NLL(x, y_hat, y_var, gt):
    L = 1/(2*np.pi*y_var) * np.exp(-np.square(y_hat-gt)/(2*y_var))
    np.clip(L, 1e-10, np.inf, out=L)
    return -np.log(L)


def toggle_grad(model, on_or_off):
    for param in model.parameters():
        param.requires_grad = on_or_off


def find_cutpoints(discretization, num_bin, alpha, beta):
    if discretization == "SID":
        cutpoints = [
            np.exp(
                np.log(alpha) + ((np.log(beta)-np.log(alpha)) * float(b + 1) / num_bin)
            )
            for b in range(num_bin)
        ]
    elif discretization == "UD":
        cutpoints = [alpha + (beta - alpha) * (float(b + 1) / num_bin) for b in range(num_bin)]
    else:
        raise NotImplementedError

    return torch.tensor(cutpoints, requires_grad=False)


def create_nominal_label(gts, discretization, num_bin, alpha, beta):
    # N, H, W = gts.shape
    # print("gt shape:", gt.shape)
    N = gts.shape[0]
    if discretization == "SID":
        bin = torch.tensor(num_bin * (torch.log(gts) - np.log(alpha)) / (np.log(beta) - np.log(alpha))).to(gts.device)
    elif discretization == "UD":
        bin = torch.tensor(num_bin * (gts - alpha) / (beta - alpha)).to(gts.device)
    else:
        bin = None
    bin_label = bin.long()
    bin_label[gts > beta] = num_bin + 1
    bin_label = bin_label.view(N)

    return bin_label.to(gts.device)


def create_ordinal_label(gts, discretization, num_bin, alpha, beta):
    N = gts.shape[0]
    # N, H, W = gts.shape

    ord_label = torch.zeros(N, num_bin).to(gts.device)
    # ord_label = torch.zeros(N, self.ord_num, H, W).to(gts.device)
    if discretization == "SID":
        bin_label = num_bin * (torch.log(gts) - np.log(alpha)) / (np.log(beta) - np.log(alpha))
    elif discretization == "UD":
        bin_label = num_bin * (gts - alpha) / (beta - alpha)
    else:
        bin_label = None

    bin_label = bin_label.long().view(N, 1)
    # label = label.long().view(N, 1, H, W)
    # mask = torch.linspace(0, self.num_bin - 1, self.num_bin, requires_grad=False).view(1, self.num_bin, 1, 1).to(gts.device)
    # mask = mask.repeat(N, 1, H, W).contiguous().long()
    mask = torch.linspace(0, num_bin - 1, num_bin, requires_grad=False).view(-1, num_bin).to(gts.device)
    mask = mask.repeat(N, 1).contiguous().long()
    mask = (mask >= bin_label)
    ord_label[mask] = 1

    return ord_label




def inf_plot_1D(y, y_hat, y_var_hat, y_ent_hat, x, y_mu_gt, config, epoch, y_var_gt, y_ent_gt, m_var=None, m_var_gt=None):
    # mu plot
    sns.set_style("darkgrid")
    fig, ax = plt.subplots()
    # fig.suptitle('Mean on evalset')
    # ax.scatter(x[:, 0], y_mu_gt, s=5, color='g', label='gt')
    # ax.scatter(x[:, 0], y_hat, s=5, color='r', label='est')
    # # ax.set(xlim=[-1, 11], ylim=[0, 40])
    sns.scatterplot(x=x[:, 0], y=y_mu_gt, s=2, label='gt')
    sns.scatterplot(x=x[:, 0], y=y_hat, s=2, label='hat')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Mu hat vs. gt')
    # fig.legend()
    fig.savefig(config.savingroot + '/Mean_epoch_' + str(epoch) + '.png')
    plt.cla()
    plt.clf()
    plt.close()

    # y variance plot
    fig, ax = plt.subplots()
    # fig.suptitle('Y Variance on evalset')
    sns.scatterplot(x[:, 0], y_var_gt, s=2, label='gt')
    sns.scatterplot(x[:, 0], y_var_hat, s=2, label='hat')
    # ax.set(xlim=[-1, 11], ylim=[0, np.max(y_var_gt)+1])
    plt.xlabel('x')
    plt.ylabel('Var(y)')
    plt.title('Var(y) hat vs. gt')
    # fig.legend()
    fig.savefig(config.savingroot + '/Yvar_epoch_' + str(epoch) + '.png')
    plt.cla()
    plt.clf()
    plt.close()

    # y entropy plot
    fig, ax = plt.subplots()
    # fig.suptitle('Y Variance on evalset')
    sns.scatterplot(x[:, 0], y_ent_gt, s=2, label='gt')
    sns.scatterplot(x[:, 0], y_ent_hat, s=2, label='hat')
    # ax.set(xlim=[-1, 11], ylim=[0, np.max(y_var_gt)+1])
    plt.xlabel('x')
    plt.ylabel('H(y)')
    plt.title('H(y) hat vs. gt')
    # fig.legend()
    fig.savefig(config.savingroot + '/Yent_epoch_' + str(epoch) + '.png')
    plt.cla()
    plt.clf()
    plt.close()

    # model variance plot
    fig, ax = plt.subplots()
    # fig.suptitle('M Variance on evalset')
    sns.scatterplot(x[:, 0], m_var_gt, s=2, color='g', label='gt')
    sns.scatterplot(x[:, 0], m_var, s=2, color='r', label='est')
    # ax.set(xlim=[-1, 11], ylim=[0, np.max(m_var_gt)+1])
    plt.xlabel('x')
    plt.ylabel('Var(m)')
    plt.title('Var(m) hat vs. gt')
    # fig.legend()
    fig.savefig(config.savingroot + '/MVar_epoch_' + str(epoch) + '.png')
    plt.cla()
    plt.clf()
    plt.close()

    # # CI plot
    # fig, ax = plt.subplots()
    # fig.suptitle('Inference on evalset')
    # ax.scatter(x[:, 0], y, s=5, label='obs')
    # ax.scatter(x[:, 0], y_hat, s=5, color='r', label='mean')
    # ax.scatter(x[:, 0], y_hat + np.sqrt(y_var)*1.96, color='c', s=0.2)
    # ax.scatter(x[:, 0], y_hat - np.sqrt(y_var)*1.96, color='c', s=0.2)
    # if m_var is not None:
    #     ax.scatter(x[:, 0], y_hat + np.sqrt(y_var + m_var)*1.96, color='b', s=0.2)
    #     ax.scatter(x[:, 0], y_hat - np.sqrt(y_var + m_var)*1.96, color='b', s=0.2)

    # fig.legend()
    # # ax.set_xlim([-2, 12])
    # # ax.set_ylim([0, 100])
    # fig.savefig(config.savingroot + '/Inf_epoch_' + str(epoch) + '.png')
    # plt.cla()
    # plt.clf()
    # plt.close()



def inf_plot_ND(y, y_hat, y_var, y_mu_gt, config, epoch, y_var_gt):
    # mu plot
    idx = np.argsort(y_mu_gt)
    fig, ax = plt.subplots()
    fig.suptitle('Mu hat vs. ground truth mu on evalset')
    ax.scatter(np.arange(len(y)), y_mu_gt[idx], s=2, color='g', label='gt')
    ax.scatter(np.arange(len(y)), y_hat[idx], s=2, color='r', label='est')
    fig.legend()
    ax.set_xlabel('# of obs')
    ax.set_ylabel('Mu value')
    fig.savefig(config.savingroot + '/MuHat_epoch_' + str(epoch) + '.png')
    plt.cla()
    plt.clf()
    plt.close()

    # variance plot
    fig, ax = plt.subplots()
    idx = np.argsort(y_var_gt)
    fig.suptitle('Var hat vs. ground truth var on evalset')
    ax.scatter(np.arange(len(y)), y_var_gt[idx], s=2, color='g', label='gt')
    ax.scatter(np.arange(len(y)), y_var[idx], s=2, color='r', label='est')
    fig.legend()
    ax.set_xlabel('# of obs')
    ax.set_ylabel('Var value')
    fig.savefig(config.savingroot + '/VarHat_epoch_' + str(epoch) + '.png')
    plt.cla()
    plt.clf()
    plt.close()



def cal_95CR(x, y_hat, y_var, gt, model_var=None, discrete=False):
    if model_var is not None:
        y_upper = y_hat + np.sqrt(y_var + model_var)*1.96
        y_lower = y_hat - np.sqrt(y_var + model_var)*1.96
    else:
        y_upper = y_hat + np.sqrt(y_var)*1.96
        y_lower = y_hat - np.sqrt(y_var)*1.96

    x = x.reshape(-1)
    eval_xs = np.sort(np.unique(x))
    res = []
    for ele in eval_xs:
        idx = x==ele
        y_ta = gt[idx]
        y_lo = y_lower[idx]
        y_up = y_upper[idx]
        is_covered = np.logical_and(y_ta >= y_lo, y_ta <= y_up)
        p_hat = np.mean(is_covered)
        res.append(p_hat)
    return np.array(res)


class MyLogger(object):
    def __init__(self, config, item='alog', reinitialize=False, logstyle='%3.3f'):
        self.root = config.savingroot
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        # self.reinitialize = reinitialize
        # self.metrics = []
        self.item = item
        self.logstyle = logstyle  # One of '%3.3f' or like '%3.3e'

        if os.path.exists('%s/%s.txt' % (self.root, self.item)):
            print('{} exists, deleting...'.format('%s_%s.txt' % (self.root, self.item)))
            os.remove('%s/%s.txt' % (self.root, self.item))

    def log(self, epoch, rmse, mae, absrel, y_var_mae, y_var_absrel, y_ent_mae, y_ent_absrel, m_var_mae, m_var_absrel, cre, nll):
        with open('%s/%s.txt' % (self.root, self.item), 'a') as f:
            f.write('Epoch %d: RMSE %s, MAE %s, AbsRel %s, YV_MAE %s, YV_AbsRel %s, MV_MAE %s, MV_AbsRel %s, 95CRE %s, NLL %s\n'
                    % (epoch, self.logstyle % rmse, self.logstyle % mae, self.logstyle % absrel, self.logstyle % y_var_mae, self.logstyle % y_var_absrel,
                       self.logstyle % m_var_mae, self.logstyle % m_var_absrel, self.logstyle % cre, self.logstyle % nll))

    def midbreak(self):
        with open('%s/%s.txt' % (self.root, self.item), 'a') as f:
            f.write('\nMidbreak \n')


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=1000, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_rmse = np.inf
        self.best_epoch = 0
        self.early_stop = False
        self.model_ckpts = None

    def __call__(self, rmse, model, epoch):

        if rmse > self.best_rmse:
            self.counter += 1
            # print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_rmse = rmse
            self.best_epoch = epoch
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''

        # torch.save(model.state_dict(),modelname+str+'.m')
        self.model_ckpts = copy.deepcopy(model.state_dict())

