import math
import numpy as np
import torch
from scipy.stats import norm
from scipy.stats import gamma
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import os


def gen_ar1_corr_matrix(corr, p):
    return np.array([[np.power(corr, abs(i-j)) for i in range(p)] for j in range(p)])

def sample_x(n, C, distr):
    if distr == 'uniform':
        X = np.random.uniform(high=10, size=(n, C))
    elif distr == 'gaussian':
        X = np.concatenate(
            (np.random.normal(loc=2.0, scale=0.6, size=(int(0.25*n), C)),
             np.random.normal(loc=7.0, scale=1.0, size=(int(0.75*n), C)),
             )
        )
        while np.sum(X <= 0):
            X[X <= 0] = np.random.uniform(low=0.0, high=10.0, size=(np.sum(X <= 0), C)).reshape(-1)
        while np.sum(X >= 10):
            X[X >= 10] = np.random.uniform(low=0.0, high=10.0, size=(np.sum(X >= 10), C)).reshape(-1)

        if not np.sum(X <= 0.2):
            X[np.random.randint(0, n, 1)] = np.random.uniform(low=0.0, high=0.2, size=(1, C)).reshape(-1)
        if not np.sum(X >= 9.8):
            X[np.random.randint(0, n, 1)] = np.random.uniform(low=9.8, high=10.0, size=(1, C)).reshape(-1)

    else:
        raise NotImplementedError
    return X


def data_simulator(ntrain, ntest, neval, C, H, W, seeding, sample, var, noise_dist, root, multiple=None):
    seeding += 1

    # train
    np.random.seed(seeding*1)
    if multiple is not None:
        X_train = np.concatenate([sample_x(ntrain, C, sample) for _ in range(multiple)])
        ntrain *= multiple
    else:
        X_train = sample_x(ntrain, C, sample)

    # test
    np.random.seed(seeding*10)
    X_test = sample_x(ntest, C, sample)

    # eval
    np.random.seed(100)
    X_eval = np.tile(sample_x(neval, C, distr='uniform'), (1, 1))

    X = np.concatenate((X_train, X_test, X_eval))

    if C == 1:
        # gamma
        Y_mu = 200 * gamma.pdf(x=X[:, 0], a=4, loc=0, scale=2) + 10
    elif C == 5:
        Y_mu = np.power(X[:, 0]-5, 3)/10 - 2 * np.cos(1.2*X[:, 1]-2) + X[:, 2]/8 - X[:, 3]*X[:, 4]/12 + 30
    else:
        raise NotImplementedError

    if var == 'hetero':
        Y_var = 0.5 + 5*np.exp(-0.1*np.square(X[:, 0]-5))
    elif var == 'homo':
        Y_var = np.repeat(1, X.shape[0])
    else:
        raise NotImplementedError

    # store ground truth value of eval set
    Y_ent = 0.5*(1+np.log(2*Y_var*np.pi))
    f = os.path.join(root, 'EvalY_gt.npz')
    np.savez(f, **{'X': X[ntrain + ntest:, 0], 'Y_mu': Y_mu[ntrain + ntest:], 'Y_var': Y_var[ntrain + ntest:], 'Y_ent': Y_ent[ntrain + ntest:]})

    # noise
    if noise_dist == 'N':
        np.random.seed(seeding*2020)
        Eps_train = np.random.normal(scale=1, size=len(X_train))
        np.random.seed(seeding*2021)
        Eps_test = np.random.normal(scale=1, size=len(X_test))
        np.random.seed(seeding*2022)
        Eps_eval = np.random.normal(scale=1, size=len(X_eval))
    elif noise_dist == 'L':
        np.random.seed(seeding * 2020)
        Eps_train = np.random.laplace(scale=np.sqrt(0.5), size=len(X_train))
        np.random.seed(seeding * 2021)
        Eps_test = np.random.laplace(scale=np.sqrt(0.5), size=len(X_test))
        np.random.seed(seeding * 2022)
        Eps_eval = np.random.laplace(scale=np.sqrt(0.5), size=len(X_eval))
    else:
        raise NotImplementedError

    noise = np.concatenate((Eps_train, Eps_test, Eps_eval))

    Y = Y_mu + noise * np.sqrt(Y_var)
    Y_u = Y_mu + np.sqrt(Y_var)*1.96
    Y_l = Y_mu - np.sqrt(Y_var)*1.96

    a, b = np.min(Y), np.max(Y)
    print("ymin:", a)
    print("ymax:", b)

    if not os.path.exists(os.path.join(root, 'gt_figs')):
        os.mkdir(os.path.join(root, 'gt_figs'))

    # Can visualize if x is 1D
    if C == 1:
        sns.set_style("darkgrid")
        sns.set(font_scale=1.2)
        # trainset
        fig, ax = plt.subplots()
        sns.scatterplot(x=X[:ntrain, :].reshape(-1), y=Y[:ntrain].reshape(-1), s=10, label='Sample')
        sns.lineplot(x=X[ntrain + ntest:, :].reshape(-1), y=Y_mu[ntrain + ntest:].reshape(-1), linestyle='-', label='Mean Function', color='orange')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='lower right')
        fig.savefig(os.path.join(root, "gt_figs", "trainset.pdf"), bbox_inches='tight', dpi=500)
        plt.cla()
        plt.clf()
        plt.close()

        # testset
        fig, ax = plt.subplots()
        sns.scatterplot(x=X[ntrain:ntrain+ntest, :].reshape(-1), y=Y[ntrain:ntrain+ntest].reshape(-1), s=10)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Test set')
        fig.savefig(os.path.join(root, "gt_figs", "testset.png"), bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

        # evalset with CI
        fig, ax = plt.subplots()
        sns.scatterplot(x=X[ntrain + ntest:, :].reshape(-1), y=Y[ntrain + ntest:].reshape(-1), s=10)
        sns.scatterplot(x=X[ntrain + ntest:, :].reshape(-1), y=Y_mu[ntrain + ntest:].reshape(-1), s=2, color='red')
        sns.scatterplot(x=np.concatenate((X[ntrain + ntest:, :], X[ntrain + ntest:, :])).reshape(-1),
                        y=np.concatenate((Y_u[ntrain + ntest:], Y_l[ntrain + ntest:])).reshape(-1), s=2)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Evaluation set')
        fig.savefig(os.path.join(root, "gt_figs", "evalset.png"), bbox_inches='tight')
        plt.cla()
        plt.clf()
        plt.close()

        # GT Variance
        fig, ax = plt.subplots()
        sns.scatterplot(x=X.reshape(-1), y=Y_var.reshape(-1), s=2)
        plt.xlabel('x')
        plt.ylabel('Var(y)')
        plt.title('Ground truth of variance')
        fig.savefig(os.path.join(root, "gt_figs", "gt_vary.png"), bbox_inches='tight')

        # GT Entropy
        fig, ax = plt.subplots()
        sns.scatterplot(x=X.reshape(-1), y=Y_ent.reshape(-1), s=2)
        plt.xlabel('x')
        plt.ylabel('H(y)')
        plt.title('Ground truth of entropy')
        fig.savefig(os.path.join(root, "gt_figs", "gt_enty.png"), bbox_inches='tight')

    Y = Y.reshape(-1)
    # exit()

    TrainX = X[:ntrain, :]
    TrainY = Y[:ntrain]

    TestX = X[ntrain:ntrain + ntest, :]
    TestY = Y[ntrain:ntrain + ntest]

    EvalX = X[ntrain + ntest:, :]
    EvalY = Y[ntrain + ntest:]

    return TrainX, TrainY, TestX, TestY, EvalX, EvalY




class SimulationDataset(Dataset):
    def __init__(self, X, Y):
        assert len(X) == len(Y)
        self.X = torch.tensor(X).float()
        self.Y = torch.tensor(Y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.Y[index]
