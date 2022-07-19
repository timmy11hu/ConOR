import argparse
import os
import glob
print("Current Working Directory ", os.getcwd())

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from simulator import data_simulator
from models import SimulationNet
import losses
from evaluation import PdfEvaluator, GaussinaEvaluator, LogGaussinaEvaluator, BinaryEvaluator, CdfEvaluator
import Trainers
from utils import MyLogger


def args():
    parser = argparse.ArgumentParser(description='Depth Uncertainty.')
    parser.add_argument('--seed', default=1000, type=int, help='seed')
    parser.add_argument('--num_bins', default=40, type=int, help='number of partitioned bins')
    parser.add_argument('--discretization', default='SID', type=str, help='SID, UD, Random')
    parser.add_argument('--cls_model', default='conor', type=str, help='lgl, gl, mcc, or, conor, bc')
    parser.add_argument('--use_writer', action='store_true', default=False, help='record training?')
    parser.add_argument('--root', default='Result', type=str, help='savingroot')

    parser.add_argument('--dataset', default='Simulation', type=str, help='dataset')
    parser.add_argument('--num_xtrain', default=2000, type=int, help='number of Xtrain')
    parser.add_argument('--num_xtest', default=2000, type=int, help='number of Xtest')
    parser.add_argument('--num_xeval', default=10000, type=int, help='number of Xeval')
    parser.add_argument('--sample', default='uniform', type=str, help='uniform, gaussian')
    parser.add_argument('--var', default='hetero', type=str, help='hetero, homo')
    parser.add_argument('--noise', default='N', type=str, help='N,L,T')
    parser.add_argument('--X_C', default=1, type=int, help='number of features')
    parser.add_argument('--dim_encoding', default=100, type=int, help='dimension of encoding')
    parser.add_argument('--alpha', default=1.0, type=float, help='value to alpha on the range')
    parser.add_argument('--beta', default=100.0, type=float, help='value to beta on the range')

    parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
    parser.add_argument('--epoch', default=10000, type=int, help='number of epoch')
    parser.add_argument('--batch_size', default=500, type=int, help='batch size')
    parser.add_argument('--dropout_rate', default=0.2, type=float, help='dropout_rate')

    parser.add_argument('--method', default='Single', type=str, help='Single, Truth, PairBS, WildBS, MultBS, MC, AdvBS')
    parser.add_argument('--num_ensemble', default=10, type=int, help='number of ensembles')

    arguments = parser.parse_args()
    return arguments


def run(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    # Housekeeping
    if os.path.exists(config.savingroot):
        files = glob.glob(config.savingroot+'/*')
        for f in files:
            os.remove(f)
    else:
        os.makedirs(config.savingroot)

    my_logger = MyLogger(config=config)
    if config.use_writer:
        writer = SummaryWriter(log_dir=os.path.join(config.savingroot, 'logs'), flush_secs=5)
    else:
        writer = None

    # Load Dataset
    multiple_dst = config.num_ensemble if config.method == 'Truth' else None
    simulator_setting = dict(
        ntrain=config.num_xtrain,
        ntest=config.num_xtest,
        neval=config.num_xeval,
        multiple=multiple_dst,
        C=config.X_C,
        H=1,
        W=1,
        seeding=config.seed,
        sample=config.sample,
        var=config.var,
        noise_dist = config.noise,
        root=os.path.join(config.root, config.dataset),
    )

    TrainX, TrainY, TestX, TestY, EvalX, EvalY = data_simulator(**simulator_setting)

    # Discretization
    beta = np.ceil(np.max(np.concatenate((TrainY, TestY, EvalY)))).item()
    alpha = np.floor(np.min(np.concatenate((TrainY, TestY, EvalY)))).item()
    print("alpha: ", alpha, ". beta: ", beta)
    # exit()

    # Model
    base_net = SimulationNet(dim_x=config.X_C, dim_encoding=config.dim_encoding, dropout_rate=config.dropout_rate,
                        num_bin=config.num_bins,
                        cls_model=config.cls_model,
                        alpha=alpha, beta=beta, discretization=config.discretization, lr=config.lr).to(device)

    # Ensemble
    ensembles = []
    for _ in range(config.num_ensemble):
        net = SimulationNet(dim_x=config.X_C, dim_encoding=config.dim_encoding, dropout_rate=config.dropout_rate,
                            num_bin=config.num_bins,
                            cls_model=config.cls_model,
                            alpha=alpha, beta=beta, discretization=config.discretization, lr=config.lr).to(device)
        ensembles.append(net)

    # Loss
    if config.cls_model == "mcc":
        criterion = losses.MultinomialLoss(num_bin=config.num_bins, alpha=alpha, beta=beta, discretization=config.discretization)
    elif config.cls_model == "or":
        criterion = losses.OrdinalRegressionLoss(num_bin=config.num_bins, alpha=alpha, beta=beta, discretization=config.discretization)
        evaluation = CdfEvaluator(num_bin=config.num_bins, alpha=alpha, beta=beta, discretization=config.discretization)
    elif config.cls_model == "conor":
        criterion = losses.MultiBinaryLoss(num_bin=config.num_bins, alpha=alpha, beta=beta, discretization=config.discretization)
        evaluation = PdfEvaluator(num_bin=config.num_bins, alpha=alpha, beta=beta, discretization=config.discretization)
    elif config.cls_model == "gl":
        criterion = losses.GaussianLoss()
        evaluation = GaussinaEvaluator()
    elif config.cls_model == "lgl":
        criterion = losses.LogGaussianLoss()
        evaluation = LogGaussinaEvaluator()
    elif config.cls_model == "bc":
        criterion = losses.BinaryClsLoss(num_bin=config.num_bins, alpha=alpha, beta=beta, discretization=config.discretization)
        evaluation = BinaryEvaluator(num_bin=config.num_bins, alpha=alpha, beta=beta, discretization=config.discretization)
    else:
        raise NotImplementedError

    # Training method
    trainer = Trainers.get_trainer(config, base_net, ensembles, criterion, evaluation,
                                   TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, my_logger, device)
    trainer.train()






if __name__ == '__main__':
    config = args()
    config.dataset = config.sample + '_' + config.var + config.noise + '_' + str(config.seed)
    config.savingroot = os.path.join(config.root, config.dataset, config.cls_model, config.method)
    print(config)
    torch.manual_seed(config.seed)
    run(config)
