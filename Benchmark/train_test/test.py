#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
import logging
import warnings
import sys

import numpy as np
import torch
warnings.filterwarnings("ignore")
logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

# running in parent dir
os.chdir("..")
sys.path.append(".")
print("Current Working Directory ", os.getcwd())

from.utils import get_ckpt_dict
from dp.core.utils import find_cutpoints2tensor, test_inference
from dp.utils.average_meter import AverageMeter
from dp.utils.config import load_config, print_config
from dp.models import _get_model
from dp.utils.pyt_io import create_summary_writer
from dp.utils.pyt_ops import tensor2cuda
from dp.datasets.loader import build_loader
from dp.utils.test_visualizer import Visualizer
from dp.utils.evaluator import Metrics


def model_inference(model, input, loss_type, bin_values):
    with torch.no_grad():
        output = model(input)
        depth, uncertainty = test_inference(loss_type, output, bin_values)
    return depth, uncertainty


def load_model(path):
    model = _get_model(config)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu'))['model'])
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--config', type=str)
parser.add_argument('--loss_type', default='conor', type=str, help='conor,or,gl,lgl,bc,mcc')
parser.add_argument('--mode', default='SG', type=str, help='SG,DE,MC')
parser.add_argument('--MC_num', default=20, type=int)
args = parser.parse_args()


config = load_config(args.config)
loss_type = args.loss_type
config['model']['params']['loss_type'] = args.loss_type
mode = args.mode
MC_num = args.MC_num

ckpts_pth = get_ckpt_dict(config)

m0_path = ckpts_pth[loss_type]["single"]
ensembles_path = ckpts_pth[loss_type]["ensemble"]

if mode == 'SG':
    exp_name = 'Test_SG'
elif mode == 'MC':
    exp_name = 'Test_MC'
elif mode == 'DE':
    exp_name = 'Test_DE'
else:
    raise NotImplementedError

ckpts_root = os.path.join(config["snap"]["path"], config['data']['name'], loss_type)
snap_dir = os.path.join(ckpts_root, exp_name)
if os.path.exists(snap_dir):
    print("Dir exist, remove ", snap_dir)
    shutil.rmtree(snap_dir)
if not os.path.exists(snap_dir):
    print("Create new dir")
    os.makedirs(snap_dir)
writer = create_summary_writer(snap_dir)
visualizer = Visualizer(config, writer)

model = load_model(os.path.join(ckpts_root, m0_path))
discretization = config["model"]["params"]["discretization"]
alpha = config["model"]["params"]["alpha"]
beta = config["model"]["params"]["beta"]
gamma = config["model"]["params"]["gamma"]
ord_num = config["model"]["params"]["ord_num"]
cutpoints, t0s, t1s, bin_values = find_cutpoints2tensor(discretization, ord_num, alpha, beta, gamma)

batch_size = 1
te_loader, _, niter_test = build_loader(config, False, batch_size, False)
# niter_test = 3

loss_meter = AverageMeter()
metric = Metrics()
metric.reset()

if mode == 'SG' or mode == 'MC':
    if mode == 'SG':
        print("Single Inference")
    else:
        print("MC Inference")
    test_iter = iter(te_loader)
    for idx in range(niter_test):
        print(idx)
        minibatch = test_iter.next()
        # if idx != 100: continue
        kwargs = {k: v for k, v in minibatch.items() if k in ('image', 'target', 'weight')}
        if torch.cuda.is_available():
            kwargs = tensor2cuda(kwargs)
        model.eval()
        if mode == 'SG':
            pred, y_unc = model_inference(model=model, input=kwargs['image'], loss_type=loss_type, bin_values=bin_values)
            model_var = None
            total_unc = y_unc
        else:
            for param in model.modules():
                if param.__class__.__name__.startswith('Dropout'):
                    param.train()
            pred_list, unc_list = [], []
            for _ in range(MC_num):
                pred, y_var = model_inference(model=model, input=kwargs['image'], loss_type=loss_type, bin_values=bin_values)
                pred_list.append(pred.cpu())
                unc_list.append(y_var.cpu())
            preds = torch.squeeze(torch.stack(pred_list))  # shape b, N, H, W
            uncs = torch.squeeze(torch.stack(unc_list))
            assert preds.shape == uncs.shape
            m, H, W = preds.shape
            pred_bar = torch.mean(preds, dim=0).view(batch_size, H, W)
            unc_bar = torch.mean(uncs, dim=0).view(batch_size, H, W)
            model_var = torch.var(preds, dim=0).view(batch_size, H, W)
            pred, y_unc = pred_bar.to(kwargs['target'].device), unc_bar.to(kwargs['target'].device)
            if loss_type in ('multi_cumsum', 'gaussian_nll', 'log_gaussian_nll'):
                total_unc = y_unc + model_var.to(kwargs['target'].device)
            else:
                total_unc = y_unc
        metric.compute_metric(pred, kwargs['target'], total_unc)
        print("Test img ", idx, ": ", metric.get_result_info())
        if idx % 10 == 0:
            visualizer.visualize(minibatch, pred, epoch=0, y_var=y_unc, m_var=model_var)
    metric.add_scalar(writer, tag='Test', epoch=0)
    writer.close()
elif mode == 'DE':
    print("DE Inference")
    mu_bar_list, var_bar_list, m_var_list = [], [], []
    DE_num = len(ensembles_path)
    pred_ensemble_lists, unc_ensemble_lists = [[] for b in range(DE_num)], [[] for b in range(DE_num)]
    for b, path in enumerate(ensembles_path):
        print("m: ", b)
        model = load_model(os.path.join(ckpts_root, path))
        test_iter = iter(te_loader)
        for idx in range(niter_test):
            print(idx)
            minibatch = test_iter.next()
            kwargs = {k: v for k, v in minibatch.items() if k in ('image', 'target', 'weight')}
            if torch.cuda.is_available():
                kwargs = tensor2cuda(kwargs)
            pred, y_var = model_inference(model=model, input=kwargs['image'], loss_type=loss_type,
                                          bin_values=bin_values)
            pred_ensemble_lists[b].append(pred.cpu().to(dtype=torch.float16))  # pred: N, H, W
            unc_ensemble_lists[b].append(y_var.cpu().to(dtype=torch.float16))
    for idx in range(niter_test):
        pred_list = [pred_ensemble_lists[b][idx] for b in range(DE_num)]  # ele: N, H, W
        preds = torch.squeeze(torch.stack(pred_list)) # shape b, N, H, W
        unc_list = [unc_ensemble_lists[b][idx] for b in range(DE_num)]
        uncs = torch.squeeze(torch.stack(unc_list))
        m, H, W = preds.shape
        mu_bar = torch.mean(preds, dim=0).view(batch_size, H, W)
        unc_bar = torch.mean(uncs, dim=0).view(batch_size, H, W)
        model_var = torch.var(preds, dim=0).view(batch_size, H, W)
        mu_bar_list.append(mu_bar)
        var_bar_list.append(unc_bar)
        m_var_list.append(model_var)
    del pred_ensemble_lists, unc_ensemble_lists

    metric0 = Metrics()
    metric0.reset()
    ause0, ause1 = [], []
    print("Final Inference")
    test_iter = iter(te_loader)
    for idx in range(niter_test):
        print(idx)
        minibatch = test_iter.next()
        kwargs = {k: v for k, v in minibatch.items() if k in ('image', 'target', 'weight')}
        if torch.cuda.is_available():
            kwargs = tensor2cuda(kwargs)
        target = kwargs['target']
        pred, y_unc = mu_bar_list[idx].to(target.device), var_bar_list[idx].to(target.device)
        pred, y_unc = pred.float(), y_unc.float()
        if len(m_var_list) == 0:
            model_var = None
            total_unc = y_unc
        else:
            model_var = m_var_list[idx].to(target.device)
            if loss_type in ('multi_cumsum', 'gaussian_nll', 'log_gaussian_nll'):
                metric0.compute_metric(pred, kwargs['target'], y_unc)
                total_unc = (y_unc + model_var)
            else:
                total_unc = y_unc
        metric.compute_metric(pred, kwargs['target'], total_unc)
        print("Test img ", idx, ": ", metric.get_result_info())

        ause0.append(metric0.ause_rmse.values()[0])
        ause1.append(metric.ause_rmse.values()[0])

        if idx % 10 == 0:
            visualizer.visualize(minibatch, pred, epoch=0, y_var=y_unc, m_var=model_var)
    metric.add_scalar(writer, tag='Test', epoch=1)
    writer.close()
    reduce_ause = np.array(ause0)-np.array(ause1)
    max_order = np.argsort(reduce_ause)
    print(max_order[:10])

else:
    raise NotImplementedError

print("ALL DONE")

