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
from dp.utils.config import load_config, print_config
from dp.core.solver import Solver
from dp.datasets.loader import build_loader
from dp.utils.test_visualizer import Visualizer
from dp.utils.evaluator import Metrics
from dp.utils.pyt_io import create_summary_writer

parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--config', type=str)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--loss_type', default='conor', type=str)
parser.add_argument('--mode', default='wild', type=str)
parser.add_argument('--BS_num', default=20, type=int)
args = parser.parse_args()


config = load_config(args.config)
loss_type = args.loss_type
config['model']['params']['loss_type'] = loss_type
bs_name = args.mode
bs_num = args.BS_num

ckpts_pth = get_ckpt_dict(config)

exp_root = os.path.join(config["snap"]["path"], config['data']['name'], loss_type)
snap_dir = os.path.join(exp_root, bs_name+str(bs_num))
m0_pth = os.path.join(exp_root, ckpts_pth[loss_type]["single"])

if os.path.exists(os.path.join(snap_dir, 'tensorboard')):
    shutil.rmtree(os.path.join(snap_dir, 'tensorboard'))

writer = create_summary_writer(snap_dir)
visualizer = Visualizer(config, writer)

# a: Error Variance, e: Estimation Variance, ae: Predictive Variance
metric_a = Metrics()
metric_a.reset()
metric_e = Metrics()
metric_e.reset()
metric_ae = Metrics()
metric_ae.reset()

solver = Solver()
te_loader, _, niter_test = build_loader(config, False, solver.world_size, solver.distributed)

# Save individual prediction:
for m in range(1, bs_num+1):
    config['model']['pretrained_model'] = os.path.join(snap_dir, 'm'+str(m)+'.pth')
    solver.init_from_scratch(config, pretrain=True)
    solver.after_epoch()
    test_iter = iter(te_loader)
    for idx in range(niter_test):
        minibatch = test_iter.next()
        filtered_kwargs = solver.parse_kwargs(minibatch)
        mu, variance = solver.step_no_grad(filtered_kwargs['image'])
        mu = torch.squeeze(mu).cpu().numpy()
        unc = torch.squeeze(variance).cpu().numpy()
        save_dir = os.path.join(snap_dir, 'preds', str(minibatch['fn'][0]))
        try:
            os.makedirs(save_dir)
        except:
            pass
        path = os.path.join(save_dir, 'm{}'.format(m))
        np.savez(path, **{'mu': mu, 'var': unc})


# Final uncertainty combination
config['model']['pretrained_model'] = m0_pth
solver.init_from_scratch(config, pretrain=True)
solver.after_epoch()
test_iter = iter(te_loader)
ause0, ause1, ause2 = [], [], []
for idx in range(niter_test):
    minibatch = test_iter.next()
    filtered_kwargs = solver.parse_kwargs(minibatch)
    img = filtered_kwargs['image']
    pred_list, var_list = [], []
    for m in range(1, bs_num + 1):
        f = os.path.join(snap_dir, 'preds', str(minibatch['fn'][0]), 'm{}.npz'.format(m))
        pred_list.append(torch.from_numpy(np.load(f)['mu']))
        var_list.append(torch.from_numpy(np.load(f)['var']))
    preds = torch.squeeze(torch.stack(pred_list))
    vars = torch.squeeze(torch.stack(var_list))

    m, H, W = preds.shape
    model_var = torch.var(preds, dim=0, unbiased=True).view(-1, H, W).to(img.device)

    pred1, variance1 = solver.step_no_grad(img)
    pred2 = torch.mean(preds, dim=0).view(-1, H, W).to(img.device)
    variance2 = torch.mean(vars, dim=0).view(-1, H, W).to(img.device)

    # frequentist:
    pred, variance = pred1, variance1
    uncertainty = (variance + model_var)

    metric_a.compute_metric(pred, filtered_kwargs['target'], variance)
    metric_e.compute_metric(pred, filtered_kwargs['target'], model_var)
    metric_ae.compute_metric(pred, filtered_kwargs['target'], uncertainty)

    ause0.append(metric_a.ause_rmse.values()[0])
    ause1.append(metric_e.ause_rmse.values()[0])
    ause2.append(metric_ae.ause_rmse.values()[0])

    print("Test img ", idx, ": ", metric_ae.get_result_info())
    if idx % 10 == 0:
        visualizer.visualize(minibatch, pred, epoch=1, y_var=variance, m_var=model_var)


# 0: Error Variance, 1: Estimation Variance, 2: Predictive Variance
metric_a.add_scalar(writer, tag='Test', epoch=0)
metric_e.add_scalar(writer, tag='Test', epoch=1)
metric_ae.add_scalar(writer, tag='Test', epoch=2)
writer.close()

