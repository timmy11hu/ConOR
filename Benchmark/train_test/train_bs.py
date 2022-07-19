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
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from dp.datasets.utils import nomalize
from PIL import Image
import PIL.Image as pil
from dp.utils.pyt_ops import tensor2cuda
from torchvision.transforms.functional import InterpolationMode as IM



def preprocess(config, image, target, scale):
    crop_h, crop_w = config['data']["te_crop_size"]
    image = TF.resize(img=image, size=[crop_h, crop_w], interpolation=IM.BILINEAR)
    # normalize
    image = np.asarray(image).astype(np.float32) / 255.0
    image = nomalize(image, type=config['data']['norm_type'])
    image = image.transpose(2, 0, 1)
    target = np.array(target, dtype=float)
    target = target.astype(np.float) / scale
    if config['data']['name'] == 'Kitti':
        target[target > 80] = 80.
    elif config['data']['name'] == 'NyuV2':
        target[target > 10] = 10.
    else:
        raise NotImplementedError
    mask = [target == 0]
    image = torch.from_numpy(np.ascontiguousarray(image)).float()
    target = torch.from_numpy(np.ascontiguousarray(target)).float()
    return image, target, mask


def str2dirpth(target_path):
    s = target_path.split('/')
    if len(s) > 4:  # kitti
        directory = os.path.join(s[1], s[2], s[3], s[4])
        path = os.path.join(directory, s[5])
    else:  # nyuv2
        directory = os.path.join(s[1])
        path = os.path.join(directory, s[2])
    return directory, path


def save_fit_res(target_path, pred, residual, fit_root, res_root, scale):
    directory, path = str2dirpth(target_path)
    fit_dir = os.path.join(fit_root, directory)
    fit_path = os.path.join(fit_root, path)
    res_dir = os.path.join(res_root, directory)
    res_path = os.path.join(res_root, path)
    # fit value
    try:
        os.makedirs(fit_dir)
    except:
        pass
    try:
        os.makedirs(res_dir)
    except:
        pass
    im = pil.fromarray((pred * scale).astype(np.uint16))
    im.save(fit_path)
    # residual value
    im = pil.fromarray((residual * scale).astype(np.uint16))
    im.save(res_path)


parser = argparse.ArgumentParser(description='Training script')
parser.add_argument('--config', type=str)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--loss_type', default='conor', type=str)
parser.add_argument('--mode', default='wild', type=str)
parser.add_argument('--BS_num', default=20, type=int)
parser.add_argument('--epoch', default=2, type=int)
args = parser.parse_args()

if not args.config:
    logging.error('args --config should be available.')
    raise ValueError
is_main_process = True if args.local_rank == 0 else False

loss_type = args.loss_type
bs_name = args.mode
bs_num = args.BS_num

config = load_config(args.config)
config['model']['params']['loss_type'] = loss_type
alpha, beta = config['model']['params']['alpha'], config['model']['params']['beta']
config['bs']['name'] = bs_name
config['bs']['num'] = bs_num

exp_root = os.path.join(config["snap"]["path"], config['data']['name'], loss_type)
snap_dir = os.path.join(exp_root, bs_name+str(bs_num))

ckpts_pth = get_ckpt_dict(config)

m0_path = ckpts_pth[loss_type]["single"]
config['model']['pretrained_model'] = os.path.join(exp_root, m0_path)

if is_main_process:
    print_config(config)
    if os.path.exists(snap_dir):
        shutil.rmtree(snap_dir)
    if not os.path.exists(snap_dir):
        os.makedirs(snap_dir)

solver = Solver()
solver.init_from_scratch(config, pretrain=True)
# dataset
te_loader, _, niter_test = build_loader(config, False, solver.world_size, solver.distributed)
dataset_name = config['data']['name']
rgb_root = config['data']['rgb_path']

if dataset_name == 'Kitti':
    scale = 256.
    train_list = 'dp/datasets/lists/kitti_eigen_train_dense.list'
elif dataset_name == 'NyuV2':
    scale = 255.0 / 10.0
    train_list = 'dp/datasets/lists/nyuv2_train_sub50k.list'
else:
    raise NotImplementedError

with open(train_list, "r") as f:
    filenames = f.readlines()

# debug:
print("niter_test: ", niter_test)

local_rank, world_size = solver.local_rank, solver.world_size

solver.after_epoch()
train_split, test_split = config['data']['split']
if bs_name == 'wild':
    config['data']['split'] = [train_split + "_wild", test_split]
    fit_root = os.path.join(exp_root, 'data_depth_fit')
    res_root = os.path.join(exp_root, 'data_depth_res')
    for i, line in enumerate(filenames):
        if i % world_size == local_rank:
            image_path, target_path = line.split()
            image = Image.open(os.path.join(rgb_root, image_path))
            target = Image.open(os.path.join(rgb_root, target_path))
            image, target, mask = preprocess(config, image, target, scale)
            filtered_kwargs = {'image': image, 'target': target}
            if torch.cuda.is_available():
                filtered_kwargs = tensor2cuda(filtered_kwargs)
            # predict and inference
            with torch.no_grad():
                image = torch.unsqueeze(filtered_kwargs['image'], dim=0)
                target = torch.unsqueeze(filtered_kwargs['target'], dim=0)
                pred, _ = solver.step_no_grad(image)
                pred = torch.unsqueeze(pred, dim=1)
                h, w = target.shape[-2:]
                pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=True)
                pred = torch.squeeze(pred).cpu().numpy()
                target = torch.squeeze(target).cpu().numpy()
                residual = np.abs(target - pred)
                pred[mask] = 0
                residual[mask] = 0
            save_fit_res(target_path, pred, residual, fit_root, res_root, scale)
    config['data']['depth_path'] = exp_root
elif bs_name == 'multiplier':
    config['data']['split'] = [train_split + "_multiplier", test_split]
    config['data']['weight_path'] = exp_root
else:
    raise NotImplementedError

solver.after_epoch()
print("Pre BS housekeeping finish..")

# start bs:
for m in range(1, bs_num+1):
    solver.init_from_scratch(config, pretrain=True)
    solver.after_epoch()
    # sample data
    torch.manual_seed(local_rank * m * 2022)
    np.random.seed(local_rank * m * 2022)
    if bs_name == 'wild':
        wild_root = os.path.join(exp_root, 'data_depth_wild')
        for i, line in enumerate(filenames):
            if i % world_size == local_rank:
                image_path, target_path = line.split()
                directory, path = str2dirpth(target_path)
                fit_path = os.path.join(fit_root, path)
                res_path = os.path.join(res_root, path)
                wild_dir = os.path.join(wild_root, directory)
                wild_path = os.path.join(wild_root, path)

                fit = Image.open(os.path.join(fit_path))
                res = Image.open(os.path.join(res_path))
                fit = np.array(fit).astype(np.float) / scale
                res = np.array(res).astype(np.float) / scale
                noise = np.random.normal(loc=0.0, scale=1.0, size=res.shape)
                gt_star = fit + np.multiply(res, noise)
                gt_star[gt_star < alpha] = alpha
                gt_star[gt_star > beta] = beta
                try:
                    os.makedirs(wild_dir)
                except:
                    pass
                im = pil.fromarray((gt_star * scale).astype(np.uint16))
                im.save(wild_path)
    elif bs_name == 'multiplier':
        for i, line in enumerate(filenames):
            if i % world_size == local_rank:
                image_path, depth_path = line.split()
                target = Image.open(os.path.join(rgb_root, depth_path))
                target = np.array(target, dtype=float) / scale
                mask = [target == 0]
                # save weights
                directory, path = str2dirpth(depth_path)
                weight = np.random.normal(loc=1.0, scale=1.0, size=target.shape).astype('float16')
                weight[mask] = 0
                weight_dir = os.path.join(exp_root, 'data_weight', directory)
                try:
                    os.makedirs(weight_dir)
                except:
                    pass
                path = os.path.join(exp_root, 'data_weight', path[:-4])
                np.save(path, weight)
    else:
        raise NotImplementedError

    solver.after_epoch()
    print("Finish sampling for num {}, start training.".format(m))

    tr_loader, sampler, niter_per_epoch = build_loader(config, True, solver.world_size, solver.distributed)
    for epoch in range(args.epoch):
        solver.before_epoch(epoch=epoch)
        if solver.distributed:
            sampler.set_epoch(epoch)
        pbar = range(niter_per_epoch)
        train_iter = iter(tr_loader)
        print(niter_per_epoch)
        for idx in pbar:
            # print(idx)
            minibatch = train_iter.next()
            filtered_kwargs = solver.parse_kwargs(minibatch)
            loss = solver.step(filtered_kwargs['image'], filtered_kwargs['target'], filtered_kwargs.get('weight'))

    solver.after_epoch()
    print("Model {} All Done".format(m))

    if is_main_process:
        snap_name = os.path.join(snap_dir, 'm{}.pth'.format(m))
        solver.save_checkpoint(snap_name)

print("Finish ", bs_num)


