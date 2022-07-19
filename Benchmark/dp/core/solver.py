#!/usr/bin/python3
# -*- coding: utf-8 -*-

import logging

logging.basicConfig(format='[%(asctime)s %(levelname)s] %(message)s',
                    datefmt=           '%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

import os
import time
import gc

import numpy as np
import random
import inspect

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import SyncBatchNorm as SBN
from torch.cuda.amp import GradScaler, autocast
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
from dp.utils.comm import reduce_tensor
from dp.utils.pyt_io import load_model
from dp.core.lr_policys import _get_lr_policy
from dp.core.optimizers import _get_optimizer
from dp.core.utils import find_cutpoints2tensor, test_inference
from dp.models import _get_model
from dp.utils.comm import synchronize
from dp.utils.pyt_ops import tensor2cuda
from dp.version import __version__
from dp.models.losses import _get_loss_func
from dp.models.modules.adv_attact import FastGradientSignUntargeted



class Solver(object):

    def __init__(self):
        """
            :param config: easydict
        """
        self.version = __version__
        # logging.info("PyTorch Version {}, Solver Version {}".format(torch.__version__, self.version))
        self.distributed = False
        # self.amp = False
        # self.scaler = GradScaler()
        self.world_size = 1
        self.local_rank = 0
        self.epoch = 0
        self.iteration = 0
        self.config = None
        self.model, self.optimizer, self.lr_policy = None, None, None
        self.step_decay = 1
        self.filtered_keys = None

        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.distributed = self.world_size > 1 or torch.cuda.device_count() > 1

        if self.distributed:
            dist.init_process_group(backend="nccl", init_method='env://')
            self.local_rank = dist.get_rank()
            torch.cuda.set_device(self.local_rank)
            logging.info('[distributed mode] world size: {}, local rank: {}.'.format(self.world_size, self.local_rank))
        else:
            logging.info('[Single GPU mode]')

    def _build_environ(self):
        if self.config['environ']['deterministic']:
            cudnn.benchmark = False
            cudnn.deterministic = True
            torch.set_printoptions(precision=10)
        else:
            cudnn.benchmark = True

        # set random seed
        torch.manual_seed(self.config['environ']['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config['environ']['seed'])
        np.random.seed(self.config['environ']['seed'])
        random.seed(self.config['environ']['seed'])

    def init_from_scratch(self, config, pretrain=False):
        self.config = config
        self._build_environ()

        # model and optimizer
        self.model = _get_model(self.config)
        if torch.cuda.is_available():
            self.model.cuda(self.local_rank)

        self.discretization = self.config["model"]["params"]["discretization"]
        self.alpha = self.config["model"]["params"]["alpha"]
        self.beta = self.config["model"]["params"]["beta"]
        self.gamma = self.config["model"]["params"]["gamma"]
        self.ord_num = self.config["model"]["params"]["ord_num"]
        self.cutpoints, self.t0s, self.t1s, self.bin_values = find_cutpoints2tensor(self.discretization, self.ord_num,
                                                                                    self.alpha, self.beta, self.gamma)
        self.loss_type = self.config["model"]["params"]["loss_type"]
        self.criterion = _get_loss_func(self.loss_type, self.ord_num,
                                        self.alpha, self.beta,self.gamma,
                                        self.discretization, self.t0s, self.t1s)
        # self.criterion = OrdinalRegressionLoss(self.ord_num, self.beta, self.gamma, self.discretization)

        # self.filtered_keys = [p.name for p in inspect.signature(self.model.forward).parameters.values()]
        # logging.info("filtered keys:{}".format(self.filtered_keys))
        # model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        model_params = []
        for params in self.model.optimizer_params():
            params["lr"] = self.config["solver"]["optimizer"]["params"]["lr"] * params["lr"]
            model_params.append(params)
        self.optimizer = _get_optimizer(config['solver']['optimizer'],
                                        model_params=model_params)

        self.lr_policy = _get_lr_policy(config['solver']['lr_policy'], optimizer=self.optimizer)
        self.step_decay = config['solver']['step_decay']

        if pretrain:
            logging.info('loadding pretrained model from {}.'.format(config['bs']['pretrained_model']))
            load_model(self.model, config['model']['pretrained_model'], distributed=False)

        if self.distributed:
            self.model = SBN.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.local_rank])

        # grad clip settings
        self.grad_clip_params = self.config["solver"]["optimizer"].get("grad_clip")
        self.use_grad_clip = True if self.grad_clip_params is not None else False
        if self.use_grad_clip:
            logging.info("Using grad clip and params is {}".format(self.grad_clip_params))
        else:
            logging.info("Not Using grad clip.")

        # self.use_adv = self.config["solver"]['use_adv']
        # if self.use_adv:
        #     logging.info("Adv Training.")
        # self.attack = FastGradientSignUntargeted(epsilon=1, alpha=0.1, min_val=0, max_val=10, max_iters=1)
        # self.adv_w = 0.5

    def parse_kwargs(self, minibatch):
        # kwargs = {k: v for k, v in minibatch.items() if k in self.filtered_keys}
        kwargs = {k: v for k, v in minibatch.items() if k in ('image', 'target', 'weight')}
        if torch.cuda.is_available():
            kwargs = tensor2cuda(kwargs)
        return kwargs

    def step(self, image, target, weight=None):
        self.iteration += 1
        output = self.model(image)
        loss = self.criterion(output, target, weight)
        if self.use_adv:
            adv_x = self.attack.perturb(model=self.model, original_images=image, labels=target,
                                        device=target.device, criterion=self.criterion, random_start=False)
            adv_loss = self.criterion(self.model(adv_x), target)
            loss = (1 - self.adv_w) * loss + self.adv_w * adv_loss
        if torch.isnan(loss):
            print("nan loss, skipping step")
            loss = torch.zeros_like(loss).to(loss.device)
        else:
            loss /= self.step_decay
            loss.backward()

        if self.iteration % self.step_decay == 0:
            if self.use_grad_clip:
                clip_grad_norm_(self.model.parameters(), **self.grad_clip_params)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_policy.step(self.epoch)

        reduced_loss = reduce_tensor(loss.data, self.world_size) if self.distributed else loss.data
        return reduced_loss

    def step_no_grad(self, image):
        with torch.no_grad():
            output = self.model(image)
            depth, uncertainty = test_inference(self.loss_type, output, self.bin_values)
        return depth, uncertainty

    def before_epoch(self, epoch):
        synchronize()
        self.iteration = 0
        self.epoch = epoch
        self.model.train()
        gc.collect()
        torch.cuda.empty_cache()

    def after_epoch(self, epoch=None):
        synchronize()
        self.model.eval()
        gc.collect()
        torch.cuda.empty_cache()

    def save_checkpoint(self, path):
        if self.local_rank == 0:

            state_dict = {}

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in self.model.state_dict().items():
                key = k
                if k.split('.')[0] == 'module':
                    key = k[7:]
                new_state_dict[key] = v

            state_dict['config'] = self.config
            state_dict['model'] = new_state_dict
            state_dict['optimizer'] = self.optimizer.state_dict()
            state_dict['lr_policy'] = self.lr_policy.state_dict()
            state_dict['epoch'] = self.epoch
            state_dict['iteration'] = self.iteration

            torch.save(state_dict, path)
            del state_dict
            del new_state_dict

    def get_learning_rates(self):
        lrs = []
        for i in range(len(self.optimizer.param_groups)):
            lrs.append(self.optimizer.param_groups[i]['lr'])
        return lrs
