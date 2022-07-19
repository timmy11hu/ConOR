# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import torch

from dp.utils.wrappers import make_nograd_func
from dp.utils.pyt_ops import interpolate
from dp.utils.comm import reduce_dict
from dp.utils.average_meter import AverageMeterList

def log10(x):
    return torch.log(x.float()) / np.log(10.)

def cal_mae(pred, target):
    return torch.mean((pred - target).abs())

def cal_rmse(pred, target):
    return torch.sqrt(torch.mean(torch.square(pred - target)))

def cal_absrel(pred, target):
    return torch.mean((pred - target).abs()/target)

def cal_1_d1(pred, target):
    return 1 - (torch.max(pred / target, target / pred) < 1.25).float().mean()

def cal_sparsification(order, y_hat, gt, fraction, step, metric_func):
    # normalizer = 1.0
    normalizer = metric_func(y_hat, gt)
    if normalizer == 0.0:
        return None, None
    y_hat = y_hat[order]
    gt = gt[order]
    x_axis = []
    y_axis = []
    for i in range(fraction):
        y = y_hat[i * step:]
        target = gt[i * step:]
        x_axis.append(1/fraction * i)
        y_axis.append(metric_func(y, target)/normalizer)
    return torch.tensor(x_axis), torch.tensor(y_axis)


def cal_ause(output, target, uncertainty, fraction=100):
    step = int(len(target) / fraction)
    # # predictive
    # unc_order = torch.argsort(-uncertainty / output)
    # # oracle
    # oracle_order = torch.argsort(-torch.abs(output - target))
    # # random
    # np.random.seed(2022)
    # random_order = torch.tensor(np.random.choice(len(target), len(target), replace=False))
    metrics = {'rmse': cal_rmse, 'mae': cal_mae, 'absrel': cal_absrel, '1_d1': cal_1_d1}
    result = {}
    for metric_name, metric_func in metrics.items():
        if metric_name == 'absrel' or metric_name == '1_d1':
            unc_order = torch.argsort(-uncertainty/output)
            oracle_order = torch.argsort(-torch.abs(output - target)/target)
        else:
            unc_order = torch.argsort(-uncertainty)
            oracle_order = torch.argsort(-torch.abs(output - target))

        unc_x, unc_y = cal_sparsification(unc_order, output, target, fraction, step, metric_func)
        oracle_x, oracle_y = cal_sparsification(oracle_order, output, target, fraction, step, metric_func)

        if unc_x is None and oracle_x is None:
            ause_value, aurg_value = torch.zeros(1).to(target.device), torch.ones(1).to(target.device)
            unc_y, oracle_y = torch.zeros(100).to(target.device), torch.zeros(100).to(target.device)
        else:
            ause_value = torch.mean(unc_y - oracle_y).to(target.device)
            aurg_value = torch.mean(1.0 - unc_y).to(target.device)

        result['ause_' + metric_name] = ause_value
        result['aurg_' + metric_name] = aurg_value

        result['scurve_' + metric_name] = unc_y
        result['ocurve_' + metric_name] = oracle_y

    return result

@make_nograd_func
def compute_metric(output, target, uncertainty):
    # print("pred shape: {}, target shape: {}".format(pred.shape, target.shape))
    assert output.shape == target.shape, "pred'shape must be same with target."
    valid_mask = target > 0
    output = output[valid_mask]
    target = target[valid_mask]
    uncertainty = uncertainty[valid_mask]

    abs_diff = (output - target).abs()
    sqrel = (torch.pow(abs_diff, 2)/target).mean()
    mse = (torch.pow(abs_diff, 2)).mean()
    rmse = torch.sqrt(mse)
    mae = abs_diff.mean()
    absrel = (abs_diff / target).mean()

    lg10 = torch.abs(log10(output) - log10(target)).mean()

    err_log = torch.log(target) - torch.log(output)
    rmse_log = torch.sqrt((torch.pow(err_log, 2)).mean())
    normalized_squared_log = (err_log ** 2).mean()
    log_mean = err_log.mean()
    silog = torch.sqrt(normalized_squared_log - log_mean * log_mean) * 100

    maxRatio = torch.max(output / target, target / output)
    delta1 = (maxRatio < 1.25).float().mean()
    delta2 = (maxRatio < 1.25 ** 2).float().mean()
    delta3 = (maxRatio < 1.25 ** 3).float().mean()

    ause_aurg_dict = cal_ause(output, target, uncertainty)
    ause_rmse = ause_aurg_dict['ause_rmse']
    ause_absrel = ause_aurg_dict['ause_absrel']
    ause_1_d1 = ause_aurg_dict['ause_1_d1']
    ause_mae = ause_aurg_dict['ause_mae']

    aurg_rmse = ause_aurg_dict['aurg_rmse']
    aurg_absrel = ause_aurg_dict['aurg_absrel']
    aurg_1_d1 = ause_aurg_dict['aurg_1_d1']
    aurg_mae = ause_aurg_dict['aurg_mae']

    scurve_rmse = ause_aurg_dict['scurve_rmse']
    ocurve_rmse = ause_aurg_dict['ocurve_rmse']
    scurve_absrel = ause_aurg_dict['scurve_absrel']
    ocurve_absrel = ause_aurg_dict['ocurve_absrel']
    scurve_1_d1 = ause_aurg_dict['scurve_1_d1']
    ocurve_1_d1 = ause_aurg_dict['ocurve_1_d1']

    # inv_output = 1 / pred
    # inv_target = 1 / target
    # abs_inv_diff = (inv_output - inv_target).abs()
    # irmse = torch.sqrt((torch.pow(abs_inv_diff, 2)).mean())
    # imae = abs_inv_diff.mean()

    # inv_output_km = (1e-3 * output) ** (-1)
    # inv_target_km = (1e-3 * target) ** (-1)
    # abs_inv_diff = (inv_output_km - inv_target_km).abs()
    # irmse = torch.sqrt((torch.pow(abs_inv_diff, 2)).mean())
    # imae = abs_inv_diff.mean()

    metric = dict(mse=mse,
                  rmse=rmse,
                  mae=mae,
                  absrel=absrel,
                  lg10=lg10,
                  silog=silog,
                  delta1=delta1,
                  delta2=delta2,
                  delta3=delta3,
                  sqrel=sqrel,
                  rmse_log=rmse_log,
                  ause_rmse=ause_rmse,
                  ause_absrel=ause_absrel,
                  ause_1_d1=ause_1_d1,
                  ause_mae=ause_mae,
                  aurg_rmse=aurg_rmse,
                  aurg_absrel=aurg_absrel,
                  aurg_1_d1=aurg_1_d1,
                  aurg_mae=aurg_mae,
                  # scurve_rmse=scurve_rmse,
                  # ocurve_rmse=ocurve_rmse,
                  # scurve_absrel=scurve_absrel,
                  # ocurve_absrel=ocurve_absrel,
                  # scurve_1_d1=scurve_1_d1,
                  # ocurve_1_d1=ocurve_1_d1,
    )

    return metric


class Metrics(object):

    def __init__(self):
        self.distributed = False
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])
            self.distributed = self.world_size > 1 or torch.cuda.device_count() > 1

        # self.irmse = AverageMeterList()
        # self.imae = AverageMeterList()

        self.mse = AverageMeterList()
        self.mae = AverageMeterList()
        self.rmse = AverageMeterList()
        self.absrel = AverageMeterList()
        self.sqrel = AverageMeterList()
        self.rmse_log = AverageMeterList()

        self.lg10 = AverageMeterList()
        self.silog = AverageMeterList()

        self.d1 = AverageMeterList()
        self.d2 = AverageMeterList()
        self.d3 = AverageMeterList()

        self.ause_rmse = AverageMeterList()
        self.ause_absrel = AverageMeterList()
        self.ause_1_d1 = AverageMeterList()
        self.ause_mae = AverageMeterList()

        self.aurg_rmse = AverageMeterList()
        self.aurg_absrel = AverageMeterList()
        self.aurg_1_d1 = AverageMeterList()
        self.aurg_mae = AverageMeterList()

        self.scurve_rmse_list = []
        self.ocurve_rmse_list = []
        self.scurve_absrel_list = []
        self.ocurve_absrel_list = []
        self.scurve_1_d1_list = []
        self.ocurve_1_d1_list = []

        self.n_stage = 1

    def reset(self):
        # self.irmse.reset()
        # self.imae.reset()

        self.mse.reset()
        self.mae.reset()
        self.rmse.reset()
        self.absrel.reset()
        self.sqrel.reset()

        self.lg10.reset()
        self.silog.reset()
        self.rmse_log.reset()

        self.d1.reset()
        self.d2.reset()
        self.d3.reset()

        self.ause_rmse.reset()
        self.ause_absrel.reset()
        self.ause_1_d1.reset()
        self.ause_mae.reset()

        self.aurg_rmse.reset()
        self.aurg_absrel.reset()
        self.aurg_1_d1.reset()
        self.aurg_mae.reset()

        self.scurve_rmse_list = []
        self.ocurve_rmse_list = []
        self.scurve_absrel_list = []
        self.ocurve_absrel_list = []
        self.scurve_1_d1_list = []
        self.ocurve_1_d1_list = []

        # self.n_stage = -1

    def compute_metric(self, pred, target, uncertainty):
        # import torch.distributed as dist
        # print(dist.get_rank(), "Start metric")
        assert pred.shape == uncertainty.shape
        if target.shape != pred.shape:
            h, w = target.shape[-2:]
            # minibatch = interpolate(minibatch, size=(h, w), mode='bilinear', align_corners=True)
            pred = interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
            uncertainty = interpolate(uncertainty, size=(h, w), mode='bilinear', align_corners=True)
        # print(dist.get_rank(), "finish intetpolate")

        sqrel, rmse_log, mse, mae, rmse, absrel, lg10, silog, d1, d2, d3 =\
            [], [], [], [], [], [], [], [], [], [], []
        ause_rmse, ause_absrel, ause_1_d1, ause_mae, aurg_rmse, aurg_absrel, aurg_1_d1, aurg_mae =\
            [], [], [], [], [], [], [], []

        metirc_dict = compute_metric(pred, target, uncertainty)
        # print(dist.get_rank(), "finish compute")
        if self.distributed:
            metirc_dict = reduce_dict(metirc_dict)
        sqrel.append(metirc_dict["sqrel"].cpu())
        rmse_log.append(metirc_dict["rmse_log"].cpu())
        mse.append(metirc_dict["mse"].cpu())
        mae.append(metirc_dict["mae"].cpu())
        rmse.append(metirc_dict["rmse"].cpu())
        absrel.append(metirc_dict["absrel"].cpu())
        lg10.append(metirc_dict["lg10"].cpu())
        silog.append(metirc_dict["silog"].cpu())
        d1.append(metirc_dict["delta1"].cpu())
        d2.append(metirc_dict["delta2"].cpu())
        d3.append(metirc_dict["delta3"].cpu())

        ause_rmse.append(metirc_dict["ause_rmse"].cpu())
        ause_absrel.append(metirc_dict["ause_absrel"].cpu())
        ause_1_d1.append(metirc_dict["ause_1_d1"].cpu())
        ause_mae.append(metirc_dict["ause_mae"].cpu())

        aurg_rmse.append(metirc_dict["aurg_rmse"].cpu())
        aurg_absrel.append(metirc_dict["aurg_absrel"].cpu())
        aurg_1_d1.append(metirc_dict["aurg_1_d1"].cpu())
        aurg_mae.append(metirc_dict["aurg_mae"].cpu())


        self.sqrel.update(sqrel)
        self.rmse_log.update(rmse_log)
        self.mse.update(mse)
        self.mae.update(mae)
        self.rmse.update(rmse)
        self.absrel.update(absrel)
        self.lg10.update(lg10)
        self.silog.update(silog)
        self.d1.update(d1)
        self.d2.update(d2)
        self.d3.update(d3)

        self.ause_rmse.update(ause_rmse)
        self.ause_absrel.update(ause_absrel)
        self.ause_1_d1.update(ause_1_d1)
        self.ause_mae.update(ause_mae)

        self.aurg_rmse.update(aurg_rmse)
        self.aurg_absrel.update(aurg_absrel)
        self.aurg_1_d1.update(aurg_1_d1)
        self.aurg_mae.update(aurg_mae)


        # self.scurve_rmse_list.append(metirc_dict["scurve_rmse"].cpu().numpy())
        # self.ocurve_rmse_list.append(metirc_dict["ocurve_rmse"].cpu().numpy())
        # self.scurve_absrel_list.append(metirc_dict["scurve_absrel"].cpu().numpy())
        # self.ocurve_absrel_list.append(metirc_dict["ocurve_absrel"].cpu().numpy())
        # self.scurve_1_d1_list.append(metirc_dict["scurve_1_d1"].cpu().numpy())
        # self.ocurve_1_d1_list.append(metirc_dict["ocurve_1_d1"].cpu().numpy())



        del sqrel
        del rmse_log
        del mse
        del rmse
        del absrel
        del lg10
        del silog
        del d1
        del d2
        del d3
        del ause_rmse
        del ause_absrel
        del ause_1_d1
        del ause_mae
        del aurg_rmse
        del aurg_absrel
        del aurg_1_d1
        del aurg_mae


    def add_scalar(self, writer=None, tag="Test", epoch=0):
        if writer is None:
            return
        keys = ["stage_{}".format(i) for i in range(self.n_stage)]
        writer.add_scalars(tag + "/mse", dict(zip(keys, self.mse.mean())), epoch)
        writer.add_scalars(tag + "/rmse", dict(zip(keys, self.rmse.mean())), epoch)
        writer.add_scalars(tag + "/mae", dict(zip(keys, self.mae.mean())), epoch)
        writer.add_scalars(tag + "/absrml", dict(zip(keys, self.absrel.mean())), epoch)
        writer.add_scalars(tag + "/silog", dict(zip(keys, self.silog.mean())), epoch)
        writer.add_scalars(tag + "/d1", dict(zip(keys, self.d1.mean())), epoch)
        writer.add_scalars(tag + "/d2", dict(zip(keys, self.d2.mean())), epoch)
        writer.add_scalars(tag + "/d3", dict(zip(keys, self.d3.mean())), epoch)
        writer.add_scalars(tag + "/lg10", dict(zip(keys, self.lg10.mean())), epoch)
        writer.add_scalars(tag + "/sqrel", dict(zip(keys, self.sqrel.mean())), epoch)
        writer.add_scalars(tag + "/rmse_log", dict(zip(keys, self.rmse_log.mean())), epoch)

        writer.add_scalars(tag + "_ause" + "/rmse", dict(zip(keys, self.ause_rmse.mean())), epoch)
        writer.add_scalars(tag + "_ause" + "/absrel", dict(zip(keys, self.ause_absrel.mean())), epoch)
        writer.add_scalars(tag + "_ause" + "/1_d1", dict(zip(keys, self.ause_1_d1.mean())), epoch)
        writer.add_scalars(tag + "_ause" + "/mae", dict(zip(keys, self.ause_mae.mean())), epoch)
        print("AUSE: ", "rmse", self.ause_rmse.mean(), "mae", self.ause_mae.mean(),
                        "absrel", self.ause_absrel.mean(), "1-d1", self.ause_1_d1.mean())

        writer.add_scalars(tag + "_aurg" + "/rmse", dict(zip(keys, self.aurg_rmse.mean())), epoch)
        writer.add_scalars(tag + "_aurg" + "/absrel", dict(zip(keys, self.aurg_absrel.mean())), epoch)
        writer.add_scalars(tag + "_aurg" + "/1_d1", dict(zip(keys, self.aurg_1_d1.mean())), epoch)
        writer.add_scalars(tag + "_aurg" + "/mae", dict(zip(keys, self.aurg_mae.mean())), epoch)

        # scurve_rmse = np.mean(self.scurve_rmse_list, axis=0)
        # ocurve_rmse = np.mean(self.ocurve_rmse_list, axis=0)
        # scurve_absrel = np.mean(self.scurve_absrel_list, axis=0)
        # ocurve_absrel = np.mean(self.ocurve_absrel_list, axis=0)
        # scurve_1_d1 = np.mean(self.scurve_1_d1_list, axis=0)
        # ocurve_1_d1 = np.mean(self.ocurve_1_d1_list, axis=0)
        #
        # for i in range(100):
        #     writer.add_scalar('scurve_rmse', scurve_rmse[i], i)
        #     writer.add_scalar('ocurve_rmse', ocurve_rmse[i], i)
        #     writer.add_scalar('scurve_absrel', scurve_absrel[i], i)
        #     writer.add_scalar('ocurve_absrel', ocurve_absrel[i], i)
        #     writer.add_scalar('scurve_1_d1', scurve_1_d1[i], i)
        #     writer.add_scalar('ocurve_1_d1', ocurve_1_d1[i], i)

    def get_snapshot_info(self):
        info = "absrel: %.2f" % self.absrel.values()[-1] + "(%.2f)" % self.absrel.mean()[-1]
        info += " rmse: %.2f" % self.rmse.values()[-1] + "(%.2f)" % self.rmse.mean()[-1]
        return info

    def get_result_info(self):
        info = "absrel: %.3f" % self.absrel.mean()[-1] + \
               " rmse: %.3f" % self.rmse.mean()[-1] + \
               " silog: %.3f" % self.silog.mean()[-1] + \
               " ause_rmse: %.4f" % self.ause_rmse.mean()[-1]
        return info
