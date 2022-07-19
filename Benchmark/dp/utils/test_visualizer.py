#!/usr/bin/python3
# -*- coding: utf-8 -*-
import torch

from dp.utils.pyt_ops import tensor2numpy, interpolate
import numpy as np
import matplotlib.pyplot as plt
from dp.utils.fill_depth_colorization import fill_depth_colorization


def padding_img(img_array, h=10, w=10):
    top_pad = np.floor(h / 2).astype(np.uint16)
    bottom_pad = np.ceil(h / 2).astype(np.uint16)
    right_pad = np.ceil(w / 2).astype(np.uint16)
    left_pad = np.floor(w / 2).astype(np.uint16)
    return np.copy(
        np.pad(
            img_array,
            ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
            mode='constant', constant_values=255)
    )

def depth_to_color(depth):
    cmap = plt.cm.jet
    d_min = np.min(depth)
    d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C

def uncertainty_to_color(variance):
    cmap = plt.cm.jet
    d_min = np.min(variance)
    d_max = np.max(variance)
    depth_relative = (variance - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]

def error_to_color(depth, gt):
    mask = gt <= 0.
    cmap = plt.cm.jet
    err = np.abs(depth-gt)
    err[mask] = 0.
    err_min = np.min(err)
    err_max = np.max(err)
    err_rel = (err-err_min) / (err_max-err_min)
    return 255 * cmap(err_rel)[:, :, :3]


def normalize_visual(img):
    img = img.transpose((2, 0, 1)) / 255.0
    img = img.astype(np.float32)
    return img

class Visualizer():
    def __init__(self, config, writer=None):
        # self.config = config["vis_config"]
        self.writer = writer

    def visualize(self, batch, pred, epoch=0, y_var=None, m_var=None):
        """
            :param batch_in: minibatch
            :param pred_out: model output for visualization, dic, {"target": [NxHxW]}
            :param tensorboard: if tensorboard = True, the visualized image should be in [0, 1].
            :return: vis_ims: image for visualization.
            """
        fn = batch["fn"]
        # assert batch["target"].shape == pred.shape
        if batch["target"].shape != pred.shape:
            h, w = batch["target"].shape[-2:]
            pred = interpolate(pred, size=(h, w), mode='bilinear', align_corners=True)
            y_var = interpolate(y_var, size=(h, w), mode='bilinear', align_corners=True)

        images = batch["image_raw"].numpy()
        depth_gts = tensor2numpy(batch["target"])

        for i in range(len(fn)):
            image = images[i].astype(np.float)
            depth = tensor2numpy(pred[i])
            depth_gt = depth_gts[i]

            assert depth.shape == depth_gt.shape
            mask = depth_gt == -1

            line = np.where(np.sum(depth_gt, axis=1) != -1*depth_gt.shape[1])[0][0]
            depth_gt[mask] = 0
            depth_gt = fill_depth_colorization(imgRgb=image/255, imgDepthInput=depth_gt)
            depth_gt[:line, :] = -1
            mask = depth_gt == -1

            # to color
            depth_error = error_to_color(depth, depth_gt)
            depth = depth_to_color(depth)
            depth_gt = depth_to_color(depth_gt)

            unc_y = tensor2numpy(y_var[i])
            if m_var is None:
                unc_m = np.zeros(shape=y_var[i].shape)
            else:
                unc_m = tensor2numpy(m_var[i])
            unc_all = unc_y + unc_m
            unc_clp = np.copy(unc_all)
            unc_clp[mask] = 0

            unc_y = uncertainty_to_color(unc_y)
            unc_m = uncertainty_to_color(unc_m)
            unc_all = uncertainty_to_color(unc_all)
            unc_clp = uncertainty_to_color(unc_clp)

            mask = np.repeat(np.expand_dims(mask, axis=2), repeats=3, axis=2)
            depth_gt[mask] = 0
            depth_error[mask] = 0
            unc_clp[mask] = 0

            group_mean = np.concatenate((padding_img(image),
                                         padding_img(depth_gt),
                                         padding_img(depth),
                                         padding_img(depth_error)), axis=0)

            group_unc = np.concatenate((padding_img(unc_y),
                                        padding_img(unc_m),
                                        padding_img(unc_all),
                                        padding_img(unc_clp)), axis=0)

            group = np.concatenate((group_mean, group_unc), axis=1)

            if self.writer is not None:
                self.writer.add_image(fn[i] + "/image", normalize_visual(group), epoch)
                self.writer.add_image(fn[i] + "/image1", normalize_visual(image), epoch)
                self.writer.add_image(fn[i] + "/image2", normalize_visual(depth_gt), epoch)
                self.writer.add_image(fn[i] + "/image3", normalize_visual(depth), epoch)
                self.writer.add_image(fn[i] + "/image4", normalize_visual(depth_error), epoch)
                self.writer.add_image(fn[i] + "/image5", normalize_visual(unc_y), epoch)
                self.writer.add_image(fn[i] + "/image6", normalize_visual(unc_m), epoch)
                self.writer.add_image(fn[i] + "/image7", normalize_visual(unc_all), epoch)
                self.writer.add_image(fn[i] + "/image8", normalize_visual(unc_clp), epoch)






