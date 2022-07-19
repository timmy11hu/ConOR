#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF



from dp.datasets.base_dataset import BaseDataset
from dp.datasets.utils import nomalize, PILLoader, KittiDepthLoader


class NYUv2(BaseDataset):

    def __init__(self, config, is_train=True, image_loader=PILLoader, depth_loader=KittiDepthLoader):
        super().__init__(config, is_train, image_loader, depth_loader)

        file_list = "./dp/datasets/lists/nyuv2_{}.list".format(self.split)
        with open(file_list, "r") as f:
            self.filenames = f.readlines()

    def _parse_path(self, index):
        spl = self.filenames[index].split()
        if len(spl) == 2:
            image_path, depth_path = spl
            weigh_path = None
        else:
            image_path, depth_path, weigh_path = spl
            weigh_path = os.path.join(self.weight_root, weigh_path)
        image_path = os.path.join(self.rgb_root, image_path)
        depth_path = os.path.join(self.depth_root, depth_path)
        return image_path, depth_path, weigh_path

    def _tr_preprocess(self, image, depth, weight=None):
        crop_h, crop_w = self.config["tr_crop_size"]
        # print(crop_h, crop_w)
        W, H = image.size
        dW, dH = depth.size
        assert W == dW and H == dH, \
            "image shape should be same with depth, but image shape is {}, depth shape is {}".format((H, W), (dH, dW))

        # 1. Random scaling
        scale = 1.0
        # scale = np.random.uniform(1.0, 1.2)
        # W = math.ceil(W*scale)
        # H = math.ceil(H*scale)
        # image = TF.resize(img=image, size=[H, W], interpolation=IM.BILINEAR)
        # depth = TF.resize(img=depth, size=[H, W], interpolation=IM.NEAREST)

        # 2. Random crop
        i, j, h, w = T.RandomCrop.get_params(image, output_size=(crop_h, crop_w))
        image = TF.crop(image, i, j, h, w)
        depth = TF.crop(depth, i, j, h, w)
        if weight is not None:
            # weight = TF.crop(weight, i, j, h, w)
            weight = weight[i:i + h, j:j + w]

        # 3. Random flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            depth = TF.hflip(depth)
            if weight is not None:
                weight = TF.hflip(weight)

        # 4. Normalize & Color jettering
        # train rgb: 0-254
        image = np.asarray(image).astype(np.float32) / 255.
        jetter = np.random.uniform(low=0.8, high=1.2, size=3)
        image *= jetter
        image = nomalize(image, type=self.config['norm_type'])
        image = image.transpose(2, 0, 1)

        depth = np.array(depth, dtype=float)
        # assert (np.max(depth) > 255)
        # depth = depth.astype(np.float) * 10.0 / 65535.0
        depth = depth.astype(np.float) * 10.0 / 255.0
        assert np.min(depth) >= 0
        assert np.max(depth) <= 10
        depth /= scale
        depth[depth == 0] = -1.
        depth[depth > 10] = -1.

        # if weight is not None:
        #     weight = torch.t(weight)

        return image, depth, weight, None

    def _te_preprocess(self, image, depth, weigh=None):
        crop_h, crop_w = self.config["te_crop_size"]
        # resize
        W, H = image.size
        dW, dH = depth.size
        assert W == dW and H == dH, \
            "image shape should be same with depth, but image shape is {}, depth shape is {}".format((H, W), (dH, dW))

        image = TF.center_crop(image, [crop_h, crop_w])
        depth = TF.center_crop(depth, [crop_h, crop_w])

        # image = image.crop((x, y, x + crop_w, y + crop_h))
        # depth = depth[dy:dy + crop_dh, dx:dx + crop_dw]
        # image_n = image_n.crop((dx, dy, dx + crop_dw, dy + crop_dh))

        # normalize
        image_raw = np.array(image.copy()).astype(np.float32)
        # test rgb: 0-255
        image = np.asarray(image).astype(np.float32) / 255.
        image = nomalize(image, type=self.config['norm_type'])
        image = image.transpose(2, 0, 1)

        depth = np.array(depth, dtype=float)
        assert (np.max(depth) > 255)
        depth = depth.astype(np.float) * 10.0 / 65535.0
        depth[depth == 0] = -1.
        depth[depth > 10] = -1.

        extra_dict = {"image_raw": image_raw}

        return image, depth, None, extra_dict
