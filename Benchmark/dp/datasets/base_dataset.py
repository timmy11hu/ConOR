
import torch
import numpy as np

import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self, config, is_train=True, image_loader=None, depth_loader=None):
        super(BaseDataset, self).__init__()
        self.config = config
        self.rgb_root = self.config['rgb_path']
        self.depth_root = self.config['depth_path']
        self.weight_root = self.config['weight_path']
        self.split = self.config["split"]
        self.split = self.split[0] if is_train else self.split[1]
        self.image_loader, self.depth_loader = image_loader, depth_loader
        if is_train:
            self.preprocess = self._tr_preprocess
        else:
            self.preprocess = self._te_preprocess

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image_path, depth_path, weigh_path = self._parse_path(index)
        item_name = '_'.join([depth_path.split("/")[3], image_path.split("/")[-1].split(".")[0]])
        image_file, depth_file = self._fetch_data(image_path, depth_path)
        weight = None
        if weigh_path is not None:
            weight = torch.tensor(np.load(weigh_path))

        image, depth, weight, extra_dict = self.preprocess(image_file, depth_file, weight)
        image = torch.from_numpy(np.ascontiguousarray(image)).float()
        depth = torch.from_numpy(np.ascontiguousarray(depth)).float()

        if weight is None:
            output_dict = dict(image=image,
                               fn=str(item_name),
                               image_path=image_path,
                               n=self.get_length(),
                               target=depth,
                               target_path=depth_path)
        else:
            output_dict = dict(image=image,
                               fn=str(item_name),
                               image_path=image_path,
                               n=self.get_length(),
                               target=depth,
                               target_path=depth_path,
                               weight=weight)

        if extra_dict is not None:
            output_dict.update(**extra_dict)

        return output_dict

    def _fetch_data(self, image_path, depth_path, weight_path=None):
        image_file = self.image_loader(image_path)
        depth_file = self.depth_loader(depth_path)
        return image_file, depth_file

    def _parse_path(self, index):
        raise NotImplementedError

    def get_length(self):
        return self.__len__()

    def _tr_preprocess(self, image, depth, weigh=None):
        raise NotImplementedError

    def _te_preprocess(self, image, depth, weigh=None):
        raise NotImplementedError
