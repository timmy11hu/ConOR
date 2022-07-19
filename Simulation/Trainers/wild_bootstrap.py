import numpy as np
from simulator import SimulationDataset
from torch.utils.data import DataLoader
import torch
import copy
import utils

from Trainers.base_trainer import Base_Trainer


class WildTrainer(Base_Trainer):
    def __init__(self, config, base_net, ensembles, criterion, evaluation,
                 TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device):
        super().__init__(config, base_net, ensembles, criterion, evaluation,
                 TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device)
        self.epsilon_sample = 'Gaussian'

    def __sample(self):
        self.train_loaders = []
        # self.base_net.load_state_dict(copy.deepcopy(self.base_net.state_dict()))
        y_hat, res = self.cal_residuals(self.base_net)

        # bootstrape residuals
        for m in range(self.config.num_ensemble):
            if self.epsilon_sample == 'Gaussian':
                eps = np.random.normal(loc=0, scale=1, size=self.TrainY.shape[0])
            else:
                raise NotImplementedError

            # res[np.abs(self.res)<1] = np.sign(np.random.uniform(low=-1, high=1, size=np.sum(np.abs(self.res)<1)))
            TrainY = y_hat + eps * res
            train_set = SimulationDataset(self.TrainX, TrainY)
            self.train_loaders.append(DataLoader(train_set, batch_size=self.config.batch_size,
                                                 shuffle=True, num_workers=0, drop_last=True, pin_memory=True))

    def train(self):
        self.single_train()
        self.logger.midbreak()
        print("\nBase net training finished, now start Sampling residual\n")

        self.__sample()
        print("Datasets(residuals) sampling finished, now start Wild bootstrap\n")

        for net in self.ensembles:
            net.load_state_dict(copy.deepcopy(self.base_net.state_dict()))
        self.config.epoch = int(self.config.epoch / 5)
        self.ensembles_train()
        self.bootstrap_inference()
        print("Wild Train All done!")
