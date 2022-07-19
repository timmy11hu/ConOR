import numpy as np
from simulator import SimulationDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import utils
import torch
import copy
import os

from Trainers.base_trainer import Base_Trainer

class TruthTrainer(Base_Trainer):
    def __init__(self, config, base_net, ensembles, criterion, evaluation,
                 TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device):
        super().__init__(config, base_net, ensembles, criterion, evaluation,
                         TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device)

    def __setup(self):
        # bootstrap dataset
        for i in range(self.config.num_ensemble):
            x = self.TrainX[i * self.config.num_xtrain:(i + 1) * self.config.num_xtrain, :]
            y = self.TrainY[i * self.config.num_xtrain:(i + 1) * self.config.num_xtrain]
            train_set = SimulationDataset(x, y)
            train_loader = DataLoader(train_set, batch_size=self.config.batch_size, shuffle=True, num_workers=0,
                                      drop_last=True,
                                      pin_memory=True)
            self.train_loaders.append(train_loader)


        # use same initialization
        for net in self.ensembles:
            net.load_state_dict(copy.deepcopy(self.base_net.state_dict()))

    def train(self):
        self.__setup()
        self.ensembles_train()
        self.ensembles_inference()
        print("Truth Train All Done!")
