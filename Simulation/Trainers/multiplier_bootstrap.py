import numpy as np
from simulator import SimulationDataset
from torch.utils.data import DataLoader
import torch
import copy

from Trainers.base_trainer import Base_Trainer


class MultiplierTrainer(Base_Trainer):
    def __init__(self, config, base_net, ensembles, criterion, evaluation,
                 TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device):
        super().__init__(config, base_net, ensembles, criterion, evaluation,
                 TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device)
        self.w_sample = 'Gaussian'
        self.use_adv = False
        self.config.eps = 1
        self.config.alpha = 0.1
        # self.attack = FastGradientSignUntargeted(self.config.eps, self.config.alpha, min_val=0, max_val=10, max_iters=1)
        # self.adv_w = 0.5

    def __setup(self):
        # bootstrape weights
        if self.w_sample == 'Gaussian':
            w = np.random.normal(loc=1.0, scale=1.0, size=self.config.num_ensemble*self.TrainX.shape[0])
        elif self.w_sample == 'Uniform':
            w = np.random.uniform(low=0.0, high=2.0, size=self.config.num_ensemble * self.TrainX.shape[0])
        else:
            raise NotImplementedError

        self.w = w.reshape(self.config.num_ensemble, self.TrainX.shape[0])
        self.train_loaders = []
        for m in range(self.config.num_ensemble):
            w = self.w[m]
            TrainY = np.stack([self.TrainY, w]).T
            train_set = SimulationDataset(self.TrainX, TrainY)
            self.train_loaders.append(DataLoader(train_set, batch_size=self.config.batch_size,
                                                 shuffle=True, num_workers=0, drop_last=True, pin_memory=True))

    def ensembles_step(self, net, x, y):
        y_target, weights = torch.t(y)
        net.zero_grad()
        net.optim.zero_grad()
        x, y_target, weights = x.to(self.device), y_target.to(self.device), weights.to(self.device)
        y_probs = net(x)
        loss = self.criterion(y_probs, y_target, weights)
        if self.use_adv:
            adv_x = self.attack.perturb(model=net, original_images=x, labels=y_target,
                                        device=self.device, criterion=self.criterion, random_start=True)
            adv_y_probs = net(adv_x)
            adv_loss = self.criterion(adv_y_probs, y_target, weights)
            loss = (1-self.adv_w) * loss + self.adv_w * adv_loss
        loss.backward()
        net.optim.step()
        return loss

    def train(self):
        self.__setup()
        self.single_train()
        self.logger.midbreak()
        print("\nBase net training finished, now start Multiplier bootstrap\n")

        for net in self.ensembles:
            net.load_state_dict(copy.deepcopy(self.base_net.state_dict()))
        self.config.epoch = int(self.config.epoch / 5)
        self.ensembles_train()
        self.bootstrap_inference()
        print("Multiplier Train All done!")
