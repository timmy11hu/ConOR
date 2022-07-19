
from simulator import SimulationDataset
from torch.utils.data import DataLoader
from Trainers.base_trainer import Base_Trainer


class PairTrainer(Base_Trainer):
    def __init__(self, config, base_net, ensembles, criterion, evaluation,
                 TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device):
        super().__init__(config, base_net, ensembles, criterion, evaluation,
                 TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device)


    def __setup(self):
        # bootstrap dataset
        for m in range(self.config.num_ensemble):
            train_set = SimulationDataset(self.TrainX, self.TrainY)
            self.train_loaders.append(DataLoader(train_set, batch_size=self.config.batch_size,
                                                     shuffle=True, num_workers=0, drop_last=True, pin_memory=True))

    def train(self):
        self.__setup()
        self.ensembles_train()
        self.ensembles_inference()
        print("Pair Bootstrape Train All done!")

