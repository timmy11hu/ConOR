from Trainers.base_trainer import Base_Trainer

class McTrainer(Base_Trainer):
    def __init__(self, config, base_net, ensembles, criterion, evaluation,
                 TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device):
        super().__init__(config, base_net, ensembles, criterion, evaluation,
                 TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device)

    def train(self):
        self.single_train()
        self.mc_inference()
        print("MC Train All done!")

