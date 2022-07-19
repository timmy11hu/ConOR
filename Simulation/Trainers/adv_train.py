from simulator import SimulationDataset
from torch.utils.data import DataLoader
import torch

from Trainers.base_trainer import Base_Trainer

class FastGradientSignUntargeted():
    b"""
        Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
        with iterative grad sign updates
    """

    def __init__(self, epsilon, alpha, min_val, max_val, max_iters, _type='linf'):
        # self.model.eval()

        # Maximum perturbation
        self.epsilon = epsilon
        # Movement multiplier per iteration
        self.alpha = alpha
        # Minimum value of the pixels
        self.min_val = min_val
        # Maximum value of the pixels
        self.max_val = max_val
        # Maximum numbers of iteration to generated adversaries
        self.max_iters = max_iters
        # The perturbation of epsilon
        self._type = _type

    def perturb(self, model, original_images, labels, device, criterion, random_start=False):
        # original_images: values are within self.min_val and self.max_val

        # The adversaries created from random close points to the original data
        if random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon).to(device)
            x = original_images + rand_perturb
            x.clamp_(self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x.requires_grad = True

        # max_x = original_images + self.epsilon
        # min_x = original_images - self.epsilon
        model.eval()
        with torch.enable_grad():
            for _iter in range(self.max_iters):
                outputs = model(x)
                loss = criterion(outputs, labels)

                grad_outputs = None
                grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs, only_inputs=True)[0]
                x.data += self.alpha * torch.sign(grads.data)

                '''
                # the adversaries' pixel value should within max_x and min_x due
                # to the l_infinity / l2 restriction
                x = project(x, original_images, self.epsilon, self._type)
                # the adversaries' value should be valid pixel value
                # x.clamp_(self.min_val, self.max_val)
                '''
        model.train()
        return x.detach()

class AdvTrainer(Base_Trainer):
    def __init__(self, config, base_net, ensembles, criterion, evaluation,
                 TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device):
        super().__init__(config, base_net, ensembles, criterion, evaluation,
                 TrainX, TrainY, TestX, TestY, EvalX, EvalY, writer, logger, device)

        # adv setup
        self.eps = 1
        self.alpha = 0.1
        self.attack = FastGradientSignUntargeted(self.eps, self.alpha, min_val=0, max_val=10, max_iters=1)
        self.adv_w = 0.5

    def __setup(self):
        # bootstrap dataset
        for m in range(self.config.num_ensemble):
            train_set = SimulationDataset(self.TrainX, self.TrainY)
            self.train_loaders.append(DataLoader(train_set, batch_size=self.config.batch_size,
                                                     shuffle=True, num_workers=0, drop_last=True, pin_memory=True))

    def ensembles_step(self, net, x, y_target):
        net.zero_grad()
        net.optim.zero_grad()
        x, y_target = x.to(self.device), y_target.to(self.device)
        y_probs = net(x)
        loss = self.criterion(y_probs, y_target)
        adv_x = self.attack.perturb(model=net, original_images=x, labels=y_target,
                                    device=self.device, criterion=self.criterion, random_start=True)
        adv_y_probs = net(adv_x)
        adv_loss = self.criterion(adv_y_probs, y_target)

        total_loss = (1-self.adv_w) * loss + self.adv_w * adv_loss
        total_loss.backward()
        net.optim.step()
        return total_loss

    def train(self):
        self.__setup()
        self.ensembles_train()
        self.ensembles_inference()
        print("Adv Train All done!")
