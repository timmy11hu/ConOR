import torch

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
        # model.zero_grad()
        # model.optim.zero_grad()
        return x.detach()
