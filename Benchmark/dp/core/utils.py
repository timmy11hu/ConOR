import numpy as np
import torch
import torch.nn.functional as F


def pdf2depth(pdf, bin_values):
    # p = pdf[0].numpy()
    N, C, H, W = pdf.shape
    bin_values = bin_values[None, :, None, None].repeat(N, 1, H, W).to(pdf.device)
    mu = torch.sum(bin_values * pdf, dim=1).view(N, -1, H, W)
    sigma2 = torch.sum(torch.square(bin_values - mu) * pdf, dim=1)
    return mu.view(N, H, W), sigma2.view(N, H, W)


def cdf2depth(label, bin_values):
    N, H, W = label.shape

    num_bin = len(bin_values)
    label = torch.unsqueeze(label, dim=1)
    label = label.repeat(1, num_bin, 1, 1).contiguous().long()

    label_cls = torch.linspace(0, num_bin - 1, num_bin, requires_grad=False)
    label_cls = label_cls.view(1, num_bin, 1, 1).to(label.device)
    label_cls = label_cls.repeat(N, 1, H, W).contiguous().long()

    onehot_label = torch.where(label_cls == label, 1, 0)
    bin_values = bin_values[None, :, None, None].repeat(N, 1, H, W).to(label.device)
    pred = torch.sum(bin_values * onehot_label, dim=1)
    return pred, pred

def find_cutpoints2tensor(discretization, ord_num, alpha, beta, gamma):
    alpha_star = alpha + gamma
    beta_star = beta + gamma
    assert alpha_star == 1.

    if discretization == "SID":
        cutpoints = [
            np.exp(np.log(alpha_star) + ((np.log(beta_star) - np.log(alpha_star)) * float(b + 1) / ord_num))
            for b in range(ord_num)]
    elif discretization == "UD":
        cutpoints = [alpha_star + (beta_star - alpha_star) * (float(b + 1) / ord_num) for b in range(ord_num)]
    else:
        cutpoints = np.sort(np.random.uniform(low=alpha_star, high=beta_star, size=ord_num))

    cutpoints = torch.tensor(cutpoints, requires_grad=False) - gamma
    t0s = torch.cat((torch.tensor(alpha).view(-1), cutpoints), dim=0)
    t1s = torch.cat((cutpoints, torch.tensor(beta).view(-1)), dim=0)
    bin_values = (t0s + t1s) / 2
    print("cutpoints:", cutpoints)
    print("bin-values:", bin_values)
    return cutpoints, t0s, t1s, bin_values


def logits2depth(logits, bin_values):
    N, C, H, W = logits.size()
    pdf = F.softmax(logits, dim=1).view(N, C, H, W)
    bin_values = bin_values[None, :, None, None].repeat(N, 1, H, W).to(pdf.device)
    mu = torch.sum(bin_values * pdf, dim=1).view(N, -1, H, W)
    clipped_probs = torch.clamp(input=pdf, min=1e-7, max=1 - 1e-7)
    ent = -torch.sum(clipped_probs*torch.log(clipped_probs), dim=1)
    return mu.view(N, H, W), ent.view(N, H, W)


def probs2depth(probs, bin_values):
    N, C, H, W = probs.shape
    num_bin = len(bin_values)
    bin_values = bin_values[None, :, None, None].repeat(N, 1, H, W).to(probs.device)
    # label = torch.argmax(probs, dim=1).view(N, -1, H, W)
    confidence, label = torch.max(probs, dim=1)

    label = torch.unsqueeze(label, dim=1)
    label = label.repeat(1, num_bin, 1, 1).contiguous().long()

    label_cls = torch.linspace(0, num_bin-1, num_bin, requires_grad=False).view(1, num_bin, 1, 1).to(probs.device)
    label_cls = label_cls.view(1, num_bin, 1, 1).to(probs.device)
    label_cls = label_cls.repeat(N, 1, H, W).contiguous().long()

    onehot_label = torch.where(label_cls==label, 1, 0)
    pred = torch.sum(bin_values*onehot_label, dim=1)
    return pred.view(N, H, W), (1-confidence).view(N, H, W)


def test_inference(loss_type, output, bin_values):
    if loss_type == "conor":
        output[output < 0] = 0
        depth, uncertainty = pdf2depth(output, bin_values)
    elif loss_type == "bc":
        depth, uncertainty = logits2depth(output, bin_values)
    elif loss_type == "gll":
        output = torch.log(1 + torch.exp(output))
        depth, uncertainty = output[:, 0, :, :], output[:, 1, :, :]
    elif loss_type == "lgl":
        if torch.max(bin_values) > 20:
            upper_bound = 5
        else:
            upper_bound = 3
        mu_z,  = torch.clamp(output[:, 0, :, :]-1, min=0, max=upper_bound)
        sigma2_z = torch.clamp(output[:, 1, :, :], min=0, max=upper_bound)
        depth = torch.exp(mu_z + sigma2_z/2)
        uncertainty = (torch.exp(sigma2_z) - 1) * torch.exp(2*mu_z+sigma2_z)
    elif loss_type == "mcc":
        depth, uncertainty = probs2depth(output, bin_values)
    elif loss_type == "or":
        depth, uncertainty = pdf2depth(output, bin_values[1:-1])
    else:
        raise NotImplementedError
    return depth, uncertainty


