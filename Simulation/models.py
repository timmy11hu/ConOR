import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimulationNet(nn.Module):
    def __init__(self, dim_x, dim_encoding, dropout_rate, num_bin, cls_model, alpha, beta, discretization,
                 lr, B1=0.9, B2=0.999, adam_eps=1e-8, weight_decay=0):
        super().__init__()
        saved_args = locals()
        self.dim_x = dim_x
        self.dim_encoding = dim_encoding
        self.dropout_rate = dropout_rate
        self.num_bin = num_bin
        self.cls_model = cls_model
        self.lr = lr
        self.B1 = B1
        self.B2 = B2
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay
        self.hiddens_depth = 8
        self.hiddens_width = 100

        self.encoder = nn.Sequential(
            nn.Linear(self.dim_x, self.hiddens_width),
            nn.BatchNorm1d(self.hiddens_width),
            nn.ELU(),
        )

        for _ in range(self.hiddens_depth):
            self.encoder = nn.Sequential(
                self.encoder,
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hiddens_width, self.hiddens_width),
                nn.BatchNorm1d(self.hiddens_width),
                nn.ELU(),
            )

        self.encoder = nn.Sequential(
            self.encoder,
            nn.Linear(self.hiddens_width, self.dim_encoding),
        )

        if cls_model == "mcc":
            self.output_prob = nn.Sequential(
                nn.Linear(self.dim_encoding, self.num_bin + 1),
                nn.Softmax(dim=1),
            )
        elif cls_model == "or":
            self.output_prob = MultiBinaryLayer(dim_encoding=self.dim_encoding, num_bin=self.num_bin)
        elif cls_model == "conor":
            self.output_prob = MultiCumsumLayer(dim_encoding=self.dim_encoding, num_bin=self.num_bin)
        elif cls_model == "gl" or cls_model == "lgl":
            self.output_prob = nn.Sequential(
                nn.Tanh(),
                nn.Linear(self.dim_encoding, 2),
            )
        elif cls_model == "bc":
            self.output_prob = nn.Sequential(nn.Linear(self.dim_encoding, self.num_bin + 1))
        else:
            raise NotImplementedError

        if cls_model == "gl" or cls_model == "lgl":
            self.optim = torch.optim.SGD(params=self.parameters(), lr=self.lr)
        else:
            self.optim = torch.optim.Adam(params=self.parameters(), lr=self.lr, betas=(self.B1, self.B2),
                                      weight_decay=self.weight_decay, eps=self.adam_eps)

    def forward(self, input):
        encoding = self.encoder(input)
        probs = self.output_prob(encoding)
        return probs


class MultiBinaryLayer(nn.Module):
    def __init__(self, dim_encoding, num_bin):
        super(MultiBinaryLayer, self).__init__()
        self.dim_encoding = dim_encoding
        self.num_bin = num_bin
        self.class_logits = nn.Linear(self.dim_encoding, 2*self.num_bin)

    def forward(self, encoding):
        x = self.class_logits(encoding)
        N, C = x.size()
        assert C == 2*self.num_bin
        x = x.view(-1, 2, self.num_bin)
        probs = F.softmax(x, dim=1)
        return probs


class MultiCumsumLayer(nn.Module):
    def __init__(self, dim_encoding, num_bin):
        super(MultiCumsumLayer, self).__init__()
        self.dim_encoding = dim_encoding
        self.num_bin = num_bin
        self.class_logits = nn.Linear(self.dim_encoding, self.num_bin + 1)

    def forward(self, encoding):
        x = self.class_logits(encoding)
        N, C = x.size()
        assert C == self.num_bin + 1
        pdf = F.softmax(x, dim=1)
        return pdf

