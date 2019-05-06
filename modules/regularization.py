import torch as th
from torch import nn as nn
import torch.nn.functional as F

def dropout_mask(size, p):
    return th.zeros(size).bernoulli_(1-p) / (1-p)

class RecurrentDropout(nn.Module):
    def __init__(self, size, p):
        super(RecurrentDropout, self).__init__()
        self.size = size
        self.p = p
        self.mask = dropout_mask(size, p)

    def reset(self):
        self.mask = dropout_mask(self.size, self.p)

    def forward(self, x):
        if self.training:
            return self.mask.expand_as(x).to(x) * x
        else:
            return x# / (1 - self.p)

