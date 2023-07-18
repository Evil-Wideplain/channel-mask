import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
from models.backbones import *
from models.TC import *
from utils.Dis import dis_func

# 监督学习框架


class SupervisedLearning(nn.Module):
    def __init__(self, backbone, dim):
        super(SupervisedLearning, self).__init__()
        self.encoder = backbone
        self.bb_dim = self.encoder.out_dim

    def forward(self, x, y=None):
        o, _ = self.encoder(x)
        return o


class SimCLR(nn.Module):
    def __init__(self, backbone, dim=128):
        super(SimCLR, self).__init__()

        self.encoder = backbone
        self.bb_dim = self.encoder.out_dim
        self.projector = Projector(
            model='SimCLR', bb_dim=self.bb_dim, prev_dim=self.bb_dim, dim=dim)
        self.distance = 0.0

    def forward(self, x1, x2):
        if self.encoder.__class__.__name__ in ['AE', 'CNN_AE']:
            x1_encoded, z1 = self.encoder(x1)
            x2_encoded, z2 = self.encoder(x2)
        else:
            _, z1 = self.encoder(x1)
            _, z2 = self.encoder(x2)

        if len(z1.shape) == 3:
            z1 = z1.reshape(z1.shape[0], -1)
            z2 = z2.reshape(z2.shape[0], -1)
        with torch.no_grad():
            # euclidean 之前是欧几里得距离 一直呈增大趋势 换成cosine看看效果
            self.distance = dis_func(z1.clone().detach().cpu().numpy(
            ), z2.clone().detach().cpu().numpy(), metrics='cosine')
        z1 = self.projector(z1)
        z2 = self.projector(z2)

        if self.encoder.__class__.__name__ in ['AE', 'CNN_AE']:
            return x1_encoded, x2_encoded, z1, z2
        else:
            return z1, z2
