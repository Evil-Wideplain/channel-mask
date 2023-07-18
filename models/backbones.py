import torch
from torch import nn
from .MMB import *
from collections import OrderedDict


class FCN(nn.Module):
    def __init__(self, n_channels, n_timesteps, n_classes, out_channels=128, backbone=True):
        super(FCN, self).__init__()

        self.backbone = backbone

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=(8,),
                      stride=(1,), bias=False, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35))
        self.conv_block2 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=(8,), stride=(1,), bias=False, padding=4),
                                         nn.BatchNorm1d(64),
                                         nn.ReLU(),
                                         nn.MaxPool1d(kernel_size=2, stride=2, padding=1))
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, out_channels, kernel_size=(8,),
                      stride=(1,), bias=False, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1))

        # if n_channels == 9:  # ucihar
        #     self.out_len = 18
        # elif n_channels == 3:  # shar
        #     self.out_len = 21
        # if n_channels == 6:  # hhar
        #     self.out_len = 15
        # if n_channels == 12:
        #     self.out_len = 27
        # if n_channels == 52:
        #     self.out_len = 23
        # if n_channels == 77:
        #     self.out_len = 6
        # if n_channels == 3:
        #     self.out_len = 13

        self.out_len = n_timesteps
        for _ in range(3):
            self.out_len = (self.out_len + 1) // 2 + 1

        self.out_channels = out_channels
        self.out_dim = self.out_len * self.out_channels

        if not backbone:
            self.logits = Classifier(self.out_dim, n_classes)

    def forward(self, x_in):
        x_in = x_in.permute(0, 2, 1)
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        if self.backbone:
            return None, x
        else:
            x_flat = x.reshape(x.shape[0], -1)
            logits = self.logits(x_flat)
            return logits, x


class Classifier(nn.Module):
    def __init__(self, bb_dim, n_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(bb_dim, n_classes)
        self.softmax = nn.Softmax(dim=1)
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()

    def forward(self, x):
        return self.softmax(self.classifier(x))


class Projector(nn.Module):
    def __init__(self, model, bb_dim, prev_dim, dim):
        super(Projector, self).__init__()
        if model == 'SimCLR':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim))
        elif model == 'byol':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(
                                               prev_dim, dim, bias=False),
                                           nn.BatchNorm1d(dim, affine=False))
        elif model == 'NNCLR':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(
                                               prev_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(
                                               prev_dim, dim, bias=False),
                                           nn.BatchNorm1d(dim))
        elif model == 'TS-TCC':
            self.projector = nn.Sequential(nn.Linear(dim, bb_dim // 2),
                                           nn.BatchNorm1d(bb_dim // 2),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(bb_dim // 2, bb_dim // 4))
        else:
            # 默认一层线性层
            self.projector = nn.Sequential(nn.Linear(bb_dim, dim))
            # raise NotImplementedError

    def forward(self, x):
        x = self.projector(x)
        return x


class Predictor(nn.Module):
    def __init__(self, model, dim, pred_dim):
        super(Predictor, self).__init__()
        self.predictor = None
        if model == 'SimCLR':
            pass
        elif model == 'byol':
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim),
                                           nn.BatchNorm1d(pred_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(pred_dim, dim))
        elif model == 'NNCLR':
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim),
                                           nn.BatchNorm1d(pred_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(pred_dim, dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        if self.predictor is not None:
            x = self.predictor(x)
        return x
