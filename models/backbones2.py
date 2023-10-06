from logging import warning
import os
from turtle import forward
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import LSTM
from torch import optim
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import warnings
from collections import OrderedDict


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        self.is_padding = 0
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.AvgPool2d((2, 1), (2, 2))
            if in_planes != self.expansion * planes:
                self.is_padding = 1

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.is_padding:
            shortcut = self.shortcut(x)
            out += torch.cat([shortcut, torch.FloatTensor(torch.zeros(shortcut.shape)).to(x.device)], 1)
        else:

            out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_Z(nn.Module):
    def __init__(self, block, num_blocks, n_classes, backbone=True):
        super(ResNet_Z, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.backbone = backbone
        self.out_dim = 512
        if not backbone:
            self.classifier = nn.Linear(512, n_classes)
        self.l2norm = Normalize(2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.unsqueeze(1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, (16, 2))
        x = out.view(out.size(0), -1)  # (256,512)
        if not self.backbone:
            out = self.classifier(x)
            return out, x
        else:
            return None, x


class Flatten(nn.Module):
    def __init__(self, dim=-1):
        super(Flatten, self).__init__()
        self.dim = dim

    def forward(self, feat):
        return torch.flatten(feat, start_dim=self.dim)


class ResNetEncoder(models.resnet.ResNet):
    """Wrapper for TorchVison ResNet Model
    This was needed to remove the final FC Layer from the ResNet Model"""

    def __init__(self, block, layers, cifar_head=False, hparams=None):
        super().__init__(block, layers)
        self.cifar_head = cifar_head
        if cifar_head:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            self.bn1 = self._norm_layer(64)
            self.relu = nn.ReLU(inplace=True)
        self.hparams = hparams

        # print('** Using avgpool **')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.cifar_head:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class ResNet18(ResNetEncoder):
    def __init__(self, cifar_head=True):
        super().__init__(models.resnet.BasicBlock,
                         [2, 2, 2, 2], cifar_head=cifar_head)


class ResNet50(ResNetEncoder):
    def __init__(self, cifar_head=True, hparams=None):
        super().__init__(models.resnet.Bottleneck, [
            3, 4, 6, 3], cifar_head=cifar_head, hparams=hparams)


class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False


class Resnet(nn.Module):
    def __init__(self, base_net='resnet50'):
        super(Resnet, self).__init__()
        # simclr中两侧映射头，最后进行下游任务的时候要去掉
        if (base_net == 'resnet50'):
            self.convnet = ResNet50()
            self.encoder_dim = 2048
        elif (base_net == 'resnet18'):
            self.convnet = ResNet18()
            self.encoder_dim = 512

        self.proj_dim = 128
        projection_layers = [
            ('fc1', nn.Linear(self.encoder_dim, self.encoder_dim, bias=False)),
            ('bn1', nn.BatchNorm1d(self.encoder_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.encoder_dim, self.proj_dim, bias=False)),
            ('bn2', BatchNorm1dNoBias(self.proj_dim)),
        ]

        self.projection = nn.Sequential(OrderedDict(projection_layers))
        self.reset_parameters()

    def reset_parameters(self):
        def conv2d_weight_truncated_normal_init(p):
            fan_in = p.shape[1]
            stddev = np.sqrt(1. / fan_in) / .87962566103423978
            r = scipy.stats.truncnorm.rvs(-2, 2, loc=0, scale=1., size=p.shape)
            r = stddev * r
            with torch.no_grad():
                p.copy_(torch.FloatTensor(r))

        def linear_normal_init(p):
            with torch.no_grad():
                p.normal_(std=0.01)

        for m in self.modules():
            # print("modules have {} model".format(m))
            if isinstance(m, nn.Conv2d):
                conv2d_weight_truncated_normal_init(m.weight)
            elif isinstance(m, nn.Linear):
                linear_normal_init(m.weight)

    def forward(self, x):
        feature = self.convnet(x)
        out = self.projection(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class HCL(nn.Module):
    def __init__(self, feature_dim=128):
        super(HCL, self).__init__()

        self.f = []
        for name, module in models.resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=(3,), stride=(1,), padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.convnet = nn.Sequential(*self.f)
        # projection head
        self.projection = nn.Sequential(nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512),
                                        nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.convnet(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.projection(feature)
        # F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
