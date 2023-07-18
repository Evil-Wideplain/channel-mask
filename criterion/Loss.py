import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Softmax
from torch import optim
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms

from aug.augmentations import gen_aug
from utils.Util import args_contains
from criterion.NTXent import NTXentLoss


class Loss(nn.Module):
    def __init__(self, args, device=torch.device("cuda:0"), recon=None, nn_replacer=None):
        super(Loss, self).__init__()
        self.device = device
        self.args = args
        # loss fn
        criterion_n = args_contains(args, 'criterion', 'NTXent')
        framework_n = args_contains(args, 'framework', 'simclr')
        if criterion_n == 'cos_sim':
            criterion = nn.CosineSimilarity(dim=1)
            self.criterion = [criterion]
        elif criterion_n == 'NTXent':
            criterion = NTXentLoss(
                device, args.batch_size, temperature=0.1)
            self.criterion = [criterion]

        self.recon = recon
        self.nn_replacer = nn_replacer

    def NXTLoss(self, xis, xjs):
        temperature = args_contains(self.args, 'temperature', 0.1)
        zis = F.normalize(xis, p=2, dim=-1).to(self.device)
        zjs = F.normalize(xjs, p=2, dim=-1).to(self.device)

        bs = zis.shape[0]
        masks = F.one_hot(torch.arange(bs), bs).float().to(self.device)
        masks = torch.ones_like(masks) - masks

        labels = F.one_hot(torch.arange(bs), bs * 2).float().to(self.device)
        logits_aa = torch.mm(zis, zis.T) / temperature
        logits_aa = logits_aa * masks

        logits_bb = torch.mm(zjs, zjs.T) / temperature
        logits_bb = logits_bb * masks

        logits_ab = torch.mm(zis, zjs.T) / temperature
        logits_ba = torch.mm(zjs, zis.T) / temperature

        loss_a = F.cross_entropy(
            target=labels, input=torch.cat([logits_ab, logits_aa], dim=1))
        loss_b = F.cross_entropy(
            target=labels, input=torch.cat([logits_ba, logits_bb], dim=1))

        return loss_a + loss_b

    def calculate_loss(self, x1, x2):
        loss = None
        for criterion in self.criterion:
            if criterion is None:
                continue
            if loss is None:
                loss = criterion(x1, x2)
            else:
                loss += criterion(x1, x2)
        if loss is None:
            loss = 0.0
        return loss

    def forward(self, sample, model):
        aug1 = args_contains(self.args, 'aug1', 'resample')
        aug2 = args_contains(self.args, 'aug2', 'na')

        aug_sample1 = gen_aug(sample, aug1, self.args)
        aug_sample2 = gen_aug(sample, aug2, self.args)
        sample1_x, sample1_y = aug_sample1['x'].to(
            self.device).float(), aug_sample1['y'].to(self.device).long()
        sample2_x, sample2_y = aug_sample2['x'].to(
            self.device).float(), aug_sample2['y'].to(self.device).long()
        # aug_sample1, aug_sample2 = aug_sample1.to(self.device).float(), aug_sample2.to(self.device).float()
        # target = sample['y'].to(self.device).long()

        framework = args_contains(self.args, 'framework', 'simclr')
        backbone = args_contains(self.args, 'backbone', 'TPN')
        criterion_n = args_contains(self.args, 'criterion', 'NTXent')

        if framework == 'simclr':
            z1, z2 = model(x1=sample1_x, x2=sample2_x)
            loss = self.calculate_loss(z1, z2)

        return loss
