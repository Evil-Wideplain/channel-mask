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
from criterion.Cluster import ClusterLoss
from criterion.Svm import SvmLoss


def calculate_cls(sample, target, trained_backbone, classifier, criterion):
    output, feat = trained_backbone(sample)
    if len(feat.shape) == 3:
        feat = feat.reshape(feat.shape[0], -1)
    if classifier is not None:
        output = classifier(feat)
    if len(target.shape) == 1:
        target = F.one_hot(target.to(torch.int64), num_classes=output.shape[-1]).float()
    loss = criterion(output, target)
    _, predicted = torch.max(output.data, 1)
    return loss, feat, predicted


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
            if framework_n == 'tstcc':
                criterion = NTXentLoss(device, args.batch_size, temperature=0.2)
            else:
                criterion = NTXentLoss(device, args.batch_size, temperature=0.1)
            self.criterion = [criterion]
        elif criterion_n == 'cluster':
            criterion = ClusterLoss(args, device)
            self.criterion = [criterion]
        elif criterion_n == 'svm':
            criterion = SvmLoss(args, device)
            self.criterion = [criterion]
        elif criterion_n == 'svm_cluster':
            criterion1 = ClusterLoss(args, device)
            criterion2 = SvmLoss(args, device)
            self.criterion = [criterion1, criterion2]

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
        logits_aa = torch.mm(zis, zjs.T) / temperature
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

    def forward(self, sample, target, model):
        aug1 = args_contains(self.args, 'aug1', 'resample')
        aug2 = args_contains(self.args, 'aug2', 'na')
        aug_sample1 = gen_aug(sample, aug1, self.args)
        aug_sample2 = gen_aug(sample, aug2, self.args)
        aug_sample1, aug_sample2, target = aug_sample1.to(self.device).float(), aug_sample2.to(self.device).float(), \
                                           target.to(self.device).long()
        framework = args_contains(self.args, 'framework', 'simclr')
        backbone = args_contains(self.args, 'backbone', 'TPN')
        criterion_n = args_contains(self.args, 'criterion', 'NTXent')
        if framework in ['byol', 'simsiam']:
            assert criterion_n == 'cos_sim'
        if framework in ['tstcc', 'simclr', 'nnclr']:
            assert criterion_n in ['NTXent', 'cluster', 'svm', 'svm_cluster']

        lambda1 = args_contains(self.args, 'lambda1', 1.0)
        lambda2 = args_contains(self.args, 'lambda2', 1.0)
        if framework in ['byol', 'simsiam', 'nnclr']:
            if backbone in ['AE', 'CNN_AE']:
                x1_encoded, x2_encoded, p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
                if self.recon is not None:
                    recon_loss = self.recon(aug_sample1, x1_encoded) + self.recon(aug_sample2, x2_encoded)
            else:
                p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            if framework == 'nnclr':
                if self.nn_replacer is not None:
                    z1 = self.nn_replacer(z1, update=False)
                    z2 = self.nn_replacer(z2, update=True)

            if criterion_n == 'cos_sim':
                loss = -(self.calculate_loss(p1, z2).mean() + self.calculate_loss(p2, z1).mean()) * 0.5
            elif criterion_n == 'NTXent':
                loss = (self.calculate_loss(p1, z2) + self.calculate_loss(p2, z1)) * 0.5

            if backbone in ['AE', 'CNN_AE']:
                loss = loss * lambda1 + recon_loss * lambda2
        if framework == 'simclr':
            if backbone in ['AE', 'CNN_AE']:
                x1_encoded, x2_encoded, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
                recon_loss = self.recon(aug_sample1, x1_encoded) + self.recon(aug_sample2, x2_encoded)
            else:
                z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            loss = self.calculate_loss(z1, z2)
            if backbone in ['AE', 'CNN_AE']:
                loss = loss * lambda1 + recon_loss * lambda2
        if framework == 'tstcc':
            nce1, nce2, p1, p2 = model(x1=aug_sample1, x2=aug_sample2)
            tmp_loss = nce1 + nce2
            ctx_loss = self.calculate_loss(p1, p2)
            loss = tmp_loss * lambda1 + ctx_loss * lambda2
        return loss
