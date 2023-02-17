import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Softmax
from torch import optim
from torchvision import datasets
import torchvision.models as models
import torchvision.transforms as transforms

import numpy as np
import PIL
import matplotlib.pyplot as plt
import pandas as pd
import time

from aug.augmentations import gen_aug

'''
聚类算法:
    AffinityPropagation
    AgglomerativeClustering
    Birch
    DBSCAN
    OPTICS
    cluster_optics_dbscan
    cluster_optics_xi
    compute_optics_graph
    KMeans
    FeatureAgglomeration
    MeanShift
    MiniBatchKMeans
    SpectralClustering
    affinity_propagation
    dbscan
    estimate_bandwidth
    get_bin_seeds
    k_means
    kmeans_plusplus
    linkage_tree
    mean_shift
    spectral_clustering
    ward_tree
    SpectralBiclustering
    SpectralCoclustering
'''
from sklearn.cluster import *
from utils.Util import args_contains
from utils.SVM import get_mask, get_kmask, compute_kernel, pgd_with_nesterov, pgd_simple_short
from utils.Cluster import get_cluster


class NTXentLoss(nn.Module):

    def __init__(self, device, batch_size, temperature=0.1, use_cosine_similarity=True):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)
        self.batch_size = zis.shape[0]
        mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        similarity_matrix = self.similarity_function(representations, representations)
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)

        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class ClusterLoss(nn.Module):
    def __init__(self, args, device=torch.device('cuda:0')):
        super(ClusterLoss, self).__init__()
        self.args = args
        self.device = device
        # cluster
        self.cluster = get_cluster(args)

    def forward(self, zis, zjs):
        temperature = args_contains(self.args, 'temperature', 0.1)
        p1 = F.normalize(zis, p=2, dim=1).to(self.device)
        p2 = F.normalize(zjs, p=2, dim=1).to(self.device)

        bs = p1.shape[0]
        LARGE_NUM = 1e9

        cluster_data1 = p1.data.cpu()
        cluster_data2 = p2.data.cpu()

        pos_class = self.cluster.fit_predict(cluster_data1.numpy())
        masks_aa = torch.Tensor(
            np.array([i == pos_class for i in pos_class])).to(self.device)

        masks_aa.requires_grad = False
        # masks_ab = masks_aa - F.one_hot(torch.arange(0, bs), bs).to(self.device)
        masks_ab = masks_aa - torch.eye(bs).to(self.device)

        logits_ab = torch.mm(p1, p2.T) / temperature
        logits_aa = torch.mm(p1, p1.T) / temperature
        logits_aa = logits_aa - masks_aa * LARGE_NUM
        logits_ab = logits_ab - masks_ab * LARGE_NUM

        neg_class = self.cluster.fit_predict(cluster_data2.numpy())
        masks_bb = torch.Tensor(
            np.array([i == neg_class for i in neg_class])).to(self.device)
        # masks_bb.detach()
        masks_bb.requires_grad = False
        # maska_ba = masks_bb - F.one_hot(torch.arange(0, bs), bs).to(self.device)
        maska_ba = masks_bb - torch.eye(bs).to(self.device)
        logits_ba = torch.mm(p2, p1.T) / temperature
        logits_bb = torch.mm(p2, p2.T) / temperature
        logits_bb = logits_bb - masks_bb * LARGE_NUM
        logits_ba = logits_ba - maska_ba * LARGE_NUM

        labels = torch.arange(0, bs).to(self.device)
        # labels = F.one_hot(torch.arange(bs), bs * 2).float().to(self.device)

        # F.cross_entropy
        # TODO: 23 F.cross_entropy 默认输入的数据是logits数据，也就是未经过处理的初始数据，于是F.cross_entropy会对输入的数据进行log_softmax处理 softmax处理后，最小值就会变成一个很小的值 1/(1 + e^-t) 无线接近0
        loss_a = F.cross_entropy(target=labels,
                                 input=torch.cat([logits_ab, logits_aa], dim=1))
        loss_b = F.cross_entropy(target=labels,
                                 input=torch.cat([logits_ba, logits_bb], dim=1))
        return loss_a + loss_b


class SvmLoss(nn.Module):
    def __init__(self, args, device=torch.device('cuda:0')):
        super(SvmLoss, self).__init__()
        self.args = args
        self.device = device

    def forward(self, xis, xjs):
        zis = F.normalize(xis, p=2, dim=1)
        zjs = F.normalize(xjs, p=2, dim=1)
        bs = zis.shape[0]

        kernel_type = args_contains(self.args, 'kernel_type', 'rbf')
        sigma = args_contains(self.args, 'sigma', 0.1)
        K = compute_kernel(zis, torch.cat([zis, zjs], dim=0), kernel_type=kernel_type, gamma=sigma)

        with torch.no_grad():
            block = torch.zeros(bs, 2 * bs).bool().to(self.device)
            block[:bs, :bs] = True
            KK = torch.masked_select(K.detach(), block).reshape(bs, bs)

            no_diag = (1 - torch.eye(bs)).bool().to(self.device)
            KK_d0 = KK * no_diag
            KXY = -KK_d0.unsqueeze(1).repeat(1, bs, 1)
            KXY = KXY + KXY.transpose(2, 1)

            reg = args_contains(self.args, 'reg', 0.1)
            oneone = (torch.ones(bs, bs) + torch.eye(bs) * reg).to(self.device)
            Delta = (oneone + KK).unsqueeze(0) + KXY

            DD_KMASK = get_kmask(bs, device=self.device)
            DD = torch.masked_select(Delta, DD_KMASK).reshape(bs, bs - 1, bs - 1)

            C = args_contains(self.args, 'C', 1.0)
            if C == -1:
                alpha_y = torch.relu(torch.randn(bs, bs - 1, 1, device=DD.device))
            else:
                alpha_y = torch.relu(torch.randn(
                    bs, bs - 1, 1, device=DD.device)).clamp(min=0, max=C)

            solver_type = args_contains(self.args, 'solver_type', 'nesterov')
            eta = args_contains(self.args, 'eta', 1e-3)
            num_iter = args_contains(self.args, 'num_iter', 1000)
            use_norm = args_contains(self.args, 'use_norm', True)
            stop_condition = args_contains(self.args, 'stop_condition', 1e-2)
            one_bs = torch.ones(bs, bs - 1, 1).to(self.device)
            if solver_type == 'nesterov':
                alpha_y, iter_no, abs_rel_change, rel_change_init = pgd_with_nesterov(
                    eta, num_iter, DD, 2 * one_bs, alpha_y.clone(), C, use_norm=use_norm,
                    stop_condition=stop_condition)
            elif solver_type == 'vanilla':
                alpha_y, iter_no, abs_rel_change, rel_change_init = pgd_simple_short(
                    eta, num_iter, DD, 2 * one_bs, alpha_y.clone(), C, use_norm=use_norm,
                    stop_condition=stop_condition)

            alpha_y = alpha_y.squeeze(2)
            if C == -1:
                alpha_y = torch.relu(alpha_y)
            else:
                alpha_y = torch.relu(alpha_y).clamp(min=0, max=C).detach()
            alpha_x = alpha_y.sum(1)

        block12 = torch.zeros(bs, 2 * bs).bool().to(self.device)
        block12[:bs, bs:] = True
        Ks = torch.masked_select(K, block12).reshape(bs, bs)

        anchor_count = args_contains(self.args, 'anchor_count', 2)

        mask, logits_mask = get_mask(bs, anchor_count)
        eye = torch.eye(anchor_count * bs).to(self.device)
        pos_mask = mask[:bs, bs:].bool()
        neg_mask = (mask * logits_mask + 1) % 2
        neg_mask = neg_mask - eye
        neg_mask = neg_mask[:bs, bs:].bool()
        Kn = torch.masked_select(Ks.T, neg_mask).reshape(bs, bs - 1).T

        pos_loss = (alpha_x * (Ks * pos_mask).sum(1)).mean()
        neg_loss = (alpha_y.T * Kn).sum() / bs

        # 琢磨
        loss = torch.exp(neg_loss - pos_loss)
        # loss = neg_loss - pos_loss

        sparsity = (alpha_y == C).sum() / ((alpha_y > 0).sum() + 1e-10)
        num_zero = (alpha_y == 0).sum() / alpha_y.numel()
        # (Ks * pos_mask).sum(1).mean(), Kn.mean(), sparsity, num_zero, 0.0
        return loss


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
            if loss is None:
                loss = criterion(x1, x2)
            else:
                loss += criterion(x1, x2)
        if loss is None:
            loss = 0.0
        return loss

    def forward(self, sample, target, model):
        aug1 = args_contains(self.args, 'aug1', 'jit_scal')
        aug2 = args_contains(self.args, 'aug2', 'resample')
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
