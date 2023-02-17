import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

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
from utils.Cluster import get_cluster


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

