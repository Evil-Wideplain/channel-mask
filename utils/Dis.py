import numpy as np
import torch
from sklearn.metrics.pairwise import distance_metrics, kernel_metrics, pairwise_kernels
from sklearn.metrics.pairwise import *

PAIRWISE_DISTANCE_FUNCTIONS = {
    'cityblock': manhattan_distances,
    'cosine': cosine_distances,
    'euclidean': euclidean_distances,
    'haversine': haversine_distances,
    'l2': euclidean_distances,
    'l1': manhattan_distances,
    'manhattan': manhattan_distances,
    'precomputed': None,  # HACK: precomputed is always allowed, never called
    'nan_euclidean': nan_euclidean_distances,
}

PAIRWISE_KERNEL_FUNCTIONS = {
    'additive_chi2': additive_chi2_kernel,
    'chi2': chi2_kernel,
    'linear': linear_kernel,
    'polynomial': polynomial_kernel,
    'poly': polynomial_kernel,
    'rbf': rbf_kernel,
    'laplacian': laplacian_kernel,
    'sigmoid': sigmoid_kernel,
    'cosine': cosine_similarity,
}


def dis_func(X, Y, metrics='euclidean'):
    func = distance_metrics()[metrics]
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.numpy()
    return func(X, Y)


def pos_neg_dis(dis_array):
    if not isinstance(dis_array, np.ndarray):
        return dis_array
    num = dis_array.shape[0]
    pos_mask = np.eye(num, dtype=np.bool)
    # neg_mask = np.ones_like(dis_array) - np.eye(num)
    # neg_mask = neg_mask.astype(np.bool)
    neg_mask = ~pos_mask
    pos_distance = np.sum(np.abs(dis_array[pos_mask]))
    neg_distance = np.sum(np.abs(dis_array[neg_mask]))
    return pos_distance, neg_distance
