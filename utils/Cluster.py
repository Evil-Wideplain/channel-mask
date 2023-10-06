import numpy as np
import scipy.sparse as sp

from sklearn.base import ClusterMixin, TransformerMixin, BaseEstimator
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state
from sklearn.utils.extmath import stable_cumsum, row_norms
from sklearn.cluster import KMeans, Birch, DBSCAN

from utils.Util import args_contains


def _kmeans_plusplus(X, n_clusters, random_state, n_local_trials=None):
    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly and track index of point
    center_id = random_state.randint(n_samples)
    indices = np.full(n_clusters, -1, dtype=int)
    # x是稀疏矩阵类型吗
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]
    indices[0] = center_id

    # 计算相似度 (0, 2) 越小相似性越高
    closest_dist_sq = cosine_distances(centers[0, np.newaxis], X)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)

        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)
        distance_to_candidates = cosine_distances(X[candidate_ids], X)

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates,
                   out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]
        indices[c] = best_candidate

    return centers, indices


def EStep(X, centers, n_clusters, x_vector, centers_new, update_centers=True):
    # centers shape: n_cluster, features
    # X shape: bs, features
    # return shape: bs, n_cluster  范围：[0, 2]
    similarity_distance = cosine_distances(X, centers)
    labels = np.argmin(similarity_distance, axis=-1)
    if update_centers:
        for c in range(n_clusters):
            centers_new[c] = np.mean(x_vector[labels == c], axis=0)

    return labels, centers_new


def kmeans_single(X, centers_init, x_norms, max_iter=300, tol=1e-4, ):
    from numpy import sum
    n_samples = X.shape[0]
    n_features = X.shape[1]
    n_clusters = centers_init.shape[0]

    # n_clusters, features
    centers = centers_init
    centers_new = np.zeros_like(centers)
    labels = np.full(n_samples, -1, dtype=np.int32)
    labels_old = labels.copy()
    # 这个是用来判断新的中心点移动的距离是不是小于可容忍的范围
    # center_shift = np.zeros(n_clusters, dtype=X.dtype)

    x_norms = np.repeat(x_norms, n_features, axis=0).reshape(n_samples, n_features)

    x_vector = X / x_norms
    # x_vector = np.where(np.isinf(x_vector), np.ones_like(x_vector), x_vector)

    strict_convergence = False
    i = 0
    for i in range(max_iter):
        labels, centers_new = EStep(
            X, centers, n_clusters, x_vector, centers_new)
        # 对角线上旧与新中心直接的相似性

        # center_shift = cosine_distances(centers, centers_new)[np.eye(n_clusters, dtype=np.bool)]
        centers, centers_new = centers_new, centers
        if np.array_equal(labels, labels_old):
            strict_convergence = True
            break
        # else:
        #     center_shift_tot = center_shift.sum()
        #     if center_shift_tot <= tol:
        #         break
        labels_old[:] = labels

    if not strict_convergence:
        # 提前结束，需要更新一下新的标签
        labels, _ = EStep(X, centers, n_clusters, x_vector,
                          centers_new, update_centers=False)

    inertia = sum(sum(cosine_distances(
        centers[c].reshape(1, -1), X[labels == c])) for c in range(n_clusters))
    return labels, inertia, centers, i + 1


class CosineKMeans(ClusterMixin, TransformerMixin, BaseEstimator):
    def __init__(self, n_clusters=8, n_init=10, max_iter=300, tol=1e-4, random_state=None, copy_x=True):
        self.n_clusters = n_clusters
        self.copy_x = copy_x
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol

    def _init_centroids(self, X, random_state):
        n_samples = X.shape[0]
        n_clusters = self.n_clusters

        centers, _ = _kmeans_plusplus(X, n_clusters, random_state=random_state)
        if sp.issparse(centers):
            centers = centers.toarray()
        return centers

    def fit(self, X, y=None):
        X = self._validate_data(X, accept_sparse='csr',
                                dtype=[np.float64, np.float32],
                                order='C', copy=self.copy_x,
                                accept_large_sparse=False)
        random_state = check_random_state(self.random_state)
        # 范数 模
        x_norms = row_norms(X, squared=False)
        best_inertia = None

        for i in range(self.n_init):
            centers_init = self._init_centroids(X, random_state=random_state)

            labels, inertia, centers, n_iter_ = kmeans_single(
                X, centers_init, max_iter=self.max_iter, tol=self.tol, x_norms=x_norms)

            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_

        distinct_clusters = len(set(best_labels))

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_

    def transform(self, X):
        return

    def predict(self, X):
        return


def get_cluster(args):
    cluster = None
    cluster_type = args_contains(args, 'cluster_name', 'kmean')
    n_outputs = args_contains(args, 'n_outputs', 6)
    if 'kmean' in cluster_type:
        cluster = KMeans(n_clusters=n_outputs, init='k-means++')
    elif 'birch' in cluster_type:
        cluster = Birch(threshold=0.1, n_clusters=n_outputs)
    elif 'cosine' in cluster_type:
        n_outputs = args_contains(args, 'n_outputs', 6)
        n_init = args_contains(args, 'n_init', 10)
        max_iter = args_contains(args, 'max_iter', 300)
        tol = args_contains(args, 'tol', 1e-4)
        cluster = CosineKMeans(n_clusters=n_outputs, n_init=n_init, max_iter=max_iter, tol=tol)
    else:
        cluster = None
    return cluster
