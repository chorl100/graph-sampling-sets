import typing

import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist

from src.graph import graph_tools


class GaussianSimilarityGraph:
    """
    A class to build a Gaussian similarity graph from a feature matrix.
    """

    def __init__(self, sigma: typing.Optional[float] = None, thres: float = 0.8, val_range=(0, 1),
                 n_rand_edges=10, rand_weight=0.1):
        """
        Initializes the configuration for the graph builder.
        :param sigma: neighborhood radius
        :param thres: distance threshold; edges between nodes with a larger distance than the threshold are discarded;
            controls the graph sparsity
        :param val_range: desired value range for the distances
        :param n_rand_edges: number of random edges to add per node to ensure connectivity
        :param rand_weight: weight of added random edges
        """
        self.sigma = sigma
        self.thres = thres
        self.val_range = val_range
        self.n_rand_edges = n_rand_edges
        self.rand_weight = rand_weight

    def build(self, X) -> sparse.csr_matrix:
        """
        Computes an adjacency matrix based on Gaussian similarity between the features.
        First the pairwise Euclidean distances between all features are computed,
        then the distances are scaled to avoid getting out infinity values from the exponential.
        Finally, the adjacency matrix is made sparser and the diagonal is set to zero.
        :param X: data matrix
        :return: adjacency matrix
        """
        # compute distance matrix
        dist = cdist(X, X, metric="euclidean")
        # min-max scale distances to range
        dist = min_max_scale(dist, self.val_range)
        if self.sigma is None:
            # set sigma to first quartile of distances
            # (as in Puy and Pérez - "Structured sampling and fast reconstruction on smooth graph signals")
            self.sigma = np.percentile(dist, 25.)
        # compute adjacency matrix based on Gaussian similarity
        adj = thres_gauss_kernel(dist, self.sigma, self.thres)
        # add random edges to ensure connectivity
        adj = graph_tools.add_random_edges(adj, self.n_rand_edges, self.rand_weight)
        adj = graph_tools.remove_self_loops(adj)
        adj = graph_tools.symmetrize_adj(adj)
        return sparse.csr_matrix(adj)


def min_max_scale(X, feature_range=(0, 1), axis=None):
    """
    Transforms values by scaling each value to a given range.
    :param X: data matrix
    :param feature_range: desired range of transformed data
    :param axis: axis used to scale along; If 0, independently scale each feature, if 1, scale each sample;
        Default is None, then the global min and max are used for scaling
    :return: scaled data matrix
    """
    a, b = feature_range
    return a + (b - a)*(X - X.min(axis=axis)) / (X.max(axis=axis) - X.min(axis=axis))


def thres_gauss_kernel(dist, sigma: float, thres: typing.Optional[float]):
    """
    Thresholded Gaussian kernel weighting function.
    :param dist: square distance matrix
    :param sigma: free parameter; neighborhood radius
    :param thres: optional threshold above which to discard edges
        (since a larger distance corresponds to a higher dissimilarity); controls graph sparsity
    :return: adjacency matrix
    """
    adj = gauss_kernel(dist, sigma)
    if thres is not None:
        # make graph sparser
        adj[dist > thres] = 0
    return adj


def gauss_kernel(dist, std: float):
    """
    Gaussian kernel weighting function.
    :param dist: square distance matrix
    :param std: free parameter; neighborhood radius
    :return: adjacency matrix
    """
    return np.exp(-dist**2 / std**2)
