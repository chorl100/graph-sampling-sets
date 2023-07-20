import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from src.graph import graph_tools
from src.graph.gauss_similarity_graph import gauss_kernel


class NearestNeighborGraph:
    """
    Class for efficient computation of the k-nearest neighbors graph.
    Edge weights are computed from a Gaussian kernel.
    """

    def __init__(self, n_neighbors: int = 50, gauss_percentile: int = 25):
        """

        :param n_neighbors: number of neighbors to consider per node
        :param gauss_percentile: parameter in the computation of the Gaussian kernel;
            controls the scaling of the squared distances between the nodes
        """
        self.n_neighbors = n_neighbors
        self.gauss_percentile = gauss_percentile
        self.W = None

    def build(self, data: pd.DataFrame):
        """
        Builds a k-nearest neighbor graph that is then passed through a Gaussian kernel.
        Constructing the k-nn graph yields the distances between the neighbors
        which are then transformed to similarity values with the Gaussian function.
        Construction as in
        "Structured sampling and fast reconstruction of smooth graph signals" by Puy and Pérez
        (https://www.semanticscholar.org/paper/Structured-sampling-and-fast-reconstruction-of-Puy-P%C3%A9rez/a1f1660c2e5afda2218f1428ab2dd3e7a4ba3177)
        """

        # Find nearest neighbors
        model_nn = NearestNeighbors(n_neighbors=self.n_neighbors+1, algorithm='auto', n_jobs=-1).fit(data)
        dist, indices = model_nn.kneighbors(data)

        # Construct adjacency matrix
        n = data.shape[0]
        # connect nearest neighbors
        edges = np.arange(n, dtype='uint')
        edges = np.repeat(edges[:, np.newaxis], indices.shape[1], 1)
        edges = (edges.flatten(), indices.flatten())
        # compute Gaussian weights
        std = np.percentile(dist[:, 1:], self.gauss_percentile)
        weights = gauss_kernel(dist, std).flatten()
        # put edges and weights together
        W = sparse.csr_matrix((weights, edges), shape=(n, n))
        W = graph_tools.remove_self_loops(W)
        W = graph_tools.symmetrize_adj(W)
        W = sparse.csr_matrix(W)
        self.W = W.copy()
        return W
