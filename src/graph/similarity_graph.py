from abc import ABC

from scipy import sparse
from sklearn.neighbors import radius_neighbors_graph, kneighbors_graph


class SimilarityGraph(ABC):
    """Class for the construction of a feature-distance-based similarity graph using the scikit-learn methods."""

    def __init__(self):
        self.num_nodes = 0
        self.num_features = 0
        self.features = None
        self.adj_matrix = None
        self.graph = None

    def save(self, filepath):
        """
        Saves the sparse adjacency matrix to a file in '.npz' format.
        :param filepath: path to save file
        """
        assert self.adj_matrix is not None
        sparse.save_npz(filepath, self.adj_matrix)


class NearestNeighbor(SimilarityGraph):

    def __init__(self):
        super().__init__()

    def fit(self, X, k: int, mode='connectivity'):
        """
        Constructs a k-nearest-neighbor graph.
        :param X: array-like features
        :param k: number of neighbors for each sample
        :param mode: type of returned matrix.
            'connectivity' will return the connectivity matrix with ones and zeros,
            and 'distance' will return the distances between neighbors according to the given metric.
        """
        self.features = X
        self.num_nodes, self.num_features = X.shape
        self.adj_matrix = kneighbors_graph(X, k, mode=mode, include_self=False, n_jobs=-1)


class EpsilonNeighborhoodGraph(SimilarityGraph):

    def __init__(self):
        super().__init__()

    def fit(self, X, eps: float, mode='connectivity'):
        """
        Constructs an epsilon-neighborhood graph.
        :param X: array-like features
        :param eps: epsilon radius of neighborhoods
        :param mode: type of returned matrix.
            'connectivity' will return the connectivity matrix with ones and zeros,
            and 'distance' will return the distances between neighbors according to the given metric.
        """
        self.features = X
        self.num_nodes, self.num_features = X.shape
        self.adj_matrix = radius_neighbors_graph(X, eps, mode=mode, include_self=False, n_jobs=-1)
