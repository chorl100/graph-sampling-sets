import numpy as np
from scipy import sparse


class Graph:
    """
    Simple graph class.
    Stores a list of neighboring nodes and the weights to neighbors for acceleration of the BSGDA method.
    """

    def __init__(self, W):
        # weight matrix
        self.W = sparse.csr_matrix(W)
        self.num_nodes = self.W.shape[0]
        # (unweighted) binary adjacency matrix
        self.adj = (self.W > 0).astype(int)
        # Laplacian matrix (only computed if needed)
        self.lap = None
        # degree vector (number of neighbors of each node)
        self.deg = np.asarray(self.adj.sum(axis=1)).squeeze()
        # weighted degree vector (sum of edge weights of each node)
        self.deg_w = np.asarray(self.W.sum(axis=1)).squeeze()
        # adjacency list (for efficient neighborhood queries)
        self.neighbors = self._build_adj_list()
        self.neighbors_w = self._precompute_neighbor_weights()

    def _build_adj_list(self):
        """Converts an adjacency matrix into an adjacency list."""
        # create empty list for each node
        adj_list = [list() for _ in range(self.num_nodes)]
        # save indices of all neighbors per node
        for start_node, end_node in zip(*self.adj.nonzero()):
            adj_list[start_node].append(end_node)
        # cast every list to array
        for i in range(self.num_nodes):
            adj_list[i] = np.array(adj_list[i])
        return adj_list

    def _precompute_neighbor_weights(self):
        """
        Returns the weights to neighboring nodes for each node.
        """
        neighbors_w = np.empty(self.num_nodes, dtype=list)
        for i in range(self.num_nodes):
            neigh = self.neighbors[i]
            neighbors_w[i] = self._get_row(self.W, i)[neigh]
        return neighbors_w

    def _get_row(self, A, i):
        """Returns the ith row of a sparse matrix A."""
        return A.getrow(i).toarray().flatten()

    def laplacian(self) -> sparse.csr_matrix:
        """Returns Laplacian matrix. Stores the result in cache to avoid re-computation."""
        if self.lap is None:
            L = sparse.diags(self.deg_w, 0) - self.W
            self.lap = L
        return self.lap.copy()
