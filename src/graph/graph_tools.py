import numpy as np
from scipy import sparse


def add_random_edges(adj: sparse.csr_matrix, num_edges: int, weight: float = 0.1):
    """
    Adds random edges to the adjacency matrix to ensure connectivity.
    :param adj: adjacency matrix
    :param num_edges: number of random edges to add per node
        (note: the final graph can have more additional edges due to symmetrization)
    :param weight: edge weight of random edges
    :return: adjacency matrix with more edges (not necessarily connected)
    """
    n = adj.shape[0]
    # find isolated nodes (sum of weights is either 1 (self loops) or 0)
    isolated_nodes = np.flatnonzero(np.sum(adj, axis=1) <= 1)
    # add random edges
    adj = sparse.lil_matrix(adj)
    for i in isolated_nodes:
        adj[i, np.random.randint(0, n, num_edges)] = weight
    return adj


def remove_self_loops(adj: sparse.csr_matrix):
    """
    Sets the diagonal entries of the adjacency matrix to zero
    which is equivalent to removing all self-loops.
    :param adj: adjacency matrix
    :return: adjacency matrix without self-loops
    """
    adj_copy = adj.copy()
    for i in range(adj.shape[0]):
        adj_copy[i, i] = 0
    adj_copy.eliminate_zeros()
    return adj_copy


def symmetrize_adj(A, weighted=True):
    """
    Symmetrizes an adjacency matrix A.
    Either by computing the boolean sum (logical OR) of A and its transpose (only for unweighted graphs) or
    by computing the maximum of A and its transpose.
    :param A: adjacency matrix
    :param weighted: whether the graph is weighted
    :return: symmetric adjacency matrix
    """
    if not weighted:
        return (A.astype(bool) + A.astype(bool).T).astype(A.dtype)
    return A.maximum(A.T)
