import numpy as np
from scipy import sparse


def adj_to_lap(A: sparse.csr_matrix) -> sparse.csr_matrix:
    """
    Computes the graph Laplacian from the sparse adjacency matrix.
    :param A: sparse adjacency matrix
    :return: graph Laplacian
    """
    degree_vec = adj_to_deg_vec(A)
    D = sparse.diags(degree_vec, 0)
    return D - A


def adj_to_deg_vec(A):
    """
    Computes the degree vector from an adjacency matrix.
    :return: degree vector
    """
    degree_mat = A.sum(axis=1)
    return np.squeeze(np.asarray(degree_mat))


def lap_quad_form(L: sparse.csr_matrix, s: np.ndarray):
    """
    Computes the Laplacian quadratic form. Measures signal smoothness on a graph.
    :param L: graph Laplacian matrix
    :param s: signal vector
    :return: Laplacian quadratic form
    """
    s = s.reshape(-1, 1)
    return (s.T @ L @ s).item()


def norm_lap_quad_form(L: sparse.csr_matrix, s: np.ndarray):
    """
    Computes the normalized Laplacian quadratic form. Measures signal smoothness on a graph.
    :param L: graph Laplacian matrix
    :param s: signal vector
    :return: normalized Laplacian quadratic form
    """
    s = s.reshape(-1, 1)
    return (lap_quad_form(L, s) / (s.T@s)).item()
