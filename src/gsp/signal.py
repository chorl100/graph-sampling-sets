import math
from typing import Optional

import networkx as nx
import numpy as np
from scipy import sparse
import scipy.sparse.linalg


def gs1(L: sparse.csr_matrix, size: int = 50) -> np.ndarray:
    """
    Generates a graph signal of type GS1 of shape (n, size).
    Description from the paper:
    "GS1: The true signals are exactly $\omega$-bandlimited,
    where $\omega = \theta_{N/10}$ is the $N/10$-th eigenvalue of L.
    [...] The non-zero GFT coefficients are randomly generated from [the normal distribution] $N(0, 10)$."
    :param L: graph Laplacian matrix of shape (n, n)
    :param size: number of signals to generate
    :return: signal matrix of shape (n, size) where each column corresponds to a signal vector
    """
    # compute eigenvectors corresponding to up to the n/10-th smallest eigenvalue
    k = math.floor(L.shape[0] / 10.)
    return bandlimited_signal(L, k, size, 0, 10)


def bandlimited_signal(L, k: int, size: int = 50, gft_coef_mean=0, gft_coef_var=10):
    """
    Generates a smooth, k-bandlimited signal by taking a linear combination
    of the first k eigenvectors of the graph Laplacian.
    The graph Fourier coefficients are generated randomly from a normal distribution.
    :param L: graph Laplacian matrix of shape (n, n)
    :param k: number of eigenvectors to use
    :param size: number of signals to generate
    :param gft_coef_mean: mean of the normal distribution that generates the random coefficients
    :param gft_coef_var: variance of the normal distribution that generates the random coefficients
    :return: signal matrix of shape (n, size) where each column corresponds to a signal vector
    """
    # compute eigenvectors corresponding to up to the k-th smallest eigenvalue
    eigvals, eigvecs = sparse.linalg.eigsh(L, k=k, which='SM')
    # randomly generate GFT coefficients
    gft_coeffs = np.random.normal(gft_coef_mean, np.sqrt(gft_coef_var), (size, k))
    signals = list()
    for coeffs in gft_coeffs:
        # create signal as a linear combination of eigenvectors and random coefficients
        s = eigvecs @ coeffs[:, np.newaxis]
        signals.append(s)
    return np.hstack(signals)


def gs2(L: sparse.csr_matrix, size: int = 50, delta: float = 1e-5) -> np.ndarray:
    """
    Generates a graph signal of type GS2 of shape (n, size).
    Description from the paper:
    "GS2: The true signals are generated from multivariate Gaussian distribution $N(0, (L + \delta I)^-1)$,
    where $\delta = 10^-5$. Because the power of the generated graph signals is inconsistent,
    we normalize the signals using $x' = (x-mean(x)) / std(x)$."
    :param L: graph Laplacian matrix of shape (n, n)
    :param size: number of signals to generate
    :param delta: small value to multiply with identity matrix to ensure inverse of Laplacian exists
    :return: signal matrix of shape (n, size) where each column corresponds to a signal vector
    """
    n = L.shape[0]
    # cov = inverse Laplacian (+ small positive delta term to ensure invertibility)
    cov = np.linalg.inv(L + delta * np.eye(n))
    # generate signal from multivar. normal dist.
    s = np.random.multivariate_normal(np.zeros(n), cov, size).T
    # normalize the signal
    normal_s = (s - np.mean(s, axis=0, keepdims=True)) / np.std(s, axis=0, keepdims=True)
    return normal_s


def gauss_noise(mean=0, std=0.1, size=50) -> np.ndarray:
    """Returns a vector of Gaussian noise of shape (size,)."""
    return np.random.normal(mean, std, size)


def propagate_signal(G: nx.Graph, p: float, start=None):
    """
    Simulates the propagation of a value from one or more nodes to their neighbors.
    The initial signal at the starting node(s) is propagated with probability p to the neighboring nodes.
    :param G: graph
    :param start: starting node(s) where the signal is set to 1 (discrete centrality vector)
    :param p: propagation probability
    :return: continuous centrality vector
    """
    n = len(G)
    s = np.zeros(n)
    if start is None:
        start = np.random.choice(n)
    s[start] = 1.
    H = nx.bfs_successors(G, start)
    for node, neighbors in H:
        for neighbor in neighbors:
            s[neighbor] += p * s[node]
    return s


def mask_signal(s: np.ndarray, num_masked: Optional[int] = None, index: Optional = None, val=0):
    """
    Masks a signal vector either by setting random indices to a specified value (default: 0)
    or by setting a given list of indices to that value.
    :param s: signal vector
    :param num_masked: number of entries to mask
    :param index: (optional) indices of entries that shall be masked
    :param val: masking value
    :return: masked signal
    """
    masked_s = s.copy()
    if index is None:
        rng = np.random.default_rng()
        rand_idx = rng.choice(len(s), num_masked, replace=False)
        masked_s[rand_idx] = val
    else:
        masked_s[index] = val
    return masked_s


def mask_nonadjacent_nodes(G: nx.Graph, s: np.ndarray, num_masked: int, val=0):
    """
    Masks only non-adjacent nodes by masking num_masked random nodes of the independent set of the graph.
    :param G: graph
    :param s: signal vector
    :param num_masked: number of masked nodes
    :param val: masking value
    :return: masked signal
    """
    masked_s = s.copy()
    indep_set = list(nx.algorithms.approximation.maximum_independent_set(G))
    p = np.full(s.shape[0], 0.01)
    p[indep_set] = 1.
    p = p / p.sum()
    rng = np.random.default_rng()
    idx = rng.choice(len(s), num_masked, replace=False, p=p)
    masked_s[idx] = val
    return masked_s
