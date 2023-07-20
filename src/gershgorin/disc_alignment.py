import itertools
import queue
from functools import partial

import mpire
import numpy as np
from numba import njit

from src.graph.graph import Graph


def estimate_coverage_subsets(nodes, graph: Graph, thres: float, mu: float, p_hops: int, parallel=False):
    """
    Estimating Coverage Subset. (Algorithm 1)
    Parallel execution is recommended for large graphs (more than 5000 nodes).
    :param nodes: list of nodes
    :param graph: graph
    :param thres: threshold on smallest eigenvalue of condition matrix
    :param mu: regularization strength
    :param p_hops: number of hops to make from a node
    :param parallel: whether to use parallelization
    :return:
    """
    if parallel:
        # split up the nodes into batches
        batches = np.array_split(nodes, mpire.cpu_count())
        # start as many processes as cores are available
        with mpire.WorkerPool(mpire.cpu_count(), shared_objects=graph, keep_alive=True) as pool:
            return list(itertools.chain.from_iterable(
                pool.map(
                    partial(estimate_coverage_subset_batched_, thres=thres, mu=mu, p_hops=p_hops),
                    batches, n_splits=mpire.cpu_count()
                )
            ))
    else:
        coverage_subsets = list()
        for i in nodes:
            coverage_subset = estimate_coverage_subset_(i, graph, thres, mu, p_hops)
            coverage_subsets.append(coverage_subset)
        return coverage_subsets


def estimate_coverage_subset_batched_(graph, batch, thres, mu, p_hops):
    coverage_subsets = list()
    for node in batch:
        coverage_subsets.append(estimate_coverage_subset_(node, graph, thres, mu, p_hops))
    return coverage_subsets


def estimate_coverage_subset_(i: int, graph: Graph, thres: float, mu: float, p_hops: int) -> set:
    """Estimating Coverage Subset. (Algorithm 1)
    :param i: candidate sampling node
    :param graph: graph
    :param thres: threshold for lower bound
    :param mu: parameter for graph Laplacian based gsp reconstruction
    :param p_hops: number of hops
    :return: coverage subset
    """
    n = graph.num_nodes
    # initial disc radii
    s = np.ones(n)
    # candidate sampling vector
    a = np.zeros(n, dtype=bool)
    a[i] = 1
    # hop numbers
    h = np.zeros(n, dtype=np.int8)
    coverage_subset = set()
    # visited nodes
    visited = np.zeros(n, dtype=bool)
    q = queue.Queue()
    q.put(i)
    visited[i] = 1
    while not q.empty():
        k = q.get()
        s[k] = expand_radius(graph, s, k, a, mu, thres)
        if s[k] >= 1 and h[k] <= p_hops:
            coverage_subset.add(k)
            for t in graph.neighbors[k]:
                if not visited[t]:
                    q.put(t)
                    visited[t] = 1
                    h[t] = h[k] + 1
    return coverage_subset


def expand_radius(graph: Graph, s: np.array, i: int, a: np.array, mu: float, thres: float):
    """
    Computes the expansion factor of the node i.
    See also Equation 25.
    """
    return expand_radius_numba(s, a[i], graph.deg_w[i],
                               graph.neighbors[i], graph.neighbors_w[i],
                               mu, thres)


@njit
def expand_radius_numba(s, a, deg, neighbors, neighbors_weights, mu, thres) -> float:
    """Efficient implementation of the radius expansion method.
    :param s: signal vector
    :param a: sampling vector
    :param deg: degree of one node
    :param neighbors: neighbors of a node
    :param neighbors_weights: edge weights to neighbors
    :param mu: parameter for graph Laplacian based gsp reconstruction
    :param thres: threshold
    :return: scaled radius of a node
    """
    # use s where s >= 1, else set it to 1
    s_tmp = np.ones_like(s)
    idx = s > 1
    s_tmp[idx] = s[idx]
    numerator = a + mu * deg - thres
    disc_radii = s_tmp[neighbors]
    denominator = mu * np.sum(neighbors_weights / disc_radii)
    return numerator / denominator
