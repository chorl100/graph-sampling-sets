import typing

from src.gershgorin.greedy_sampling import greedy_sampling, greedy_sampling_extension
from src.graph.graph import Graph


def bs_gda(graph: Graph, k: int, mu: float = 0.01, eps: float = 1e-5, p_hops: int = 6,
           shuffle: bool = False, parallel: bool = True) \
        -> typing.Tuple[list, float]:
    """Binary Search with Gershgorin Disc Alignment. (Algorithm 3)
    :param graph: graph
    :param k: sampling budget
    :param mu: tradeoff parameter to balance GLR against the l2-norm reconstruction error
    :param eps: numerical precision for binary search
    :param p_hops: number of hops
    :param shuffle: whether to shuffle the nodes
    :param parallel: whether to apply parallelization where possible
    :return: valid sampling set, maximum lower bound for the smallest eigenvalue
    """
    left = 0
    right = 1
    valid_sampling_set = list()
    flag = False

    while abs(right - left) > eps:
        threshold = (right + left) / 2.
        sampling_set, vf = greedy_sampling(graph, threshold, k, mu, p_hops, shuffle, parallel)

        if not vf:
            right = threshold
        else:
            left = threshold
            valid_sampling_set = sampling_set
            flag = True

        if right < left:
            raise ValueError('binary search error!')
    max_lower_bound = left

    if not flag:
        print("Warning: epsilon is set too large, sub-optimal lower bound is output.\n")

    return valid_sampling_set, max_lower_bound


def bs_gda_extension(graph: Graph, preselection: list, k: int, mu: float = 0.01, eps: float = 1e-5, p_hops: int = 12,
                     shuffle: bool = False, parallel: bool = True) \
        -> typing.Tuple[list, float]:
    """Binary Search with Gershgorin Disc Alignment. (Algorithm 3)
    :param graph: graph
    :param preselection: selection of nodes that are definitely in the sampling set
    :param k: sampling budget
    :param mu: tradeoff parameter to balance GLR against the l2-norm data fidelity term
    :param eps: numerical precision for binary search
    :param p_hops: number of hops
    :param shuffle: whether to shuffle the nodes
    :param parallel: whether to apply parallelization where possible
    :return: valid sampling set, maximum lower bound for the smallest eigenvalue
    """
    left = 0
    right = 1
    valid_sampling_set = list()
    flag = False

    while abs(right - left) > eps:
        threshold = (right + left) / 2.
        sampling_set, vf = greedy_sampling_extension(graph, preselection, threshold, k, mu, p_hops, shuffle, parallel)

        if not vf:
            right = threshold
        else:
            left = threshold
            valid_sampling_set = sampling_set
            flag = True

        if right < left:
            raise ValueError('binary search error!')
    max_lower_bound = left

    if not flag:
        print("Warning: epsilon is set too large, sub-optimal lower bound is output.\n")

    return valid_sampling_set, max_lower_bound
