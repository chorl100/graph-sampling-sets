import multiprocessing
import typing

import mpire
import numpy as np

from src.gershgorin.bucket_queue import BucketQueue
from src.gershgorin.disc_alignment import estimate_coverage_subsets
from src.gershgorin.solving_set_covering import select_max_coverage_set_prio, \
    select_max_coverage_set, _greedy_set_cover_parallel_step
from src.graph.graph import Graph


def greedy_sampling(graph: Graph, thres: float, k: int, mu: float, p_hops: int,
                    shuffle: bool = False, parallel: bool = False):
    """
    Disc Alignment via Greedy Set Cover. (Algorithm 2)
    Selects a sampling set of at most size k that represents the graph structure best.
    :param graph: graph
    :param thres: threshold
    :param k: sampling budget
    :param mu: parameter for graph Laplacian based GSP reconstruction
    :param p_hops: number of hops
    :param shuffle: whether to shuffle the nodes
    :param parallel: whether to apply parallelization where possible
    :return: sampling set, validity flag
    """
    nodes = range(graph.num_nodes)
    if shuffle:
        nodes = np.random.permutation(nodes)
    coverage_subsets = estimate_coverage_subsets(nodes, graph, thres, mu, p_hops, parallel)
    sampling_set, vf = greedy_set_cover(coverage_subsets, nodes, k, parallel=False)
    return sampling_set, vf


def greedy_sampling_extension(graph: Graph, preselection: list, thres: float, k: int, mu: float, p_hops: int,
                              shuffle: bool = False, parallel: bool = False):
    """
    Disc Alignment via Greedy Set Cover. (Algorithm 2)
    Extends the preselection.
    :param graph: graph
    :param preselection: selection of nodes that are definitely in the sampling set
    :param thres: threshold
    :param k: sampling budget
    :param mu: parameter for graph Laplacian based GSP reconstruction
    :param p_hops: number of hops
    :param shuffle: whether to shuffle the nodes
    :param parallel: whether to apply parallelization where possible
    :return: sampling set, validity flag
    """
    nodes = range(graph.num_nodes)
    if shuffle:
        nodes = np.random.permutation(nodes)
    coverage_subsets = estimate_coverage_subsets(nodes, graph, thres, mu, p_hops, parallel)
    sampling_set, vf = greedy_set_cover_extension(coverage_subsets, preselection, nodes, k)
    return sampling_set, vf


def greedy_set_cover(coverage_subsets: typing.List[typing.Set[int]], nodes, k: int, parallel=False) \
        -> typing.Tuple[list, bool]:
    """
    Solves the set cover problem with a greedy approach.
    :param coverage_subsets: list of coverage subsets for each node
    :param nodes: list of nodes
    :param k: sample size
    :param parallel: whether to use parallelization (always False because it's too slow)
    :return:
    """
    if parallel:
        return greedy_set_cover_parallel(coverage_subsets, nodes, k, mpire.cpu_count())
    else:
        return greedy_set_cover_(coverage_subsets, nodes, k)


def greedy_set_cover_(coverage_subsets: typing.List[typing.Set[int]], nodes, k: int) -> typing.Tuple[list, bool]:
    """Greedy set cover algorithm.
    :param coverage_subsets: list of node subsets
    :param nodes: graph nodes
    :param k: sampling budget
    :return: sampling set, validity flag
    """
    N = len(nodes)
    # set of uncovered nodes
    uncovered = set(range(N))
    # covered sets
    covered = np.zeros(N, dtype=bool)
    # selected sets
    selected = np.zeros(N, dtype=bool)
    sampling_set = list()

    num_selected = 0
    while len(uncovered) > 0 and num_selected < k:
        max_coverage_set, i = select_max_coverage_set(coverage_subsets, uncovered, covered, selected)
        # remove covered nodes from uncovered nodes
        uncovered -= max_coverage_set
        # take node i into the sampling set
        sampling_set.append(i)
        selected[i] = 1
        num_selected += 1

    # init validity flag
    vf = True
    # Does each node belong to at least one coverage set?
    if len(uncovered) > 0:
        vf = False

    return sampling_set, vf


def greedy_set_cover_parallel(coverage_subsets: typing.List[typing.Set[int]], nodes, k: int, n_jobs: int) \
        -> typing.Tuple[list, bool]:
    """Greedy set cover algorithm (parallel variant).
    :param coverage_subsets: list of node subsets
    :param nodes: graph nodes
    :param k: sampling budget
    :param n_jobs: number of parallel processes to start
    :return: sampling set, validity flag
    """
    N = len(nodes)
    # set of uncovered nodes
    uncovered = multiprocessing.Array('b', N, lock=False)
    sampling_set = list()
    # covered sets
    covered = multiprocessing.Array('b', N, lock=False)
    # selected sets
    selected = multiprocessing.Array('b', N, lock=False)
    for i in range(N):
        uncovered[i] = 1
        covered[i] = 0
        selected[i] = 0

    batches = np.array_split(coverage_subsets, n_jobs)
    set_idx = list(range(N))
    batched_set_idx = np.array_split(set_idx, n_jobs)
    with mpire.WorkerPool(n_jobs, shared_objects=(uncovered, covered, selected), keep_alive=True) as pool:
        num_selected = 0
        while np.any(uncovered) and num_selected < k:
            candidate_sets = list(
                pool.map_unordered(
                    _greedy_set_cover_parallel_step,
                    zip(batched_set_idx, batches), iterable_len=len(batches), n_splits=n_jobs
                ))
            max_coverage_set, max_idx = max(candidate_sets, key=lambda x: len(x[0]))
            # remove covered nodes from uncovered nodes
            for node in max_coverage_set:
                uncovered[node] = 0
            # take node i into the sampling set
            sampling_set.append(max_idx)
            selected[max_idx] = 1
            num_selected += 1

    # init validity flag
    vf = True
    # Does each node belong to at least one coverage set?
    if np.any(uncovered):
        vf = False

    return sampling_set, vf


def greedy_set_cover_priority_queue(coverage_subsets: typing.List[typing.Set[int]], nodes, k: int) \
        -> typing.Tuple[list, bool]:
    """Greedy set cover algorithm using a priority queue.
    :param coverage_subsets: list of node subsets
    :param nodes: graph nodes
    :param k: sampling budget
    :return: sampling set, validity flag
    """
    N = len(nodes)
    # set of uncovered nodes
    uncovered = np.ones(N, dtype=bool)
    # selected nodes
    sampling_set = list()
    # queue
    prio_queue = BucketQueue(coverage_subsets)
    updates = {i: set() for i in range(prio_queue.len)}

    num_selected = 0
    while np.any(uncovered) and num_selected < k:
        max_coverage_set, i = select_max_coverage_set_prio(prio_queue, coverage_subsets, uncovered, updates)
        # take node i into the sampling set
        sampling_set.append(i)
        num_selected += 1

    # init validity flag
    vf = True
    # Does each node belong to at least one coverage set?
    if np.any(uncovered):
        vf = False

    return sampling_set, vf


def greedy_set_cover_extension(coverage_subsets: typing.List[typing.Set[int]], preselection: list, nodes, k: int) \
        -> typing.Tuple[list, bool]:
    """Greedy set cover algorithm.
    :param coverage_subsets: list of node subsets
    :param preselection: selection of nodes that are definitely in the sampling set
    :param nodes: graph nodes
    :param k: sampling budget
    :return: sampling set, validity flag
    """
    N = len(nodes)
    # set of uncovered nodes
    uncovered = set(range(N))
    for node in preselection:
        uncovered -= coverage_subsets[node]
    # covered sets
    covered = np.zeros(N, dtype=bool)
    covered[preselection] = 1
    # selected sets
    selected = np.zeros(N, dtype=bool)
    selected[preselection] = 1
    sampling_set = preselection.copy()

    num_selected = len(sampling_set)
    while len(uncovered) > 0 and num_selected < k:
        max_coverage_set, i = select_max_coverage_set(coverage_subsets, uncovered, covered, selected)
        # remove covered nodes from uncovered nodes
        uncovered -= max_coverage_set
        # take node i into the sampling set
        sampling_set.append(i)
        selected[i] = 1
        num_selected += 1

    # init validity flag
    vf = True
    # Does each node belong to at least one coverage set?
    if len(uncovered) > 0:
        vf = False

    return sampling_set, vf
