import typing

import numpy as np

from src.gershgorin.bucket_queue import BucketQueue


def select_max_coverage_set(sets: list, uncovered: set, covered, selected) -> typing.Tuple[set, int]:
    """Selects the set that covers the most uncovered nodes.
    :param sets: list of coverage subsets
    :param uncovered: set of uncovered nodes
    :param covered: array-like binary vector that indicates which lists and their nodes are covered
    :param selected: array-like binary vector that indicates which subsets have already been selected
    :return: maximum coverage set, index of maximum set in sets
    """
    max_coverage_set = set()
    max_coverage = 0
    max_idx = None
    for node, s in enumerate(sets):
        if not (covered[node] or selected[node]):
            coverage_set = s & uncovered
            num_covered = len(coverage_set)
            if num_covered > max_coverage:
                max_coverage_set = coverage_set
                max_coverage = num_covered
                max_idx = node
            elif num_covered == 0:
                covered[node] = 1
    return max_coverage_set, max_idx


def _greedy_set_cover_parallel_step(shared_objects, set_idx, sets):
    uncovered, covered, selected = shared_objects
    max_coverage_set = set()
    max_coverage = 0
    max_idx = None
    for node, s in zip(set_idx, sets):
        if not (covered[node] or selected[node]):
            s_indicator = np.zeros(len(uncovered), dtype=bool)
            s_indicator[list(s)] = 1
            covered_nodes = s_indicator & uncovered
            coverage_set = np.flatnonzero(covered_nodes)
            num_covered = len(coverage_set)
            if num_covered > max_coverage:
                max_coverage_set = coverage_set
                max_coverage = num_covered
                max_idx = node
            elif num_covered == 0:
                covered[node] = 1
    return max_coverage_set, max_idx


def select_max_coverage_set_prio(prio_queue: BucketQueue, sets: list, uncovered, updates) -> typing.Tuple[set, int]:
    """Selects the set that covers the most uncovered nodes using a Priority Queue.
    :param prio_queue: priority queue of subsets
    :param sets: list of coverage subsets
    :param uncovered: set of uncovered nodes
    :param updates: list of items to remove from sets
    :return: maximum coverage set, index of maximum set in sets
    """
    uncovered_sets = dict(enumerate(sets.copy()))
    # extract the set with the highest coverage
    _, max_idx = prio_queue.extract_max()
    max_coverage_set = sets[max_idx]
    del uncovered_sets[max_idx]

    # find occurrences of covered items in other sets S_j
    for item in max_coverage_set:
        if uncovered[item]:
            # mark as covered
            uncovered[item] = 0
            for j, set_ in uncovered_sets.items():
                if (item in set_) and (j != max_idx):
                    updates[j].add(item)

    # update other sets S_j
    for j in range(len(updates)):
        if updates[j]:
            # delete the set from the queue
            prio, _ = prio_queue.delete(j)
            # update the set
            uncovered_sets[j] -= updates[j]
            if len(uncovered_sets[j]) == 0:
                del uncovered_sets[j]
                updates[j] = set()
                continue
            # insert the set with a different prio
            prio_queue.insert(prio-len(updates[j]), j)
            updates[j] = set()
    return max_coverage_set, max_idx
