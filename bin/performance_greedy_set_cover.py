import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.gershgorin.greedy_sampling import greedy_set_cover, greedy_set_cover_priority_queue


def build_instance(n, p=0.2):
    # sampling budget is p*100 % of graph size
    k = np.floor(p * n)
    nodes = list(range(n))
    coverage_subsets = list()
    for i in range(n):
        # pick random set size between 2 and n/2
        size = np.random.randint(2, np.floor(n / 2))
        # select random nodes
        subset = set(np.random.randint(0, n, size))
        coverage_subsets.append(subset)
    return coverage_subsets, nodes, k


def main():
    ns = [500, 1000, 2500, 5000, 7500, 10000, 12500, 15000]
    t_seq = list()
    t_prio = list()
    t_parallel = list()

    for n in tqdm(ns):
        coverage_subsets, nodes, k = build_instance(n, p=0.2)
        t1 = time.perf_counter()
        greedy_set_cover(coverage_subsets, nodes, k, parallel=False)
        t_seq.append(time.perf_counter() - t1)
        t2 = time.perf_counter()
        greedy_set_cover_priority_queue(coverage_subsets, nodes, k)
        t_prio.append(time.perf_counter() - t2)
        t3 = time.perf_counter()
        greedy_set_cover(coverage_subsets, nodes, k, parallel=True)
        t_parallel.append(time.perf_counter() - t3)

    plot_runtime(
        ns, {"seq": t_seq, "prio": t_prio, "parallel": t_parallel}, filepath="greedy_set_cover_performance.pdf"
    )


def plot_runtime(sizes, times, figsize=(6, 6), filepath=None):
    plt.figure(figsize=figsize)
    plt.plot(sizes, times["seq"], label="Seq")
    plt.plot(sizes, times["prio"], label="Prio")
    plt.plot(sizes, times["parallel"], label="Parallel")
    plt.yscale("log")
    plt.legend()
    if filepath is not None:
        plt.savefig(filepath)


if __name__ == '__main__':
    main()
