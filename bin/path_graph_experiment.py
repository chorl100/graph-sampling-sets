"""
Path graph experiment from the paper
"""
import math
import time

import numpy as np
from scipy import sparse
import scipy.sparse.linalg

import src.utils.plotting as plt
from src.gershgorin.bs_gda import bs_gda
from src.graph.graph import Graph


def main():
    # path graph generation
    num_nodes = 1000
    # build path graph adjacency matrix
    A = np.zeros((num_nodes, num_nodes))
    v1 = np.ones(num_nodes - 1)
    A1 = np.diag(v1, 1)
    A += A1 + A1.T
    # construct adjacency list for BS_GDA function from adjacency matrix
    G = Graph(A)

    sampling_budget = math.floor(num_nodes * 0.1)
    mu = 0.01
    epsilon = 1e-5
    p_hops = 12

    # run BS-GDA graph sampling
    tic = time.perf_counter()
    sampling_set, thres = bs_gda(G, sampling_budget, mu, epsilon, p_hops, parallel=False)
    toc = time.perf_counter()
    print(f"BS-GDA running time: {toc - tic:.3f} s\n")
    print(f"Sampling: {len(sampling_set)} nodes, threshold: {thres}\n")

    # plot sampling nodes
    sampled_nodes = np.zeros(num_nodes)
    sampled_nodes[list(sampling_set)] = 1
    plt.plot_stems(range(num_nodes), sampled_nodes)

    # compute the smallest eigenvalue via eigendecomposition
    L = G.laplacian()
    a = np.zeros(num_nodes)
    a[list(sampling_set)] = 1
    B = np.diag(a) + mu * L

    B = sparse.csr_matrix(B)
    se, _ = sparse.linalg.eigsh(B, k=1, which='SM')
    print(f"smallest eigenvalue via eigen-decomposition: {se[0]}\n")


if __name__ == '__main__':
    main()
