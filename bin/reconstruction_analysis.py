"""
Experiments regarding reconstruction analysis.
Measures signal smoothness with increasing sampling budget.
"""
import math
import os

import numpy as np
from scipy.spatial import distance

import src.utils.plotting as util_plt
from src.gershgorin.bs_gda import bs_gda
from src.graph import graph_builder, graph_tools
from src.graph.graph import Graph
from src.gsp import signal
from src.gsp.reconstruction import reconstruct_signal, mse
from src.gsp.signal import gauss_noise
from src.utils.yaml_reader import YamlReader

OUT_PATH = "out/paper/reconstruction"


def main():
    config_path = "config/reconstruction_analysis.yml"
    config = YamlReader(config_path).config
    # cast eps to float
    config["method"]["eps"] = float(config["method"]["eps"])
    graph_params = config["graph"]
    alg_params = config["method"]
    n_nodes = graph_params["size"]
    class_name = graph_params["class"]

    build_func = getattr(graph_builder, class_name)
    signal_func = getattr(signal, graph_params["signal"])
    os.makedirs(OUT_PATH, exist_ok=True)
    errors, lower_bounds = run_reconstruction_analysis_paper(n_nodes, build_func, signal_func,
                                                             *alg_params.values(), config["seed"])
    util_plt.plot_reconstruction_error_paper(np.array(alg_params["sampling_budgets"]), errors,
                                             class_name, signal_func.__name__, out=OUT_PATH)
    util_plt.plot_eig_lower_bound_paper(alg_params["sampling_budgets"], lower_bounds,
                                        class_name, signal_func.__name__, out=OUT_PATH)


def run_reconstruction_analysis_paper(n_nodes, build_func, signal_func, sampling_budgets: list, mu: float,
                                      eps: float, p_hops: int, runs: int, seed: int):
    errors = np.zeros((runs, len(sampling_budgets)))
    thresholds = np.zeros_like(errors)
    print("Starting reconstruction analysis.")
    for run in range(runs):
        print("Run", run + 1)
        graph = build_graph(build_func.__name__, n_nodes, seed + run)
        L = graph.laplacian()
        # sample random signals
        rand_signals = signal_func(L, size=50)
        # generate random Gaussian noise
        gauss_noises = gauss_noise(size=50)
        for j, k in enumerate(sampling_budgets):
            # if budget is a percentage, multiply with number of nodes
            sampling_budget = math.floor(n_nodes * k) if k < 1 else k
            print(f"|\tK: {sampling_budget}")
            # compute the sampling set
            sampling_set, thres = bs_gda(graph, sampling_budget, mu, eps, p_hops, parallel=False)
            print(f"|\tsample: {len(sampling_set)} nodes, threshold: {thres}\n|")
            mses = []
            for s in rand_signals.T:
                for noise in gauss_noises:
                    # add Gaussian noise to the signal
                    s_noisy = s + noise
                    # reconstruct the original signal from the sampled signal
                    s_recon = reconstruct_signal(L, sampling_set, s_noisy[sampling_set], mu)
                    mses.append(mse(s_noisy, s_recon))
            # take the average MSE over the signals
            errors[run, j] = np.mean(mses)
            thresholds[run, j] = thres
    # take the average MSE over the runs
    return errors.mean(axis=0), thresholds.mean(axis=0)


def build_graph(graph_type: str, n_nodes: int, seed: int) -> Graph:
    if graph_type == "sensor":
        W = graph_builder.sensor(n_nodes, seed=seed).W  # sensor graph is already weighted
    elif graph_type == "community":
        graph = graph_builder.community(n_nodes, seed=seed)
        W = graph.W.copy().astype(float)
        idx = W.nonzero()
        dist = distance.cdist(graph.coords, graph.coords)
        weights = gaussian_kernel(dist, sigma=1.)
        W[idx] = weights[idx]
    elif graph_type == "minnesota":
        graph = graph_builder.minnesota()
        W = graph.W.copy().astype(float)
        idx = graph.W.nonzero()
        dist = distance.cdist(graph.coords, graph.coords)
        weights = gaussian_kernel(dist, sigma=0.1)
        W[idx] = weights[idx]
    elif graph_type == "barabasi_albert":
        adj = graph_builder.barabasi_albert(n_nodes, seed=seed).W
        W = adj.copy().astype(float)
        idx = adj.nonzero()
        rand_weights = np.random.rand(*adj.shape)
        W[idx] = rand_weights[idx]
        W = graph_tools.symmetrize_adj(W)
    else:
        raise ValueError(f"Unknown graph building function '{graph_type}'.")
    return Graph(W)


def gaussian_kernel(dist, sigma: float = 1.):
    return np.exp(-dist ** 2 / sigma ** 2)


if __name__ == '__main__':
    main()
