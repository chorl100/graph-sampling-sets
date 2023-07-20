import math

import numpy as np
from tqdm import tqdm

from src.gershgorin.bs_gda import bs_gda
from src.graph.graph import Graph
from src.gsp import reconstruction, laplace_utils, signal


def run_sampling_budget_experiment(graph, s, sampling_budgets, mu, eps, p_hops, parallel=False):
    """
    Analyses reconstruction quality of a signal from a sample for growing sampling budget.
    :param graph: graph
    :param s: signal vector
    :param sampling_budgets: list of sampling budgets
    :param mu: regularization strength of smoothness prior
    :param eps: precision of binary search
    :param p_hops: size of p-hop neighborhood
    :param parallel: whether to use parallelization
    :return: metrics for reconstruction quality
    """
    sampling_sets = list()
    eig_lower_bounds = list()
    reconstructions = list()
    mses = list()
    lap_quad_forms = list()

    for k in tqdm(sampling_budgets):
        sample, thres = bs_gda(graph, k, mu, eps, p_hops, parallel=parallel)
        s_reconst = reconstruction.reconstruct_signal(graph.laplacian(), sample, s[sample], mu)
        sampling_sets.append(sample)
        eig_lower_bounds.append(thres)
        reconstructions.append(s_reconst)
        mses.append(reconstruction.mse(s, s_reconst))
        lap_quad_forms.append(laplace_utils.norm_lap_quad_form(graph.laplacian(), s_reconst))
    return mses, eig_lower_bounds, lap_quad_forms, sampling_sets, reconstructions


def run_p_hops_experiment(graph: Graph, s, p_hops, k, mu, eps, parallel):
    """
    Analyses reconstruction quality of a signal from a sample for growing p-hop neighborhood.
    :param graph: graph
    :param s: signal vector
    :param p_hops: list of number of hops
    :param k: sampling budget
    :param mu: regularization strength of smoothness prior
    :param eps: precision of binary search
    :param parallel: whether to use parallelization
    :return: metrics for reconstruction quality
    """
    sampling_sets = list()
    eig_lower_bounds = list()
    reconstructions = list()
    mses = list()
    lap_quad_forms = list()

    for p in tqdm(p_hops):
        sample, thres = bs_gda(graph, k, mu, eps, p, parallel=parallel)
        s_reconst = reconstruction.reconstruct_signal(graph.laplacian(), sample, s[sample], mu)
        sampling_sets.append(sample)
        eig_lower_bounds.append(thres)
        reconstructions.append(s_reconst)
        mses.append(reconstruction.mse(s, s_reconst))
        lap_quad_forms.append(laplace_utils.norm_lap_quad_form(graph.laplacian(), s_reconst))
    return mses, eig_lower_bounds, lap_quad_forms, sampling_sets, reconstructions


def run_reconstruction_analysis_budget_avg(graph, signal_func, sampling_budgets: list, n_signals: int = 50,
                                           mu: float = 0.01,
                                           eps: float = 1e-5, p_hops: int = 12, parallel: bool = False, runs: int = 5,
                                           seed: int = 0):
    """
    Evaluates the signal reconstruction quality for growing sampling budget, averaged over n_signals^2 many signals.
    :param graph: graph
    :param signal_func: signal function (one from src.gsp.signal)
    :param sampling_budgets: list of sample sizes
    :param n_signals: number of signals and random noises to generate
    :return:
    """
    n_nodes = graph.num_nodes
    errors = np.zeros((runs, len(sampling_budgets)))
    thresholds = np.zeros_like(errors)
    print("Starting reconstruction analysis.")
    for run in range(runs):
        print("Run", run + 1)
        L = graph.laplacian()
        # sample random signals
        rand_signals = signal_func(L, size=n_signals)
        # generate random Gaussian noise
        gauss_noises = signal.gauss_noise(size=n_signals)
        for j, k in enumerate(sampling_budgets):
            # if budget is a percentage, multiply with number of nodes
            sampling_budget = math.floor(n_nodes * k) if k < 1 else k
            print(f"|\tK: {sampling_budget}")
            # compute the sampling set
            sampling_set, thres = bs_gda(graph, sampling_budget, mu, eps, p_hops, parallel)
            print(f"|\tsample: {len(sampling_set)} nodes, threshold: {thres}\n|")
            mses = []
            for s in rand_signals.T:
                for noise in gauss_noises:
                    # add Gaussian noise to the signal
                    s_noisy = s + noise
                    # reconstruct the original signal from the sampled signal
                    s_recon = reconstruction.reconstruct_signal(L, sampling_set, s_noisy[sampling_set], mu)
                    mses.append(reconstruction.mse(s_noisy, s_recon))
            # take the average MSE over the signals
            errors[run, j] = np.mean(mses)
            thresholds[run, j] = thres
    # take the average MSE over the runs
    return errors.mean(axis=0), thresholds.mean(axis=0)


def run_reconstruction_analysis_hops_avg(graph, signal_func, hops: list, k: int, n_signals: int = 50, mu: float = 0.01,
                                         eps: float = 1e-5, p_hops: int = 12, parallel: bool = False, runs: int = 5,
                                         seed: int = 0):
    n_nodes = graph.num_nodes
    errors = np.zeros((runs, len(hops)))
    thresholds = np.zeros_like(errors)
    print("Starting reconstruction analysis.")
    for run in range(runs):
        print("Run", run + 1)
        L = graph.laplacian()
        # sample random signals
        rand_signals = signal_func(L, size=n_signals)
        # generate random Gaussian noise
        gauss_noises = signal.gauss_noise(size=n_signals)
        for j, p in enumerate(hops):
            # if budget is a percentage, multiply with number of nodes
            sampling_budget = math.floor(n_nodes * k) if k < 1 else k
            print(f"|\tp_hops: {p}")
            # compute the sampling set
            sampling_set, thres = bs_gda(graph, k, mu, eps, p, parallel)
            print(f"|\tsample: {len(sampling_set)} nodes, threshold: {thres}\n|")
            mses = []
            for s in rand_signals.T:
                for noise in gauss_noises:
                    # add Gaussian noise to the signal
                    s_noisy = s + noise
                    # reconstruct the original signal from the sampled signal
                    s_recon = reconstruction.reconstruct_signal(L, sampling_set, s_noisy[sampling_set], mu)
                    mses.append(reconstruction.mse(s_noisy, s_recon))
            # take the average MSE over the signals
            errors[run, j] = np.mean(mses)
            thresholds[run, j] = thres
    # take the average MSE over the runs
    return errors.mean(axis=0), thresholds.mean(axis=0)
