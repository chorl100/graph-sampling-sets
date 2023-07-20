"""
Experiments regarding runtime analysis.
Measures runtime with increasing graph size.
"""
import os
import time

import numpy as np

import src.utils.plotting as plt
from src.gershgorin.bs_gda import bs_gda
from src.graph import graph_builder
from src.utils.yaml_reader import YamlReader


def main():
    config_path = "config/runtime_analysis.yml"
    config = YamlReader(config_path).config
    # cast eps to float
    config["method"]["eps"] = float(config["method"]["eps"])
    graph_params = config["graph"]
    alg_params = config["method"]
    graph_sizes = graph_params["sizes"]
    class_name = graph_params["class"]

    build_func = getattr(graph_builder, class_name)
    times = run_runtime_analysis(graph_sizes, build_func, *alg_params.values())
    os.makedirs("./out", exist_ok=True)
    plt.plot_runtime(graph_sizes, times, name=class_name, out="./out")


def run_runtime_analysis(sizes: list, build_func, k: float, mu: float, eps: float, runs: int, parallel: bool):
    runtimes = np.zeros((runs, len(sizes)))
    print("Starting runtime analysis.")
    for i in range(runs):
        print("Run", i+1)
        for j, n in enumerate(sizes):
            # if k is a percentage, multiply with number of nodes
            sampling_budget = int(n*k) if k < 1 else k
            print(f"|\tn: {n}")
            G = build_func(n)
            tic = time.perf_counter()
            sampling_set, thres = bs_gda(G, sampling_budget, mu, eps, parallel=parallel)
            toc = time.perf_counter()
            print(f"|\tsampling: {len(sampling_set)} nodes, threshold: {thres}\n|")
            runtimes[i, j] = toc - tic
    return runtimes.mean(axis=0)


if __name__ == '__main__':
    main()
