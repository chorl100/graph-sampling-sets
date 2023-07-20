import math
import time

from src.graph import graph_builder
from src.gershgorin.bs_gda import bs_gda


def main():
    n = 2000
    run_runtime_analysis_single(n, 0.1, 0.01, 1e-5, 0)


def run_runtime_analysis_single(n: int, p: float, mu: float, eps: float, seed: int):
    G = graph_builder.sensor(n, seed)
    tic = time.perf_counter()
    sampling_set, thres = bs_gda(G, math.floor(n*p), mu, eps, parallel=True)
    toc = time.perf_counter()
    print(f"sampling: {len(sampling_set)} nodes, threshold: {thres}\n")
    print(toc - tic)


if __name__ == '__main__':
    main()
