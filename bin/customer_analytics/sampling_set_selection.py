import argparse
import json
import os
import time

import pandas as pd
from scipy import sparse

from src.gershgorin.bs_gda import bs_gda
from src.graph.graph import Graph
from src.utils import data_handler
from src.utils.yaml_reader import YamlReader


def main(config):
    IN_DIR = config["in"]["dir"]
    in_files = config["in"]["files"]

    print("Loading adjacency matrix...")
    adj_matrix = sparse.load_npz(f"{IN_DIR}/{in_files['adj_matrix']}")
    print("Building graph (adjacency list)...")
    graph = Graph(adj_matrix)

    print("Calculating sampling set...")
    start = time.perf_counter()
    sampling_set, thres = bs_gda(graph, **config["method"])
    runtime = time.perf_counter() - start
    sample_size = len(sampling_set)
    print(f"This took {runtime:.3f} s")
    print(f"Sample size: {sample_size} (budget was {config['method']['k']})")

    results = {
        "n_nodes": graph.num_nodes,
        **config["method"],
        "thres": thres,
        "sample_size": sample_size,
        "runtime": runtime
    }

    OUT_DIR = config["out"]["dir"]
    os.makedirs(OUT_DIR, exist_ok=True)
    out_files = config["out"]["files"]
    # store runtime and method config in JSON file
    with open(f"{OUT_DIR}/{out_files['results']}", "w") as file:
        file.write(json.dumps(results, indent=4))
    print("Saving sets...")
    # store sampling set on disk
    data_handler.save_df_as_csv(pd.DataFrame(sampling_set, columns=["node_id"]), f"{OUT_DIR}/{out_files['sampling_set']}")
    # map sampled node ids to client ids
    client_ids = pd.Series(data_handler.load_csv_to_list(f"{IN_DIR}/{in_files['client_ids']}"))
    data_handler.save_df_as_csv(client_ids[sampling_set], f"{OUT_DIR}/{out_files['client_ids_sampling_set']}")


if __name__ == '__main__':
    # Read the arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--config", required=True,
                           help="YAML config file to read the method configuration.")
    args = vars(argParser.parse_args())
    configuration = YamlReader(args["config"]).config
    # cast eps to float
    configuration["method"]["eps"] = float(configuration["method"]["eps"])
    main(configuration)
