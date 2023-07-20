import argparse
import json
import os
import time

import pandas as pd
from scipy import sparse

from bin.customer_analytics.data_preprocessing import choose_random_sample
from src.gershgorin.bs_gda import bs_gda_extension
from src.graph.graph import Graph
from src.utils import data_handler
from src.utils.yaml_reader import YamlReader


def main(config):
    # input directory
    IN_DIR = config["in"]["dir"]
    # names of input files
    in_files = config["in"]["files"]

    print("Loading adjacency matrix...")
    adj_matrix = data_handler.load_sparse_matrix(f"{IN_DIR}/{in_files['adj_matrix']}")
    print("Building graph (adjacency list)...")
    # wrap graph with adjacency list for faster neighborhood queries
    graph = Graph(adj_matrix)

    if config["preselection"]["rand"]:
        preselection = choose_random_sample(graph.num_nodes, config["preselection"]["size"])
    else:
        preselection = config["preselection"]["set"]

    # specify max. number of nodes to sample
    sampling_budget = config["method"]["k"]
    assert sampling_budget > len(preselection)

    # extend preselection
    print("Extending preselection...")
    start = time.perf_counter()
    sampling_set_extended, thres = bs_gda_extension(graph, preselection, **config["method"])
    runtime = time.perf_counter() - start
    sample_size = len(sampling_set_extended)
    print(f"Graph sampling set extension took {runtime:.3f} s")
    print("Total number of sampled nodes:", sample_size)

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
    # store preselection on disk
    data_handler.save_df_as_csv(pd.DataFrame(preselection, columns=["node_id"]),
                                f"{OUT_DIR}/{out_files['preselection']}")
    # store extended sampling set on disk
    data_handler.save_df_as_csv(pd.DataFrame(sampling_set_extended, columns=["node_id"]),
                                f"{OUT_DIR}/{out_files['extended_set']}")
    # map sampled node ids to client ids
    client_ids = pd.Series(data_handler.load_csv_to_list(f"{IN_DIR}/{in_files['client_ids']}"))
    data_handler.save_df_as_csv(client_ids[sampling_set_extended],
                                f"{OUT_DIR}/{out_files['client_ids_extended_set']}")
    # which customers should be sampled in addition to the preselection?
    data_handler.save_df_as_csv(client_ids[set(sampling_set_extended) - set(preselection)],
                                f"{OUT_DIR}/{out_files['client_ids_extended_set_wo_preselect']}")


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
