import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn import metrics

import src.utils.plotting as plt_util
from src.gershgorin.bs_gda import bs_gda
from src.graph.graph import Graph
from src.gsp import reconstruction
from src.utils import data_handler
from src.utils.yaml_reader import YamlReader


def main(config):
    IN_DIR = config["in"]["dir"]
    in_files = config["in"]["files"]

    adj_matrix = data_handler.load_sparse_matrix(f"{IN_DIR}/{in_files['adj_matrix']}")
    graph = Graph(adj_matrix)
    print("Graph loaded.")
    s = np.load(f"{IN_DIR}/{in_files['signal']}").flatten()
    print("Signal loaded.")
    print("Selecting sampling set...")
    sampling_set, _ = bs_gda(graph, **config["method"])
    print("Reconstructing signal...")
    s_rec = reconstruction.reconstruct_signal(graph.laplacian(), sampling_set, s[sampling_set])

    print("Detecting deep detractors...")
    y_true = is_deep_detractor(s)
    # invert score to stick to scikit-learn convention which uses y_score >= thresh instead of y_score <= thresh
    y_score = 1 / (s_rec + 1e-8)

    # compute predictions with given threshold (or simple threshold of y_score < 3)
    thresh = 1 / 3. if config["thresh"] is None else config["thresh"]
    y_pred = y_score >= thresh
    # consider sampling set as training data; ignore predictions for sampled nodes in evaluation
    not_sampled = np.ones(graph.num_nodes, dtype=bool)
    not_sampled[sampling_set] = False
    y_true_pruned = y_true[not_sampled]
    y_pred_pruned = y_pred[not_sampled]
    y_score_pruned = y_score[not_sampled]

    # evaluate model predictions
    precision_vals, recall_vals, thresholds = metrics.precision_recall_curve(y_true_pruned, y_score_pruned, pos_label=1)
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(
        y_true_pruned, y_pred_pruned, pos_label=1, average="binary"
    )
    stats = {
        "precision": precision,
        "recall": recall,
        "f1_score": fscore,
        "support": support
    }

    OUT_DIR = config["out"]["dir"]
    os.makedirs(OUT_DIR, exist_ok=True)
    out_files = config["out"]["files"]
    print("Saving results...")
    with open(f"{OUT_DIR}/{out_files['stats']}", "w") as file:
        file.write(json.dumps(stats, indent=4))
    plt_util.plot_precision_recall_curve(precision_vals, recall_vals,
                                         title="Precision-recall curve\nfor detection of deep detractors",
                                         filepath=f"{OUT_DIR}/{out_files['precision_recall_curve']}"
                                         )
    np.save(f"{OUT_DIR}/{out_files['thresh']}", thresholds)
    data_handler.save_df_as_csv(pd.DataFrame(y_pred_pruned, columns=["node_id"]),
                                f"{OUT_DIR}/{out_files['pred']}")
    # map deep detractor node ids to client ids
    client_ids = pd.Series(data_handler.load_csv_to_list(f"{IN_DIR}/{in_files['client_ids']}"))
    data_handler.save_df_as_csv(pd.DataFrame(client_ids[y_pred_pruned], columns=["client_ids"]),
                                f"{OUT_DIR}/{out_files['pred_client_ids']}")
    np.save(f"{OUT_DIR}/{out_files['signal_rec']}", s_rec[not_sampled])


def is_deep_detractor(s: np.ndarray) -> np.ndarray:
    """
    Checks which customers are deep detractors.
    A deep detractor is defined as someone who gave a recommendation score <= 2.
    :param s: NPS signal vector
    :return: boolean array
    """
    return s <= 2


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
