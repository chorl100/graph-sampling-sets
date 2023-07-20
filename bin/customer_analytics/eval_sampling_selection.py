"""
Experiment regarding reconstruction analysis.
Measures reconstruction error, eigenvalue lower bound and signal smoothness
for increasing sampling budget.
"""
import argparse
import os

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

import src.utils.plotting as plt_util
from src.graph import sample_evaluation
from src.metrics import metrics
from src.utils import colors, data_handler
from src.utils.yaml_reader import YamlReader


def main(config):
    IN_DIR = config["in"]["dir"]
    in_files = config["in"]["files"]
    # load graph
    print("Loading graph...")
    adj_matrix = sparse.load_npz(f"{IN_DIR}/{in_files['adj_matrix']}")
    graph = nx.from_scipy_sparse_matrix(adj_matrix)

    print("Loading sampling sets...")
    actual_set = data_handler.load_csv_to_list(f"{IN_DIR}/{in_files['actual_set']}")
    pred_set = data_handler.load_csv_to_list(f"{IN_DIR}/{in_files['pred_set']}")

    # analyze neighborhood
    print("Analyzing neighborhood...")
    max_dist = sample_evaluation.neighborhood(graph, pred_set, actual_set, config["max_hops"])

    print("Loading client ids...")
    client_ids = data_handler.load_csv_to_series(f"{IN_DIR}/{in_files['client_ids']}")
    actual_client_ids = data_handler.load_csv_to_series(f"{IN_DIR}/{in_files['actual_client_ids']}")
    pred_client_ids = data_handler.load_csv_to_series(f"{IN_DIR}/{in_files['pred_client_ids']}")
    # compute statistics
    print("Calculating statistics...")
    stats = calculate_statistics(client_ids, pred_client_ids, actual_client_ids)
    df = pd.DataFrame.from_dict(stats, orient="index", columns=["value"]).rename_axis("metric").reset_index()

    OUT_DIR = config["out"]["dir"]
    os.makedirs(OUT_DIR, exist_ok=True)
    out_files = config["out"]["files"]
    # save sample neighborhood measure
    with open(f"{OUT_DIR}/{out_files['neighborhood']}", "w") as file:
        file.write(f"Maximum distance between actual and predicted nodes is {max_dist}.")
    # save stats
    df.to_csv(f"{OUT_DIR}/{out_files['stats']}", index=False)
    # draw venn diagram of found client ids
    plt_util.plot_venn2_diagram([set(actual_client_ids.tolist()), set(pred_client_ids.tolist())],
                                ["Actual", "Predicted"],
                                set_colors=[colors.FRESH_ORANGE, colors.SPRING_GREEN], **config["plot"],
                                filepath=f"{OUT_DIR}/{out_files['venn_diagram']}")


def calculate_statistics(all: pd.Series, pred: pd.Series, actual: pd.Series):
    # how many nodes were sampled correctly
    tp = np.count_nonzero(actual.isin(pred))
    # how many nodes were not sampled
    tn = np.count_nonzero(~all.isin(actual) & ~all.isin(pred))
    # how many nodes shouldn't have been sampled
    fp = np.count_nonzero(~actual.isin(pred))
    # how many nodes should have been sampled
    fn = np.count_nonzero(~all.isin(actual) & all.isin(pred))

    prec = metrics.precision(tp, fp)
    negprec = metrics.npv(tn, fn)
    rec = metrics.recall(tp, fn)
    spec = metrics.specificity(tn, fp)
    acc = metrics.accuracy(tp, tn, fp, fn)
    f1 = metrics.f1_score(tp, fp, fn)
    return {
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "Precision": prec, "negprec": negprec, "Recall": rec,
        "Specificity": spec, "Accuracy": acc, "F1 score": f1
    }


if __name__ == '__main__':
    # Read the arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--config", required=True,
                           help="YAML config file to read the method configuration.")
    args = vars(argParser.parse_args())
    configuration = YamlReader(args["config"]).config
    main(configuration)
