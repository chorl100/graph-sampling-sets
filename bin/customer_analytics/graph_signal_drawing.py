import argparse
import os

import numpy as np
from scipy import sparse

from src.gsp import laplace_utils
from src.utils import layout, data_handler
import src.utils.plotting as plt_util
from src.utils.yaml_reader import YamlReader


def draw_sampling_sets_on_graph(config):
    IN_DIR = config["in"]["dir"]
    in_files = config["in"]["files"]
    # load signal
    print("Loading signal...")
    s = np.load(f"{IN_DIR}/{in_files['signal']}").flatten()
    # get graph layout
    print("Loading graph layout...")
    adr_zips = data_handler.load_csv_to_list(f"{IN_DIR}/{in_files['zip_codes']}", dtype=str)
    pos = layout.geo_layout(adr_zips)
    print("Computing signal smoothness...")
    # compute Laplacian
    adj = sparse.load_npz(f"{IN_DIR}/{in_files['adj_matrix']}")
    lap = laplace_utils.adj_to_lap(adj)
    # compute normalized signal smoothness
    smoothness = laplace_utils.norm_lap_quad_form(lap, s)
    
    OUT_DIR = config["out"]["dir"]
    os.makedirs(OUT_DIR, exist_ok=True)
    out_files = config["out"]["files"]
    print("Drawing graph...")
    plt_util.scatter_graph_color_vec(
        pos, **config["plot"], node_color=s,
        title=f"Graph signal \nsmoothness={smoothness:.3f}",
        filepath=f"{OUT_DIR}/{out_files['graph_signal']}"
    )


if __name__ == '__main__':
    # Read the arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--config", required=True,
                           help="YAML config file to read the method configuration.")
    args = vars(argParser.parse_args())
    configuration = YamlReader(args["config"]).config
    draw_sampling_sets_on_graph(configuration)
