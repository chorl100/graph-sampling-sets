import argparse
import os

import src.utils.plotting as plt_util
from src.utils import data_handler, layout
from src.utils.yaml_reader import YamlReader


def draw_sampling_sets_on_graph(config):
    IN_DIR = config["in"]["dir"]
    in_files = config["in"]["files"]
    # load sampling sets
    print("Loading sampling sets...")
    preselection = data_handler.load_csv_to_list(f"{IN_DIR}/{in_files['preselection']}")
    extended_sampling_set = data_handler.load_csv_to_list(f"{IN_DIR}/{in_files['extended_set']}")
    sampling_set = data_handler.load_csv_to_list(f"{IN_DIR}/{in_files['sampling_set']}")

    # get graph layout
    print("Loading graph layout...")
    adr_zips = data_handler.load_csv_to_list(f"{IN_DIR}/{in_files['zip_codes']}", dtype=str)
    pos = layout.geo_layout(adr_zips)
    # draw samples on graph
    print("Drawing networks...")
    OUT_DIR = config["out"]["dir"]
    os.makedirs(OUT_DIR, exist_ok=True)
    out_files = config["out"]["files"]
    # preselected vs. extended (disjoint)
    plt_util.scatter_graph(pos, preselection, extended_sampling_set, ["not sampled", "preselected", "extended"],
                           hide_not_sampled=False, with_intersect=False, title="Preselected vs. extended sampling set",
                           **config["plot"],
                           filepath=f"{OUT_DIR}/{out_files['pre_vs_ext']}")
    # preselected vs. sampled (without preselection)
    plt_util.scatter_graph(pos, preselection, sampling_set, ["preselected", "sampled", "intersected"],
                           hide_not_sampled=True, with_intersect=True, title="Preselected vs. sampled sampling set",
                           **config["plot"],
                           filepath=f"{OUT_DIR}/{out_files['pre_vs_sampled']}")
    # extended vs. sampled
    plt_util.scatter_graph(pos, sampling_set, extended_sampling_set, ["extended", "sampled", "intersected"],
                           hide_not_sampled=True, with_intersect=True, title="Extended vs. sampled sampling set",
                           **config["plot"],
                           filepath=f"{OUT_DIR}/{out_files['ext_vs_sampled']}")


if __name__ == '__main__':
    # Read the arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--config", required=True,
                           help="YAML config file to read the method configuration.")
    args = vars(argParser.parse_args())
    configuration = YamlReader(args["config"]).config
    draw_sampling_sets_on_graph(configuration)
