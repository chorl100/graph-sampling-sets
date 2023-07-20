import argparse
import os

import numpy as np
from scipy import sparse

from src.db import nps
from src.gsp import signal, laplace_utils
from src.utils import data_handler
from src.utils.yaml_reader import YamlReader


def main(config):
    # generate signal
    IN_DIR = config["in"]["dir"]
    in_files = config['in']['files']
    signal_type = config["signal"]["type"]
    print("Generating signal...")
    if signal_type == "nps":
        client_ids = data_handler.load_csv_to_list(f"{IN_DIR}/{in_files['client_ids']}")
        s = nps.nps_signal(**config["db"], client_ids=client_ids)
    else:
        adj_matrix = sparse.load_npz(f"{IN_DIR}/{in_files['adj_matrix']}")
        L = laplace_utils.adj_to_lap(adj_matrix).asfptype()
        if signal_type == "gs1":
            s = np.mean(signal.gs1(L, 50) + signal.gauss_noise(size=50), axis=1)
        elif signal_type == "gs2":
            s = np.mean(signal.gs2(L, 50) + signal.gauss_noise(size=50), axis=1)
        elif signal_type == "bandlimited":
            s = np.mean(
                signal.bandlimited_signal(L, **config["signal"]["bandlimited"], size=50)
                + signal.gauss_noise(size=50), axis=1
            )
        else:
            raise ValueError("Unknown signal type")

    if config["signal"]["add_noise"]:
        noise = signal.gauss_noise(**config["signal"]["noise"], size=1)
        s += noise

    # save signal
    print("Saving signal...")
    OUT_DIR = config["out"]["dir"]
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(f"{OUT_DIR}/{config['out']['files']['signal']}", s)


if __name__ == '__main__':
    # Read the arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--config", required=True,
                           help="YAML config file to read the method configuration.")
    args = vars(argParser.parse_args())
    configuration = YamlReader(args["config"]).config
    main(configuration)
