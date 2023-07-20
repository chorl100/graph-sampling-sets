import argparse
import os

import numpy as np

import src.db.big_query as bq
from src.db.preprocessing import Preprocessor
from src.utils import data_handler
from src.utils.yaml_reader import YamlReader


def main(config):
    print("Preprocessing the data...")
    car_df, client_ids, adr_zips = _preprocess(**config["db"])

    OUT_DIR = config["out"]["dir"]
    os.makedirs(OUT_DIR, exist_ok=True)
    files = config["out"]["files"]

    print("Saving cleaned CAR data...")
    car_df.to_parquet(f"{OUT_DIR}/{files['data']}", index=True)
    print("Saving client ids...")
    data_handler.save_df_as_csv(client_ids, f"{OUT_DIR}/{files['client_ids']}")
    print("Saving addresses...")
    data_handler.save_df_as_csv(adr_zips, f"{OUT_DIR}/{files['zip_codes']}")
    print("Saving successful.")


def _preprocess(from_date, to_date, sample=False, sample_size=None):
    """
    Preprocesses the data. For details, please refer to :class:`src.db.preprocessing.Preprocessor`.
    :param from_date: start date
    :param to_date: end date
    :param sample: whether to pick a random sample of the CAR data (if True, then sample size must be specified)
    :param sample_size: size of the random sample (mandatory if sample is set to True)
    :return: preprocessed CAR DataFrame, array of client ids, zip codes
    """
    # load the ground truth
    car_df = bq.join_car_nps(from_date, to_date)
    # preprocess the data
    prep = Preprocessor(from_date, to_date, data=car_df, verbose=False)
    car_df, client_ids, adr_zips = prep.car_df, prep.client_ids, prep.adr_zips
    if sample:
        assert sample_size > 0
        # take a sample
        sample = choose_random_sample(car_df.shape[0], sample_size)
        adr_zips = adr_zips[sample].reset_index(drop=True)
        car_df = car_df.loc[sample].reset_index(drop=True)
        client_ids = client_ids[sample].reset_index(drop=True)
    return car_df, client_ids, adr_zips


def choose_random_sample(n_nodes: int, size: int) -> np.ndarray:
    """
    Generates a random sample from a given range of numbers.
    :param n_nodes: numbers from the range [0, n_nodes] will be sampled
    :param size: sample size
    :return: random sample
    """
    return np.random.choice(range(n_nodes), size, replace=False).tolist()


if __name__ == '__main__':
    # Read the arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--config", required=True,
                           help="YAML config file to read the method configuration.")
    args = vars(argParser.parse_args())
    configuration = YamlReader(args["config"]).config
    main(configuration)
