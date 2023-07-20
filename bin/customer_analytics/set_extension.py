import argparse
import time

import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

import src.db.big_query as bq
from src.db.preprocessing import Preprocessor
from src.db.zip_code_mapper import ZipCodeMapper
from src.gershgorin.bs_gda import bs_gda_extension, bs_gda
from src.graph.graph import Graph
from src.graph.nearest_neighbors import NearestNeighbors
from src.gsp.reconstruction import reconstruct_signal, mse
from src.utils import layout
from src.utils.yaml_reader import YamlReader


def main(config):
    # specify a timeframe to query
    from_date = "2023-01-01"
    to_date = "2023-01-30"

    # load the ground truth
    car_df = bq.join_car_nps(from_date, to_date)

    prep = Preprocessor(from_date, to_date, data=car_df, verbose=False)
    car_df, client_ids, adr_zips = prep.car_df, prep.client_ids, prep.adr_zips

    # sample
    sample = np.random.choice(car_df.shape[0], 2000, replace=False)
    car_df = car_df.loc[sample].reset_index(drop=True)
    client_ids = client_ids[sample].reset_index(drop=True)
    adr_zips = adr_zips[sample].reset_index(drop=True)

    # load mapper for zip_code -> (longitude, latitude)
    zip_mapper = ZipCodeMapper()
    # load zip codes of customers
    adr_zip_df = pd.DataFrame(adr_zips, dtype=str)
    # remove unknown (unmappable) zip codes
    known_zips = adr_zip_df.adr_zip.isin(zip_mapper.zip_code_map.index)
    # apply mask to all three Dataframes
    adr_zips = adr_zip_df.loc[known_zips].reset_index(drop=True)
    car_df = car_df[known_zips].reset_index(drop=True)
    client_ids = client_ids[known_zips].reset_index(drop=True)

    # map zip code to coords
    coords = zip_mapper.map_zip_codes_to_coords(adr_zips)
    # remove zip codes, keep lat and long
    coords.drop(columns="adr_zip", inplace=True)

    # read the recommendation values (NPS data) that we will use as signal
    answers_df = bq.nps_query_timeframe(from_date, to_date)
    # filter for postpaid customers in survey answers (market = 'MMC')
    answers_df = answers_df[answers_df.client_id.isin(client_ids)]

    # store data as tensor on GPU
    X = torch.tensor(np.ascontiguousarray(car_df.to_numpy()), device=torch.device('cuda', 0), dtype=torch.float32)
    # compute k-nearest neighbor graph
    knn = NearestNeighbors(device="gpu")
    t = time.perf_counter()
    _, k_neighbors = knn.knn(X, k=50)
    print(f"This took {time.perf_counter() - t:.3f} s")

    # build adjacency matrix from neighborhood index
    A = knn.to_adj_matrix(k_neighbors)
    n_nodes = A.shape[0]

    # pick a random set
    if config["rand_preselect"]:
        preselection = np.random.choice(range(n_nodes), size=20, replace=False).tolist()
    else:
        preselection = config["preselection"]

    # specify max. number of nodes to sample
    sampling_budget = 200
    assert sampling_budget > len(preselection)
    p_hops = 12

    # wrap graph with adjacency list for faster neighborhood queries
    graph = Graph(A)

    start = time.perf_counter()
    sampling_set_extended, thres_extended = bs_gda_extension(graph, preselection, sampling_budget, p_hops=p_hops,
                                                             parallel=True)
    print(f"This took {time.perf_counter() - start:.3f} s")
    print("Budget:", sampling_budget)
    print("Sampled nodes:", len(sampling_set_extended))

    # map node_ids to client_ids
    pred_sampling_set = client_ids[sampling_set_extended]
    # which customers should be sampled in addition to the preselection?
    client_ids_to_sample = client_ids[set(sampling_set_extended) - set(preselection)]

    # print("The following clients should be sampled in addition:\n", client_ids_to_sample.to_numpy())

    # draw the solution
    # compute graph layout
    # get dict of node positions
    fixed_zip_pos = layout.pos_from_coords(coords)
    # scatter customers in a circle around their zip code
    pos = layout.circular_layout_around_zip_codes(fixed_zip_pos, radius=0.1)

    # plot preselected nodes and those that should be sampled in addition
    color_extended = util_plt.sample_to_node_color_vec(graph.num_nodes, sampling_set_extended)
    color_preselected = util_plt.sample_to_node_color_vec(graph.num_nodes, preselection)
    # color scheme: c[not_sampled] = 0, c[sampled] = 1, c[preselected] = 2
    c = color_extended + color_preselected
    util_plt.draw_graph(nx.from_scipy_sparse_matrix(A), pos, node_size=20, node_color=c, hide_edges=True,
                        cmap=plt.cm.copper)# use recommendation values as signal

    s = answers_df.answer_value.to_numpy().astype(int)
    # reconstruct the original signal from the preselected signal
    L = graph.laplacian()
    t = time.perf_counter()
    s_recon = reconstruct_signal(L, preselection, s[preselection])
    print(f"This took {time.perf_counter()-start:.3f} s")
    print("MSE (preselection only)", mse(s, s_recon))

    # reconstruct the original signal from the extended sampled signal
    t = time.perf_counter()
    s_recon = reconstruct_signal(L, sampling_set_extended, s[sampling_set_extended])
    print(f"This took {time.perf_counter() - start:.3f} s")
    print("MSE (extended):", mse(s, s_recon))

    start = time.perf_counter()
    sampling_set, thres = bs_gda(graph, sampling_budget, p_hops=p_hops, parallel=True)
    print(f"This took {time.perf_counter() - start:.3f} s")
    print("Budget:", sampling_budget)
    print("Sampled nodes:", len(sampling_set))

    sampling_set = np.random.choice(range(n_nodes), 100, replace=False).tolist()
    s_recon = reconstruct_signal(L, sampling_set, s[sampling_set])
    mse_ = mse(s, s_recon)
    print("MSE without preselection:", mse_)


if __name__ == '__main__':
    # Read the arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-c", "--config", required=True,
                           help="YAML config file to read the method configuration.")
    args = vars(argParser.parse_args())
    config = YamlReader(args["config"]).config
    main(config)
