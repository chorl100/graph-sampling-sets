import math
from typing import Optional

import pygsp
from pygsp import graphs


def barabasi_albert(n_nodes: int, m: int = 1, seed: int = 0):
    """
    Builds a Barabási-Albert graph with n_nodes nodes.
    The graph is constructed by adding new nodes, each with m edges,
    that are preferentially attached to existing nodes with high degrees.
    :param n_nodes: number of nodes
    :param m: number of edges per new node
    :param seed: random seed
    :return: Barabási-Albert graph
    """
    barabasi_albert_graph = graphs.BarabasiAlbert(n_nodes, m=m, seed=seed)
    barabasi_albert_graph.set_coordinates(kind='spring', seed=seed)
    return barabasi_albert_graph


def community(n_nodes: int, n_communities: Optional[int] = None, seed: int = 0):
    """
    Builds a community graph.
    :param n_nodes: number of nodes
    :param n_communities: number of communities (optional). Default is floor(sqrt(n) / 2).
    :param seed: random seed
    :return: community graph
    """
    if n_communities is None:
        n_communities = math.floor(math.sqrt(n_nodes) / 2)
    # build community graph
    community_graph = graphs.Community(n_nodes, n_communities, seed=seed)
    if community_graph.is_connected():
        return community_graph
    W = community_graph.W
    coords = community_graph.coords
    nz = W.getnnz(axis=1) > 0
    # get largest connected component
    connected_subgraph = pygsp.graphs.Graph(W[nz][:, nz], coords=coords[nz])
    #components = community_graph.extract_components()
    #connected_subgraph = max(components, key=lambda graph: graph.N)
    #connected_subgraph.set_coordinates(community_graph.coords)
    return connected_subgraph


def minnesota():
    """
    Builds the Minnesota road graph with fixed N=2642 nodes.
    :return: Minnesota road graph
    """
    return graphs.Minnesota()


def sensor(n_nodes: int, seed: int = 0):
    """
    Builds a random sensor graph.
    :param n_nodes: number of nodes
    :param seed: random seed
    :return: random sensor graph
    """
    return graphs.Sensor(n_nodes, seed=seed)
