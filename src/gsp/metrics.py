import networkx as nx
import numpy as np

from src.gershgorin.bs_gda import bs_gda
from src.graph.graph import Graph
from src.gsp.signal import propagate_signal


def sampling_centrality(graph: nx.Graph, W, k: int, p: float):
    """Returns the sampling centrality value for each node of a graph.
    :param graph: graph
    :param W: weight matrix of the graph
    :param k: sampling budget
    :param p: propagation probability
    :return: sampling centrality vector
    """
    # compute sampling set
    sampling_set, _ = bs_gda(Graph(W), k)
    # make discrete centralities continuous
    propagated_centrality = [propagate_signal(graph, p, node) for node in sampling_set]
    return np.median(np.vstack(propagated_centrality), axis=0)


def z_score(centralities):
    """Computes the Z-scores."""
    return (centralities - np.mean(centralities)) / np.std(centralities)
