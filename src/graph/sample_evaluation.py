from collections import OrderedDict

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

from src.utils.plotting import draw_graph, plt_legend


def plot_sample_classes(graph: nx.Graph, colors: dict, labels: list, pos: dict = None, size=10,
                        subgraph: bool = False, select: list = None,
                        hide_edges: bool = False, layout_iters: int = 15,
                        figsize: tuple = (6, 6), cmap=plt.cm.viridis):
    # compute a graph layout
    pos = nx.spring_layout(graph, iterations=layout_iters) if pos is None else pos
    n_nodes = len(graph.nodes)
    node_color = np.zeros(n_nodes)
    sizes = np.ones(n_nodes)
    if subgraph and select is not None:
        # only draw a subset of the nodes
        for label, nodes in reversed(colors.items()):
            if label not in select:
                continue
            node_color[nodes] = label
            sizes[nodes] = size
    else:
        for label, nodes in reversed(colors.items()):
            node_color[nodes] = label
        sizes = np.full_like(sizes, size)
    draw_graph(graph, pos, node_color=node_color, node_size=sizes, hide_edges=hide_edges, figsize=figsize, cmap=cmap)
    plt_legend(list(colors.keys()), labels, cmap)


def prepare_colors_labels(n_nodes, predicted_set, actual_set):
    # compute sample sets (indices from {1, ..., n})
    intersect_set = list(set(predicted_set) & set(actual_set))
    not_sampled = set(range(n_nodes)) - set(actual_set) - set(predicted_set)
    # define color scheme
    colors = {0: list(not_sampled),  # nodes that appear in neither sampling set
              1: list(intersect_set),  # overlap of predicted and actually sampled nodes
              2: list(actual_set),  # actually sampled nodes
              3: list(predicted_set)  # predicted sampling set
              }
    colors = OrderedDict(colors)
    labels = ["Not sampled", r"Pred $\cap$ Actual", "Actually sampled", "Predicted"]
    return colors, labels


def neighborhood(graph: nx.Graph, sampling_set: list, actual_set: list, max_hops: int = 12):
    """
    How large do we have to choose the p-hop neighborhood of the actually sampled nodes
    to cover all nodes of the predicted sampling set?
    :param graph: graph
    :param sampling_set: sampling set
    :param actual_set: actually sampled nodes
    :param max_hops: maximum number of hops to do in a neighborhood
    :return: maximum distance from a sampled node to an actually sampled node
    """
    n_nodes = len(graph.nodes)
    # mark actually sampled nodes as uncovered
    uncovered = np.zeros(n_nodes, dtype=bool)
    uncovered[actual_set] = 1
    depths = range(1, max_hops + 1)
    for depth in depths:
        for node in sampling_set:
            # do a limited BFS
            neighbors = list(nx.bfs_tree(graph, node, depth_limit=depth).nodes())
            # mark visited nodes
            uncovered[neighbors] = 0
        if not any(uncovered):
            # if all nodes in actual_set are covered, return the current depth
            return depth
    # if not all nodes were covered, return -1
    return -1
