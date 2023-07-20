import networkx as nx
import numpy as np
import typing
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib_venn import venn2

from src.utils import colors


def plot_runtime(sizes, runtimes, name, out="."):
    plt.figure()
    plt.plot(sizes, runtimes, marker="o")
    plt.title(f"Runtime comparison on {name} graph")
    plt.xlabel("Graph size")
    plt.ylabel("Runtime (s)")
    plt.savefig(f"{out}/runtimes_{name}.pdf")
    plt.show()


def plot_reconstruction_error_paper(sampling_budgets, errors, graph_name: str, signal_func: str,
                                    figsize=(6,4), out: str = None):
    plt.figure(figsize=figsize)
    plt.plot(sampling_budgets, errors, marker="o")
    plt.title(f"Reconstruction MSE on the {graph_name} graph ({signal_func.upper()})")
    plt.xlabel("Sampling budget")
    plt.ylabel("Reconstruction MSE")
    plt.xticks(sampling_budgets[range(1, len(sampling_budgets), 2)])
    if out is not None:
        plt.savefig(f"{out}/reconstruction_error_{signal_func}_{graph_name}.pdf", bbox_inches='tight')


def plot_reconstruction_error(sampling_budgets, errors, param_name: str, xlabel: str,
                              n_nodes: typing.Optional[int] = None,
                              figsize=(6, 4), filepath=None):
    plt.figure(figsize=figsize)
    plt.plot(sampling_budgets, errors, marker="o")
    title = f"Reconstruction error for growing {param_name}"
    if n_nodes is not None:
        title += f"\nN={n_nodes}"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Reconstruction MSE")
    plt.xticks(sampling_budgets[range(1, len(sampling_budgets), 2)])
    if filepath is not None:
        plt.savefig(filepath)


def plot_stems(x, y, figsize=(6, 4)):
    plt.figure(figsize=figsize)
    plt.stem(x, y, '-o')
    plt.ylim([0, 2])
    plt.xlabel('node id')
    plt.title('sampling on graph')


def plot_cdf_gft_energy(eigs, gft_coeffs, index=False, figsize=(6, 4)):
    y = np.cumsum(gft_coeffs)
    y = y / y[-1]  # normalize to [0, 1]

    plt.figure(figsize=figsize)
    if index:
        plt.plot(range(len(eigs)), y)
        plt.xlim(0, len(eigs))
        plt.xlabel("Eigenvalue index")
    else:
        plt.plot(eigs, y)
        plt.xlim(0, max(eigs))
        plt.xlabel(r"$\lambda_i$")
    plt.title("Cumulative distribution of energy in the GFT coefficients")
    plt.ylabel("CDF of energy in GFT coefficients")


def plot_gft_coeff_dist(gft_coeffs, bins: int, inverse=False, histtype="bar", figsize=(6, 4)):
    cumulative = -1 if inverse else 1
    plt.figure(figsize=figsize)
    plt.hist(gft_coeffs, bins, density=True, cumulative=cumulative, histtype=histtype)
    title = "cumulative distribution of GFT coefficients"
    ylabel = "CDF of GFT coefficients"
    if inverse:
        title = f"Inverse {title}"
        ylabel = "I" + ylabel
    plt.title(title.capitalize())
    plt.xlabel(r"$g(\lambda_i)$")
    plt.ylabel(ylabel)
    plt.xlim(0, max(gft_coeffs))


def plot_spectral_domain(eigvals, gft_coeffs, yscale="linear", figsize=(6, 4)):
    plt.figure(figsize=figsize)
    plt.stem(eigvals, abs(gft_coeffs))
    plt.title("Signal in the spectral domain")
    plt.xlabel(r"$\lambda_i$")
    plt.ylabel(r"$g(\lambda_i)$")
    plt.yscale(yscale)


def plot_adj_matrix(A, markersize=2, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.title("Adjacency matrix")
    plt.spy(A, markersize=markersize)


def draw_graph(G, pos=None, node_size=10, node_color=None,
               width=0.5, hide_edges=False, figsize=(6, 6),
               title=None, cmap=plt.cm.viridis, filepath=None):
    if pos is None:
        pos = nx.spring_layout(G)
    edgelist = [] if hide_edges else list(G.edges())
    plt.figure(figsize=figsize)
    nx.draw(G, pos=pos, node_size=node_size, node_color=node_color, edgelist=edgelist, width=width, cmap=cmap)
    if title is not None:
        plt.title(title)
    if filepath is not None:
        plt.savefig(filepath)


def scatter_graph_color_vec(pos, node_size=10, node_color=None, figsize=(6, 6), title=None, alpha=None,
                            cmap="coolwarm", colorbar=True, filepath=None):
    x = [x for x, y in pos.values()]
    y = [y for x, y in pos.values()]
    plt.figure(figsize=figsize)
    plt.scatter(x, y, s=node_size, c=node_color, alpha=alpha, cmap=cmap)
    plt.axis("off")
    if title is not None:
        plt.title(title, pad=5)
    if colorbar:
        plt.colorbar()
    if filepath is not None:
        plt.savefig(filepath)


def scatter_graph(pos: dict, sample1: list, sample2: list, labels: typing.List[str],
                  hide_not_sampled=False, with_intersect=True, node_size=10, alpha=None,
                  figsize=(6, 6), title=None, filepath=None):
    """
    Draws two sampling sets on a graph. Draws the nodes without the edges using the layout from pos.
    This method does not require the creation of a networkx.Graph which makes it very efficient.
    Hiding not sampled nodes increases readability. The colors are based on the Vodafone color scheme.
    Note: Sample2 is always drawn without the elements of sample1.
    (Reason: in the case preselection vs. extended set, the preselection is a true subset of the extension
    so all elements of the preselection would overlap with the extension.)
    :param pos: graph layout (node positions)
    :param sample1: first sampling set. Will be plotted as a triangle down in gold
    :param sample2: second sampling set. Will be plotted as a star in Vodafone red
    :param labels: list of names
    :param hide_not_sampled: whether to hide nodes that are in neither of the sampling sets
    :param with_intersect: whether to draw intersection values with a special marker sign (circle)
    :param node_size: node size
    :param alpha: the alpha blending value, between 0 (transparent) and 1 (opaque)
    :param figsize: figure size
    :param title: title to plot
    :param filepath: storage location of output file
    """
    x = np.array([x for x, y in pos.values()])
    y = np.array([y for x, y in pos.values()])
    plt.figure(figsize=figsize)
    plots = list()
    n_nodes = len(pos)
    # scatter not sampled nodes first
    if not hide_not_sampled:
        sampled = np.isin(range(n_nodes), set(range(len(pos))) & set(sample1) & set(sample2))
        p0 = plt.scatter(x[~sampled], y[~sampled], s=node_size, c=colors.VF_SLATE_GREY, marker='.', alpha=alpha)
        plots.append(p0)
    # then scatter sampled nodes
    # plot sample1
    if len(sample1) > 0:
        if with_intersect:
            sample1_diff_sample2 = list(set(sample1) - set(sample2))
            p1 = plt.scatter(x[sample1_diff_sample2], y[sample1_diff_sample2],
                             s=node_size, c=colors.LEMON_YELLOW, marker='v', alpha=alpha)
        else:
            p1 = plt.scatter(x[sample1], y[sample1], s=node_size, c=colors.LEMON_YELLOW, marker='v', alpha=alpha)
        plots.append(p1)
    # plot sample2
    if len(sample2) > 0:
        sample2_diff_sample1 = list(set(sample2) - set(sample1))
        p2 = plt.scatter(x[sample2_diff_sample1], y[sample2_diff_sample1],
                         s=node_size, c=colors.VF_RED, marker='*', alpha=alpha)
        plots.append(p2)
    # intersection
    intersect = list(set(sample1) & set(sample2))
    if len(intersect) > 0 and with_intersect:
        p3 = plt.scatter(x[intersect], y[intersect], s=node_size, c=colors.TURQUOISE, marker='o', alpha=alpha)
        plots.append(p3)
    plt.axis("off")
    plt.legend(plots, labels)
    if title is not None:
        plt.title(title)
    if filepath is not None:
        plt.savefig(filepath, dpi=300)


def sample_to_node_color_vec(n_nodes: int, sample):
    node_color = np.zeros(n_nodes)
    node_color[sample] = 1
    return node_color


def draw_subgraph(graph, nodes, pos=None, node_size=50):
    subgraph = graph.subgraph(nodes)
    pos = nx.spring_layout(graph) if pos is None else pos
    draw_graph(subgraph, pos, node_size=node_size)


def plt_legend(colors, labels, cmap=plt.cm.viridis, filepath=None):
    m = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=min(colors), vmax=max(colors)), cmap=cmap)
    colors = m.to_rgba(colors)

    legend_elements = list()
    for color, label in zip(colors, labels):
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', label=label, markerfacecolor=color, markersize=10)
        )
    plt.legend(handles=legend_elements)
    if filepath is not None:
        plt.savefig(filepath)


def plot_eig_lower_bound_paper(sampling_budgets, lower_bounds, graph_name, signal_func, figsize=(6, 4), out="."):
    plt.figure(figsize=figsize)
    plt.plot(sampling_budgets, lower_bounds, marker="o")
    plt.title(r"Max. lower bound on $\lambda_\min$ " + f"{graph_name} graph ({signal_func.upper()})")
    plt.xlabel("Sampling budget")
    plt.ylabel(r"$\lambda_\min$")
    plt.savefig(f"{out}/max_lower_bound_{signal_func}_{graph_name}.pdf", bbox_inches='tight')


def plot_eig_lower_bound(sampling_budgets, lower_bounds, param_name: str, xlabel: str,
                         n_nodes: typing.Optional[int] = None,
                         figsize=(6, 4), filepath=None):
    plt.figure(figsize=figsize)
    plt.plot(sampling_budgets, lower_bounds)
    title = fr"Max. lower bound on $\lambda_\min$ for growing {param_name}"
    if n_nodes is not None:
        title += f"\nN={n_nodes}"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(r"$\lambda_\min$")
    plt.xticks(sampling_budgets[range(1, len(sampling_budgets), 2)])
    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')


def plot_mse_bar(samples, mses, filepath=None, figsize=(5, 5)):
    plt.figure(figsize=figsize)
    plt.bar(samples, mses)
    plt.title("MSE of reconstructed signal")
    plt.xlabel("Sample")
    plt.ylabel("MSE")
    if filepath is not None:
        plt.savefig(filepath)


def plot_smoothness_bar(samples, smoothness, filepath=None, figsize=(6, 6)):
    plt.figure(figsize=figsize)
    plt.bar(samples, smoothness)
    plt.title("Signal smoothness")
    plt.xlabel("Sample")
    plt.ylabel(r"$\frac{x^TLx}{x^Tx}$")
    if filepath is not None:
        plt.savefig(filepath)


def plot_signal_smoothness(sampling_budgets, smoothness, param_name: str, xlabel: str,
                           n_nodes: typing.Optional[int] = None,
                           figsize=(6, 4), filepath: typing.Optional[str] = None):
    plt.figure(figsize=figsize)
    plt.plot(sampling_budgets, smoothness)
    title = f"Signal smoothness for growing {param_name}"
    if n_nodes is not None:
        title += f"\nN={n_nodes}"
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Smoothness")
    plt.xticks(sampling_budgets[range(1, len(sampling_budgets), 2)])
    if filepath is not None:
        plt.savefig(filepath)


def plot_venn2_diagram(subsets: typing.List[set], labels: typing.List[str], set_colors: typing.Optional[list] = None,
                       alpha: float = 0.5, figsize=(6, 6), filepath: typing.Optional[str] = None):
    plt.figure(figsize=figsize)
    if set_colors is None:
        set_colors = ("r", "blue")
    venn2(subsets, labels, set_colors, alpha=alpha)
    if filepath is not None:
        plt.savefig(filepath)


def plot_roc_curve(roc_auc, fpr, tpr, filename=None, close=False):
    """
    Plots the Receiver Operating Characteristic (ROC) curve
    Adapted from
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    :param roc_auc: ROC AUC score
    :param fpr: false positive rate
    :param tpr: true positive rate
    """
    fig = plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    if filename is not None:
        plt.savefig(filename)
    if close:
        plt.close(fig)


def plot_precision_recall_curve(precision_vals: np.ndarray, recall_vals: np.ndarray,
                                title: str = "Precision-recall curve", filepath=None):
    """
    Draws a precision-recall curve.
    :param precision_vals: Precision values such that element i is the precision of predictions with score >= thresholds[i] and the last element is 1.
    :param recall_vals: Decreasing recall values such that element i is the recall of predictions with score >= thresholds[i] and the last element is 0.
    :param title: figure title
    :param filepath: output filepath
    """
    plt.plot(recall_vals, precision_vals)
    plt.title(title)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    if filepath is not None:
        plt.savefig(filepath)
