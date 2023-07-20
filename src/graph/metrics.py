def density(n: int, m: int, mode: str = "undirected") -> float:
    """
    Returns the graph density.
    :param n: number of nodes
    :param m: number of edges
    :param mode: whether the graph is "directed" or "undirected"
    :return: graph density
    """
    if mode == "directed":
        return m / (n*(n-1))
    else:
        return 2*m / (n*(n-1))
