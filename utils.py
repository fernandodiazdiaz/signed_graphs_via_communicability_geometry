import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.linalg import expm, eigh
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import MDS


def adjacency_matrix(obj):
    """
    Converts a NetworkX graph to an adjacency matrix if needed.
    Input:
        obj: NetworkX graph or NumPy array
    Output:
        A: adjacency matrix (NumPy array)
    """
    if isinstance(obj, nx.Graph):
        return nx.to_numpy_array(obj)
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        raise ValueError("Input must be a NetworkX graph or a NumPy array")



def communicability_metrics(obj, output="all"):
    """
    Computes communicability metrics for a network.
    Input:
        obj: NetworkX graph or NumPy array
        nodelist: List of nodes in a specific order (default: None)
        output: Which metric to return ('all', 'comm', 'distance', 'angle', 'comm_coordinates')
    Output:
        Metrics as per output specification:
            - "comm": Signed communicability matrix
            - "distance": Communicability distance matrix
            - "angle": Communicability angle matrix
            - "comm_coordinates": Position vectors of nodes in communicability space
            - "all": All the above
    """
    A = adjacency_matrix(obj)
    N = len(A)

    S = expm(A)  # Signed communicability matrix
    K = np.diag(S)

    if output in ["all", "comm_coordinates"]:
        Lamb, U = eigh(A)
        X = np.diag(np.exp(Lamb / 2)) @ U.T

    if output in ["all", "distance"]:
        xi = np.sqrt(np.outer(K, np.ones(N)) + np.outer(np.ones(N), K) - 2 * S)

    if output in ["all", "angle"]:
        cos_theta = S / np.sqrt(np.outer(K, np.ones(N)) * np.outer(np.ones(N), K))
        cos_theta = np.clip(cos_theta, -1, 1)  # Correct numerical errors
        theta = np.arccos(cos_theta)

    if output == "all":
        return S, xi, theta, X
    elif output == "comm":
        return S
    elif output == "distance":
        return xi
    elif output == "angle":
        return theta
    elif output == "comm_coordinates":
        return X
    else:
        raise ValueError("Invalid output type")
    


def compute_clustering(G):

    N = G.order()

    # Compute communicability angle
    distance = communicability_metrics(G, output="angle")
    # distance = 2 * (1 - np.cos(theta))  # Euclidean angle distance
    distance = (distance + distance.T) / 2  # Ensure symmetry

    # Optimize number of factions and MDS dimension
    MDS_dimension_v = [2, 3, 5, max(int(N / 10), 7)]
    n_factions_v = range(2, min(10, N))

    best_labels = None
    max_score = -np.inf

    for MDS_dimension in MDS_dimension_v:
        low_dim_embedding = MDS(n_components=MDS_dimension, dissimilarity="precomputed").fit_transform(distance)
        for n_factions in n_factions_v:
            labels = KMeans(n_clusters=n_factions).fit(low_dim_embedding).labels_

            if len(set(labels)) > 1 and len(set(labels)) < len(labels):  # Avoid degenerate cases
                score = silhouette_score(distance, labels, metric="precomputed")
            else:
                score = 0  # Invalid clustering

            if score > max_score + 1e-4:  # select only significant improvements
                max_score = score
                best_labels = labels

    return best_labels




def multidimensional_scaling(G, embedding_dimension=2):

    # compute communicability metrics
    distance = communicability_metrics(G, output = 'angle')

    # correct rounding errors
    assert np.allclose(distance, distance.T, atol=1e-3), "Distance not symmetric"
    distance = (distance + distance.T) / 2

    # embed in a low dimensional space
    embedding = MDS(n_components=embedding_dimension, dissimilarity="precomputed")
    coords = embedding.fit_transform(distance)

    # assign coordinates to each node
    pos = {}
    for k, node in enumerate(G.nodes()):
        pos[node] = tuple(coords[k, :])

    return pos



def position_underlying_graph(G):
    G_abs = nx.Graph()
    for u, v, w in G.edges(data="weight"):
        G_abs.add_edge(u, v)
    pos = nx.kamada_kawai_layout(G_abs)
    return pos

def draw_network(
    obj,
    ax=None,
    pos=None,
    labels=None,
    node_size=500,
    node_color="white",
    nodeedge_color="k",
    cmap_nodes=None,
    label_fontsize=12,
    pos_edge_width=2,
    neg_edge_width=2,
    pos_ls="-",
    neg_ls="--",
    with_labels=False,
    spines=False,
    differenciate_groups=False,
    group_assignments=None,
    group_markers=None,
    group_colors=None,
):
    """
    Draw a networkx graph with positive and negative edges.

    Parameters:
    -----------
    obj : networkx graph or numpy array
        The network or its adjacency matrix.
    ax : matplotlib.axes.Axes, optional
        Axis to draw the graph on.
    pos : dict, optional
        Dictionary of node positions. If None, a Kamada-Kawai layout is used.
    labels : dict, optional
        Node labels.
    node_size : int, default 500
        Size of the nodes.
    node_color : str, default "white"
        Default color of the nodes.
    nodeedge_color : str, default "k"
        Color of the node borders.
    cmap_nodes : matplotlib colormap, optional
        Colormap for the nodes.
    label_fontsize : int, default 12
        Font size for node labels.
    pos_edge_width : int, default 2
        Width for positive edges.
    neg_edge_width : int, default 2
        Width for negative edges.
    pos_ls : str, default "-"
        Linestyle for positive edges.
    neg_ls : str, default "--"
        Linestyle for negative edges.
    with_labels : bool, default False
        If True, draw node labels.
    spines : bool, default False
        If False, the axes spines and ticks are hidden.

    New Optional Parameters:
    -------------------------
    differenciate_groups : bool, default False
        If True, nodes will be drawn by group (using different markers and colors).
    group_assignments : dict, optional
        Dictionary mapping nodes to group labels. Required if differenciate_groups is True.
    group_markers : dict, optional
        Dictionary mapping group labels to marker styles. If not provided, default markers are used.
    group_colors : dict, optional
        Dictionary mapping group labels to colors. If not provided, default colors are used.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axis with the drawn graph.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    # Convert input to a networkx graph if needed.
    if isinstance(obj, nx.Graph):
        G = obj
    elif isinstance(obj, np.ndarray):
        G = nx.from_numpy_array(obj)
    else:
        raise ValueError("Input must be a networkx graph or a numpy array")

    # Compute positions if not provided.
    if pos is None:
        pos = position_underlying_graph(G)

    # Ensure all edges have a weight; if not, assign a default weight 1.
    for u, v, w in G.edges(data="weight"):
        if w is None:
            G[u][v]["weight"] = 1

    # Set edge attributes: color, linestyle, and width.
    edge_color = []
    ls = []
    width = []
    for u, v, w in G.edges(data=True):
        if w["weight"] > 0:
            edge_color.append("darkgreen")
            ls.append(pos_ls)
            width.append(pos_edge_width)
        else:
            edge_color.append("red")
            ls.append(neg_ls)
            width.append(neg_edge_width)

    # Draw nodes: either by groups or with default style.
    if differenciate_groups:
        if group_assignments is None:
            raise ValueError(
                "group_assignments must be provided when differenciate_groups is True"
            )
        # Organize nodes by group.
        groups = {}
        for node, group in group_assignments.items():
            groups.setdefault(group, []).append(node)
        # Set default colors if not provided.
        if group_colors is None:
            default_colors = plt.cm.tab10.colors
            group_colors = {
                grp: default_colors[i % len(default_colors)]
                for i, grp in enumerate(sorted(groups.keys()))
            }
        # Set default markers if not provided.
        if group_markers is None:
            default_markers = ["o", "s", "^", "D", "v", "p", "*", "X", "8"]
            group_markers = {
                grp: default_markers[i % len(default_markers)]
                for i, grp in enumerate(sorted(groups.keys()))
            }
        # Draw nodes for each group.
        for grp, nodes in groups.items():
            nx.draw_networkx_nodes(
                G,
                pos=pos,
                ax=ax,
                nodelist=nodes,
                node_size=node_size,
                node_color=[group_colors[grp]],
                node_shape=group_markers[grp],
                edgecolors=nodeedge_color,
            )
    else:
        # Default node drawing.
        if cmap_nodes is not None:
            nodes = nx.draw_networkx_nodes(
                G,
                pos=pos,
                ax=ax,
                node_size=node_size,
                node_color=node_color,
                cmap=cmap_nodes,
            )
        else:
            nodes = nx.draw_networkx_nodes(
                G, pos=pos, ax=ax, node_size=node_size, node_color=node_color
            )
        nodes.set_edgecolor(nodeedge_color)

    # Draw edges.
    nx.draw_networkx_edges(
        G, pos=pos, ax=ax, edge_color=edge_color, style=ls, width=width
    )
    if with_labels:
        nx.draw_networkx_labels(
            G, pos=pos, ax=ax, labels=labels, font_size=label_fontsize
        )
    if not spines:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

    return ax