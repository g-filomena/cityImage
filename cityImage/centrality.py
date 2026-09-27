"""Network centrality helpers used by cityImage.

This module replaces the old ``graph_centrality.py`` module.

Design decision
---------------
cityImage no longer owns generic network centrality algorithms. Standard
centralities are delegated to NetworkX or igraph. cityImage keeps only the thin
API glue and the two project-specific metrics that were already part of the
library behaviour:

* reach centrality based on reachable node attributes within a network radius;
* straightness centrality using Euclidean/network-distance ratios.

``igraph`` is a core dependency but is still imported lazily (only when a centrality
operation runs), so ``import cityImage`` stays light.
"""

from __future__ import annotations

from math import sqrt
from typing import Any

import pandas as pd

pd.set_option("display.precision", 3)


def _import_igraph():
    """Import igraph lazily, keeping it out of module import so ``import cityImage`` stays light."""
    import igraph as ig

    return ig


def _nodes_dict(ig_graph: Any) -> dict[int, tuple[float, float]]:
    """Create a dictionary of igraph node indices and their x/y coordinates."""
    return {v.index: (v["x"], v["y"]) for v in ig_graph.vs}


def _assign_attributes_from_nx_to_igraph(nx_graph: Any, ig_graph: Any) -> None:
    """Transfer NetworkX node attributes to matching igraph vertices."""
    original_ids = list(nx_graph.nodes)
    for idx, node in enumerate(ig_graph.vs):
        nx_node_attrs = nx_graph.nodes[original_ids[idx]]
        for attr, value in nx_node_attrs.items():
            node[attr] = value


def _euclidean_distance(xs: float, ys: float, xt: float, yt: float) -> float:
    """Return Euclidean distance between source and target coordinates."""
    return sqrt((xs - xt) ** 2 + (ys - yt) ** 2)


def _nx_to_igraph(nx_graph: Any, weight: str) -> tuple[Any, list[Any]]:
    """Convert a NetworkX graph to igraph and preserve node attributes."""
    ig = _import_igraph()

    original_ids = list(nx_graph.nodes)
    id_to_index = {original_id: index for index, original_id in enumerate(original_ids)}

    # One igraph edge per NetworkX edge, parallel edges of a MultiGraph included; shortest paths
    # then take the lightest of them.
    edges = []
    weights = []
    for u, v, value in nx_graph.edges(data=weight):
        edges.append((id_to_index[u], id_to_index[v]))
        weights.append(value)

    ig_graph = ig.Graph(edges=edges, directed=nx_graph.is_directed())
    ig_graph.es["weight"] = weights
    _assign_attributes_from_nx_to_igraph(nx_graph, ig_graph)

    return ig_graph, original_ids


def _networkx_centrality(
    nx_graph: Any,
    *,
    measure: str,
    weight: str,
    normalized: bool,
) -> dict[Any, float]:
    """Delegate standard centralities to NetworkX."""
    if measure == "betweenness":
        return nx_graph.__class__.__module__ and __import__("networkx").betweenness_centrality(
            nx_graph,
            weight=weight,
            normalized=normalized,
        )

    if measure == "closeness":
        # NetworkX's closeness centrality uses the ``distance`` keyword for edge
        # distances. The function is normalised according to NetworkX semantics.
        return __import__("networkx").closeness_centrality(nx_graph, distance=weight)

    raise ValueError(f"NetworkX backend does not support measure: {measure}")


def calculate_centrality(
    nx_graph: Any,
    measure: str = "betweenness",
    weight: str = "weight",
    radius: float | None = None,
    attribute: str | None = None,
    normalized: bool = False,
    backend: str = "igraph",
) -> dict[Any, float]:
    """Calculate centrality values for a NetworkX graph.

    Parameters
    ----------
    nx_graph
        Input NetworkX graph: a ``Graph``, ``DiGraph`` or ``MultiGraph``. A directed graph gives
        directed measures; parallel edges of a ``MultiGraph`` are all kept.
    measure
        One of ``"betweenness"``, ``"closeness"``, ``"straightness"``, or
        ``"reach"``.
    weight
        Edge attribute used as distance/weight.
    radius
        Required for ``measure="reach"``.
    attribute
        Required for ``measure="reach"``; node attribute to sum over reachable
        nodes.
    normalized
        Normalisation flag. Preserves the existing straightness semantics and is
        also passed through to NetworkX betweenness when ``backend="networkx"``.
    backend
        ``"igraph"`` preserves historical cityImage behaviour. ``"networkx"``
        delegates standard betweenness/closeness directly to NetworkX. Reach and
        straightness always use the igraph-distance implementation because they
        are cityImage-specific helpers.

    Returns
    -------
    dict
        Mapping from original NetworkX node IDs to centrality values.
    """
    if measure in {"betweenness", "closeness"} and backend == "networkx":
        return _networkx_centrality(
            nx_graph,
            measure=measure,
            weight=weight,
            normalized=normalized,
        )

    if backend != "igraph":
        raise ValueError("backend must be either 'igraph' or 'networkx'")

    ig_graph, original_ids = _nx_to_igraph(nx_graph, weight=weight)

    if measure == "betweenness":
        centrality_values = ig_graph.betweenness(weights="weight")
    elif measure == "closeness":
        centrality_values = ig_graph.closeness(weights="weight")
    elif measure == "straightness":
        centrality_values = straightness_centrality(
            ig_graph,
            weight="weight",
            normalized=normalized,
        )
    elif measure == "reach":
        if radius is None or attribute is None:
            raise ValueError(
                "For reach centrality, both radius and attribute parameters must be provided"
            )
        centrality_values = reach_centrality(
            ig_graph,
            weight="weight",
            radius=radius,
            attribute=attribute,
        )
    else:
        raise ValueError(f"Unsupported centrality type: {measure}")

    return {original_ids[i]: centrality_values[i] for i in range(len(original_ids))}


def reach_centrality(
    ig_graph: Any,
    weight: str,
    radius: float,
    attribute: str,
) -> dict[int, float]:
    """Calculate reach centrality from an igraph graph.

    Reach centrality is the sum of a node attribute over all other nodes
    reachable within a network-distance radius.
    """
    n_nodes = ig_graph.vcount()
    reach: dict[int, float] = {}

    shortest_paths = ig_graph.distances(weights=weight)

    for node in range(n_nodes):
        sp = shortest_paths[node]
        reach[node] = sum(
            ig_graph.vs[target][attribute]
            for target, dist in enumerate(sp)
            if dist != float("inf") and dist <= radius and target != node
        )

    return reach


def straightness_centrality(
    ig_graph: Any,
    weight: str,
    normalized: bool = True,
) -> dict[int, float]:
    """Calculate straightness centrality from an igraph graph.

    Straightness is the mean ratio between Euclidean distance and weighted
    network distance to all reachable targets.
    """
    n_nodes = ig_graph.vcount()
    coord_nodes = _nodes_dict(ig_graph)
    values: dict[int, float] = {}

    shortest_paths = ig_graph.distances(weights=weight)

    for node in range(n_nodes):
        sp = shortest_paths[node]

        if len(sp) > 0 and n_nodes > 1:
            straightness = 0.0

            for target, network_dist in enumerate(sp):
                if node != target and network_dist < float("inf"):
                    euclidean_dist = _euclidean_distance(*coord_nodes[node], *coord_nodes[target])
                    straightness += euclidean_dist / network_dist

            values[node] = straightness / (n_nodes - 1.0)

            if normalized:
                reachable_nodes = sum(1 for dist in sp if dist < float("inf")) - 1
                if reachable_nodes > 0:
                    values[node] *= (n_nodes - 1.0) / reachable_nodes
                else:
                    values[node] = 0.0
        else:
            values[node] = 0.0

    return values


def weight_nodes(
    nodes_gdf: Any,
    services_gdf: Any,
    nx_graph: Any,
    field_name: str,
    radius: float = 400,
) -> Any:
    """Assign local opportunity counts to nodes and to the NetworkX graph.

    Graph nodes are matched by ``nodeID`` (as ``graph_fromGDF`` keys them), whatever the index of
    ``nodes_gdf``; the counts are also written to ``nodes_gdf[field_name]``.
    """
    sindex = services_gdf.sindex

    for node_id, node in nodes_gdf.iterrows():
        buffer = node["geometry"].buffer(radius)
        possible_matches_index = list(sindex.intersection(buffer.bounds))
        possible_matches = services_gdf.iloc[possible_matches_index]
        precise_matches = possible_matches[possible_matches.intersects(buffer)]
        weight_value = len(precise_matches)

        nodes_gdf.at[node_id, field_name] = weight_value
        # By nodeID: the index label is another node's, or none, unless the frame is indexed by it.
        graph_node = node["nodeID"] if "nodeID" in nodes_gdf.columns else node_id
        nx_graph.nodes[graph_node][field_name] = weight_value

    return nx_graph


def _dict_to_df(list_dict: list[dict[Any, Any]], list_col: list[str]) -> pd.DataFrame:
    """Build a DataFrame from aligned dictionaries without depending on utilities.py."""
    df = pd.DataFrame(list_dict).T
    df.columns = [f"d{i}" for i, _ in enumerate(df, 1)]
    df.columns = list_col
    return df


def append_edges_metrics(
    edges_gdf: pd.DataFrame,
    graph: Any,
    dicts: list[dict[Any, Any]],
    column_names: list[str],
) -> pd.DataFrame:
    """Attach edge-level centrality values to an edges GeoDataFrame.

    ``dicts`` map the graph's edges to values, keyed as the graph yields them: ``(u, v)`` for a
    ``Graph``, ``(u, v, key)`` for a ``MultiGraph`` - the keys
    ``networkx.edge_betweenness_centrality`` returns for each.

    A ``Graph`` (``graph_fromGDF``) holds one edge per pair of nodes, so parallel streets in
    ``edges_gdf`` are missing from it and would have no value. That raises a ``ValueError``: build
    a ``MultiGraph`` (``multiGraph_fromGDF``), which gives every street its own value.
    """
    if graph.is_multigraph():
        edge_ids = {
            (u, v, key): data["edgeID"] for u, v, key, data in graph.edges(keys=True, data=True)
        }
    else:
        edge_ids = {(u, v): data["edgeID"] for u, v, data in graph.edges(data=True)}
        missing = sorted(set(edges_gdf["edgeID"]) - set(edge_ids.values()))
        if missing:
            raise ValueError(
                f"{len(missing)} edge(s) are not in the graph, which holds one edge per pair of "
                "nodes: build it with multiGraph_fromGDF to give parallel streets their own "
                f"values. Missing edgeIDs: {missing[:20]}"
            )

    dicts = [*dicts, edge_ids]
    column_names = [*column_names, "edgeID"]

    tmp = _dict_to_df(dicts, column_names)
    tmp.edgeID = tmp.edgeID.astype(int)
    edges_gdf = pd.merge(edges_gdf, tmp, on="edgeID", how="left")
    edges_gdf.index = edges_gdf.edgeID
    edges_gdf.index.name = None
    return edges_gdf


class Error(Exception):
    """Base class for centrality exceptions."""


class columnError(Error):
    """Raised when a column name is not provided."""


class nameError(Error):
    """Raised when an unsupported or missing centrality name is requested."""
