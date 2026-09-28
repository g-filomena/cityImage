"""Core graph construction and dual-graph semantics.

This module is intentionally small. It keeps only the cityImage graph boundary:

* convert prepared node/edge GeoDataFrames into NetworkX graphs;
* build the dual graph representation used by imageability/region workflows;
* map dual-graph results back to primal edge IDs;
* calculate simple node degree counts from edge tables.

Network loading, topology cleaning, centrality, and community detection now live
in dedicated modules or are delegated to external libraries.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from .angles import angle_line_geometries

pd.set_option("display.precision", 3)


def _is_missing_scalar(value: Any) -> bool:
    """Return True for scalar missing values, False for list-like/geometries."""
    if isinstance(value, list):
        return False

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _node_attribute_columns(gdf: pd.DataFrame) -> list[str]:
    """Return node attribute columns safe to attach to NetworkX nodes."""
    return [
        column
        for column in gdf.columns
        if not gdf[column].apply(lambda value: isinstance(value, list)).any()
    ]


def _set_node_attributes_from_gdf(
    graph: nx.Graph,
    nodes_gdf: gpd.GeoDataFrame,
) -> None:
    """Attach non-list, non-missing node attributes to a NetworkX graph."""
    attributes = nodes_gdf.to_dict()

    for attribute_name in _node_attribute_columns(nodes_gdf):
        attribute_values = {
            key: value
            for key, value in attributes[attribute_name].items()
            if not _is_missing_scalar(value)
        }
        nx.set_node_attributes(graph, values=attribute_values, name=attribute_name)


def _edge_attributes(row: pd.Series, exclude: set[str]) -> dict[str, Any]:
    """Return edge attributes to attach to a NetworkX edge."""
    return {
        label: value
        for label, value in row.items()
        if label not in exclude and (isinstance(value, list) or not _is_missing_scalar(value))
    }


def graph_fromGDF(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
) -> nx.Graph:
    """Create an undirected NetworkX graph from cityImage node/edge GeoDataFrames.

    A ``nx.Graph`` holds one edge per pair of nodes. Where parallel edges (different streets
    between the same two nodes) join a pair, the shortest is the one kept: the only one a
    shortest path uses, so shortest-path measures are exact. Use ``multiGraph_fromGDF`` to keep
    all of them.
    """
    nodes = nodes_gdf.copy()
    edges = edges_gdf.copy()

    nodes = nodes.set_index("nodeID", drop=False)
    nodes.index.name = None

    pair = pd.Series(
        [frozenset((u, v)) for u, v in zip(edges["u"], edges["v"], strict=True)],
        index=edges.index,
    )
    if pair.duplicated().any():
        order = np.lexsort((np.arange(len(edges)), edges.geometry.length.to_numpy()))
        edges = edges.iloc[order]
        edges = edges[~pair.iloc[order].duplicated().to_numpy()]

    graph = nx.Graph()
    graph.add_nodes_from(nodes.index)
    _set_node_attributes_from_gdf(graph, nodes)

    for _, row in edges.iterrows():
        graph.add_edge(row["u"], row["v"], **_edge_attributes(row, {"u", "v"}))

    return graph


def multiGraph_fromGDF(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
) -> nx.MultiGraph:
    """Create an undirected NetworkX MultiGraph from cityImage graph GeoDataFrames.

    Every street is an edge, parallel streets between the same two nodes included, so edge
    measures (``networkx.edge_betweenness_centrality``, ``append_edges_metrics``) give each its
    own value. An edge takes its ``key`` column value as key, or a fresh key when that one is
    taken or absent.
    """
    nodes = nodes_gdf.copy()
    edges = edges_gdf.copy()

    nodes = nodes.set_index("nodeID", drop=False)
    nodes.index.name = None

    multigraph = nx.MultiGraph()
    multigraph.add_nodes_from(nodes.index)
    _set_node_attributes_from_gdf(multigraph, nodes)

    for _, row in edges.iterrows():
        key = row["key"] if "key" in row.index else None
        # network_from_lines keys every edge 0, so parallel streets share a key: a fresh one keeps
        # the second street instead of overwriting the first.
        if key is not None and multigraph.has_edge(row["u"], row["v"], key):
            key = None
        multigraph.add_edge(
            row["u"],
            row["v"],
            key=key,
            **_edge_attributes(row, {"u", "v", "key"}),
        )

    return multigraph


def _intersecting_edge_ids(edges: gpd.GeoDataFrame, row: pd.Series) -> list[Hashable]:
    """Return edge IDs sharing either endpoint with an edge row."""
    return list(
        edges.loc[
            (edges["u"] == row["u"])
            | (edges["u"] == row["v"])
            | (edges["v"] == row["v"])
            | (edges["v"] == row["u"])
        ].index
    )


def _oneway_intersecting_edge_ids(edges: gpd.GeoDataFrame, row: pd.Series) -> list[Hashable]:
    """Return directed/oneway-aware dual-neighbour edge IDs."""
    if row["oneway"] == 1:
        mask = (edges["u"] == row["v"]) | ((edges["v"] == row["v"]) & (edges["oneway"] == 0))
    else:
        mask = (
            (edges["u"] == row["v"])
            | ((edges["v"] == row["v"]) & (edges["oneway"] == 0))
            | (edges["u"] == row["u"])
            | ((edges["v"] == row["u"]) & (edges["oneway"] == 0))
        )
    return list(edges.loc[mask].index)


def dual_gdf(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    crs: Any,
    oneway: bool = False,
    angle: str | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create dual-node and dual-edge GeoDataFrames from a primal street graph.

    Dual nodes represent primal street segments. Dual edges connect street
    segments sharing a junction. Their length is the mean of the two original
    segment lengths; optional angle values encode deflection between original
    geometries.

    Each pair of adjacent segments is one dual edge, one row with one geometry, and its
    ``oneway`` column says which moves it allows: 0 when segment ``v`` can be entered from ``u``
    and ``u`` from ``v``, 1 when only ``u -> v`` is allowed (the row points that way). With
    ``oneway=True`` the moves respect one-way streets (primal ``oneway == 1``), and a pair where
    neither move is allowed has no row; without it every pair is 0. Route on the moves with
    ``dual_graph_fromGDF(..., directed=True)``.
    """
    nodes = nodes_gdf.copy().set_index("nodeID", drop=False)
    nodes.index.name = None

    edges = edges_gdf.copy().set_index("edgeID", drop=False)
    edges.index.name = None

    centroids = edges.copy()
    centroids["centroid"] = centroids.geometry.centroid

    if oneway:
        if "oneway" not in centroids.columns:
            raise ValueError("edges_gdf must contain 'oneway' when oneway=True")
        centroids["intersecting"] = centroids.apply(
            lambda row: _oneway_intersecting_edge_ids(centroids, row),
            axis=1,
        )
    else:
        centroids["intersecting"] = centroids.apply(
            lambda row: _intersecting_edge_ids(centroids, row),
            axis=1,
        )

    nodes_dual_data = centroids.drop(columns=["geometry", "centroid"])
    nodes_dual = gpd.GeoDataFrame(nodes_dual_data, crs=crs, geometry=centroids["centroid"])
    nodes_dual["x"] = [geometry.x for geometry in nodes_dual.geometry]
    nodes_dual["y"] = [geometry.y for geometry in nodes_dual.geometry]
    nodes_dual.index = nodes_dual.edgeID
    nodes_dual.index.name = None

    # Every allowed move between two segments; "intersecting" lists the segments each one leads
    # into, so a one-way pair appears in one direction only.
    moves = {
        (row.Index, intersecting)
        for row in nodes_dual.itertuples()
        for intersecting in row.intersecting
        if intersecting != row.Index
    }

    new_edges: list[dict[str, Any]] = []
    written: set[frozenset[Hashable]] = set()

    for row in nodes_dual.itertuples():
        for intersecting in row.intersecting:
            pair = frozenset((row.Index, intersecting))
            if row.Index == intersecting or pair in written:
                continue
            written.add(pair)

            # Met first from an allowed move, so a one-way row points in its allowed direction.
            intersecting_row = nodes_dual.loc[intersecting]
            distance = (row.length + intersecting_row.length) / 2
            geometry = LineString([row.geometry, intersecting_row.geometry])
            new_edges.append(
                {
                    "u": row.Index,
                    "v": intersecting,
                    "geometry": geometry,
                    "length": distance,
                    "oneway": 0 if (intersecting, row.Index) in moves else 1,
                }
            )

    edges_dual = gpd.GeoDataFrame(
        new_edges,
        columns=["u", "v", "geometry", "length", "oneway"],
        crs=crs,
        geometry="geometry",
    )
    edges_dual["oneway"] = edges_dual["oneway"].astype(int)

    if angle != "radians":
        edges_dual["deg"] = edges_dual.apply(
            lambda row: angle_line_geometries(
                edges.loc[row["u"]].geometry,
                edges.loc[row["v"]].geometry,
                degree=True,
                calculation_type="deflection",
            ),
            axis=1,
        )
    else:
        edges_dual["rad"] = edges_dual.apply(
            lambda row: angle_line_geometries(
                edges.loc[row["u"]].geometry,
                edges.loc[row["v"]].geometry,
                degree=False,
                calculation_type="deflection",
            ),
            axis=1,
        )

    return nodes_dual, edges_dual


def dual_graph_fromGDF(
    nodes_dual: gpd.GeoDataFrame,
    edges_dual: gpd.GeoDataFrame,
    directed: bool = False,
) -> nx.Graph:
    """Create a NetworkX graph from dual-node and dual-edge GeoDataFrames.

    By default an undirected ``networkx.Graph``, one edge per row. ``directed=True`` builds a
    ``networkx.DiGraph`` of the moves each row allows (see ``dual_gdf``): ``u -> v`` always, and
    ``v -> u`` too where ``oneway`` is 0. Rows without a ``oneway`` column are read as two-way.
    Community detection (``identify_regions``) needs the undirected graph.
    """
    nodes = nodes_dual.copy().set_index("edgeID", drop=False)
    nodes.index.name = None
    edges = edges_dual.copy()
    edges["u"] = edges["u"].astype(int)
    edges["v"] = edges["v"].astype(int)

    dual_graph = nx.DiGraph() if directed else nx.Graph()
    dual_graph.add_nodes_from(nodes.index)
    _set_node_attributes_from_gdf(dual_graph, nodes)

    for _, row in edges.iterrows():
        attributes = _edge_attributes(row, {"u", "v"})
        dual_graph.add_edge(row["u"], row["v"], **attributes)
        if directed and row.get("oneway", 0) != 1:
            dual_graph.add_edge(row["v"], row["u"], **attributes)

    return dual_graph


def dual_id_dict(
    dict_values: dict[Any, Any],
    graph: nx.Graph,
    node_attribute: str,
) -> dict[Any, Any]:
    """Map a dual-graph node dictionary to a dictionary keyed by primal edge IDs."""
    return {graph.nodes[node][node_attribute]: value for node, value in dict_values.items()}


def nodes_degree(edges_gdf: gpd.GeoDataFrame) -> dict[Any, int]:
    """Return node degree counts from an edge GeoDataFrame with ``u``/``v`` columns."""
    return edges_gdf[["u", "v"]].stack().value_counts().to_dict()


def from_nx_to_gdf(
    graph: nx.Graph,
    crs: Any,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Convert a NetworkX graph with geometry attributes into node/edge GeoDataFrames.

    This is retained as a small adapter for workflows that already have a
    geometry-bearing NetworkX graph. It does not perform loading or topology
    repair.
    """
    nodes_gdf = gpd.GeoDataFrame(
        [
            {**data, "nodeID": node, "geometry": data["geometry"]}
            for node, data in graph.nodes(data=True)
        ],
        crs=crs,
    )

    edges_gdf = gpd.GeoDataFrame(
        [
            {**data, "u": u, "v": v, "geometry": data["geometry"]}
            for u, v, data in graph.edges(data=True)
        ],
        crs=crs,
    )

    return nodes_gdf, edges_gdf
