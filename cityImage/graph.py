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

from collections import defaultdict
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

    An edge keeps its row's ``u`` and ``v`` as attributes, since an undirected edge does not keep
    its order: with a ``oneway`` column, a one-way street runs ``u -> v``. A two-way street mapped
    as two opposing one-ways between the same nodes keeps only its shorter direction here; for
    one-way routing use ``multiGraph_fromGDF``.
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
        graph.add_edge(row["u"], row["v"], **_edge_attributes(row, set()))

    return graph


def multiGraph_fromGDF(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
) -> nx.MultiGraph:
    """Create an undirected NetworkX MultiGraph from cityImage graph GeoDataFrames.

    Every street is an edge, parallel streets between the same two nodes included, so edge
    measures (``networkx.edge_betweenness_centrality``, ``append_edges_metrics``) give each its
    own value. An edge takes its ``key`` column value as key, or a fresh key when that one is
    taken or absent, and keeps its row's ``u`` and ``v`` as attributes (see ``graph_fromGDF``).
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
            **_edge_attributes(row, {"key"}),
        )

    return multigraph


ONEWAY_YES_VALUES = {"yes", "true", "1"}
ONEWAY_NO_VALUES = {"no", "false", "0"}


def _oneway_flags(values: pd.Series) -> pd.Series:
    """Read a primal ``oneway`` column as 1 (one-way, ``u -> v``) or 0 (two-way).

    Booleans, 1/0 and the strings yes/true/1 and no/false/0 (any case) are read; a missing value
    is two-way. Any other value, such as OSM's ``-1`` (one-way against the drawing direction) or
    ``reversible``, raises a ValueError.
    """

    def flag(value: Any) -> int | None:
        if _is_missing_scalar(value):
            return 0
        text = str(value).strip().lower()
        if text.endswith(".0"):  # 1.0 / 0.0 from a float column
            text = text[:-2]
        if text in ONEWAY_YES_VALUES:
            return 1
        if text in ONEWAY_NO_VALUES:
            return 0
        return None

    flags = values.map(flag)
    unreadable = values[flags.isna()]
    if not unreadable.empty:
        raise ValueError(
            "oneway must be a boolean, 1/0 or yes/no; unreadable values: "
            f"{sorted(map(str, unreadable.unique()))}"
        )
    return flags.astype(int)


def _dual_neighbours(
    edges: gpd.GeoDataFrame, oneway: pd.Series | None = None
) -> list[list[Hashable]]:
    """For each segment, the segments sharing a junction with it that it leads into, in frame order.

    Without ``oneway`` (see ``_oneway_flags``) that is every segment at either end. With it, a
    one-way segment leads only out of its ``v`` end, and a one-way segment is entered only at its
    ``u`` end. Each list includes the segment itself, which the caller skips.
    """
    position = {eid: pos for pos, eid in enumerate(edges.index)}
    starting = defaultdict(list)  # node -> segments whose u it is
    ending = defaultdict(list)  # node -> segments whose v it is
    for eid, u, v in zip(edges.index, edges["u"], edges["v"], strict=True):
        starting[u].append(eid)
        ending[v].append(eid)

    def into(node):
        # Segments that can be entered at node.
        if oneway is None:
            return starting[node] + ending[node]
        return starting[node] + [eid for eid in ending[node] if oneway[eid] == 0]

    neighbours = []
    for eid, u, v in zip(edges.index, edges["u"], edges["v"], strict=True):
        found = into(v)
        if oneway is None or oneway[eid] == 0:
            found = found + into(u)
        neighbours.append(sorted(set(found), key=position.__getitem__))
    return neighbours


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
    ``oneway=True`` the moves respect one-way streets (a primal ``oneway`` that is True, 1 or
    "yes"; False, 0, "no" or missing is two-way, and any other value raises), and a pair where
    neither move is allowed has no row; without it every pair is 0.

    Parallel streets (different segments between the same two junctions) are separate dual
    nodes, each linked to the streets at both junctions. Two of them meet at both ends but are
    one dual edge: going along one and back along the other is a 180° turn at either end.
    """
    edges = edges_gdf.copy().set_index("edgeID", drop=False)
    edges.index.name = None

    centroids = edges.copy()
    centroids["centroid"] = centroids.geometry.centroid

    oneway_flags = None
    if oneway:
        if "oneway" not in centroids.columns:
            raise ValueError("edges_gdf must contain 'oneway' when oneway=True")
        oneway_flags = _oneway_flags(centroids["oneway"])
    centroids["intersecting"] = _dual_neighbours(centroids, oneway_flags)

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
    lengths = nodes_dual["length"].to_dict()
    centres = nodes_dual.geometry.to_dict()

    for row in nodes_dual.itertuples():
        for intersecting in row.intersecting:
            pair = frozenset((row.Index, intersecting))
            if row.Index == intersecting or pair in written:
                continue
            written.add(pair)

            # Met first from an allowed move, so a one-way row points in its allowed direction.
            distance = (lengths[row.Index] + lengths[intersecting]) / 2
            geometry = LineString([centres[row.Index], centres[intersecting]])
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

    geometries = edges.geometry.to_dict()
    degree = angle != "radians"
    edges_dual["deg" if degree else "rad"] = pd.Series(
        [
            angle_line_geometries(
                geometries[u], geometries[v], degree=degree, calculation_type="deflection"
            )
            for u, v in zip(edges_dual["u"], edges_dual["v"], strict=True)
        ],
        index=edges_dual.index,
        dtype=float,
    )

    return nodes_dual, edges_dual


def dual_graph_fromGDF(
    nodes_dual: gpd.GeoDataFrame,
    edges_dual: gpd.GeoDataFrame,
) -> nx.Graph:
    """Create an undirected NetworkX graph from dual-node and dual-edge GeoDataFrames.

    One edge per row. An edge keeps the row's ``u``, ``v`` and ``oneway`` (see ``dual_gdf``) as
    attributes, since an undirected edge does not keep its order: where ``oneway`` is 1, only
    ``u -> v`` is allowed, and following it is left to the modeller.
    """
    nodes = nodes_dual.copy().set_index("edgeID", drop=False)
    nodes.index.name = None
    edges = edges_dual.copy()
    edges["u"] = edges["u"].astype(int)
    edges["v"] = edges["v"].astype(int)

    dual_graph = nx.Graph()
    dual_graph.add_nodes_from(nodes.index)
    _set_node_attributes_from_gdf(dual_graph, nodes)

    for _, row in edges.iterrows():
        dual_graph.add_edge(row["u"], row["v"], **_edge_attributes(row, set()))

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
            # An edge's own u/v (a dual edge's direction) win over the graph's unordered pair.
            {"u": u, "v": v, **data, "geometry": data["geometry"]}
            for u, v, data in graph.edges(data=True)
        ],
        crs=crs,
    )

    return nodes_gdf, edges_gdf
