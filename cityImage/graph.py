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
    nodeID_column: str = "nodeID",
) -> nx.Graph:
    """Create an undirected NetworkX graph from cityImage node/edge GeoDataFrames."""
    nodes = nodes_gdf.copy()
    edges = edges_gdf.copy()

    nodes = nodes.set_index(nodeID_column, drop=False)
    nodes.index.name = None

    graph = nx.Graph()
    graph.add_nodes_from(nodes.index)
    _set_node_attributes_from_gdf(graph, nodes)

    for _, row in edges.iterrows():
        graph.add_edge(row["u"], row["v"], **_edge_attributes(row, {"u", "v"}))

    return graph


def multiGraph_fromGDF(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    nodeID_column: str = "nodeID",
) -> nx.MultiGraph:
    """Create an undirected NetworkX MultiGraph from cityImage graph GeoDataFrames.

    This function is retained for legacy workflows with parallel edges. New code
    should usually prefer ``graph_fromGDF`` unless edge multiplicity matters.
    """
    nodes = nodes_gdf.copy()
    edges = edges_gdf.copy()

    nodes = nodes.set_index(nodeID_column, drop=False)
    nodes.index.name = None

    multigraph = nx.MultiGraph()
    multigraph.add_nodes_from(nodes.index)
    _set_node_attributes_from_gdf(multigraph, nodes)

    for _, row in edges.iterrows():
        key = row["key"] if "key" in row.index else 0
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

    new_edges: list[dict[str, Any]] = []
    processed: set[tuple[Hashable, Hashable]] = set()

    for row in nodes_dual.itertuples():
        for intersecting in row.intersecting:
            if (
                row.Index == intersecting
                or (row.Index, intersecting) in processed
                or (intersecting, row.Index) in processed
            ):
                continue

            intersecting_row = nodes_dual.loc[intersecting]
            distance = (row.length + intersecting_row.length) / 2
            geometry = LineString([row.geometry, intersecting_row.geometry])
            new_edges.append(
                {
                    "u": row.Index,
                    "v": intersecting,
                    "geometry": geometry,
                    "length": distance,
                }
            )
            processed.add((row.Index, intersecting))

    edges_dual = gpd.GeoDataFrame(
        new_edges,
        columns=["u", "v", "geometry", "length"],
        crs=crs,
        geometry="geometry",
    )

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
) -> nx.Graph:
    """Create a NetworkX graph from dual-node and dual-edge GeoDataFrames."""
    nodes = nodes_dual.copy().set_index("edgeID", drop=False)
    nodes.index.name = None
    edges = edges_dual.copy()
    edges["u"] = edges["u"].astype(int)
    edges["v"] = edges["v"].astype(int)

    dual_graph = nx.Graph()
    dual_graph.add_nodes_from(nodes.index)
    _set_node_attributes_from_gdf(dual_graph, nodes)

    for _, row in edges.iterrows():
        dual_graph.add_edge(row["u"], row["v"], **_edge_attributes(row, {"u", "v"}))

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
