"""Network GeoDataFrame adapters for cityImage.

This module keeps only the non-OSM, schema-level network construction that is
specific enough to be useful inside cityImage. Live OSM acquisition is delegated
to OSMnx; file IO is delegated to GeoPandas. Use this module when you already
have a line GeoDataFrame and need cityImage-style ``nodes_gdf``/``edges_gdf``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from .geometry import fix_multiparts_LineString_gdf


def _resolve_list_edges_gdf(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Resolve list-valued edge attributes commonly returned by OSMnx."""
    edges_gdf = edges_gdf.copy()

    for column in ["highway", "name", "oneway"]:
        if column in edges_gdf.columns:
            edges_gdf[column] = [x[0] if isinstance(x, list) else x for x in edges_gdf[column]]

    for column in ["lanes", "bridge", "tunnel"]:
        if column in edges_gdf.columns:
            edges_gdf[column] = [max(x) if isinstance(x, list) else x for x in edges_gdf[column]]
            if column in {"bridge", "tunnel"}:
                edges_gdf[column] = edges_gdf[column].apply(
                    lambda x: 0 if pd.isna(x) or x is False else 1
                )

    return edges_gdf


def network_from_lines(
    edges_gdf: gpd.GeoDataFrame,
    crs: Any,
    *,
    dict_columns: Mapping[str, str | None] | None = None,
    other_columns: Sequence[str] | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Build cityImage nodes/edges from a line GeoDataFrame.

    This replaces the old ``get_network_fromGDF`` route without owning file IO
    or OSM download. Load files with ``geopandas.read_file`` and download OSM
    networks with OSMnx, then pass the resulting line GeoDataFrame here when
    endpoint-derived node construction is actually needed.
    """
    if other_columns is None:
        other_columns = []
    if dict_columns is None:
        dict_columns = {}

    if not isinstance(edges_gdf, gpd.GeoDataFrame):
        raise TypeError("edges_gdf must be a GeoDataFrame")
    if edges_gdf.empty:
        empty_edges = gpd.GeoDataFrame(columns=["edgeID", "u", "v", "length"], geometry=[], crs=crs)
        empty_nodes = gpd.GeoDataFrame(columns=["nodeID", "x", "y"], geometry=[], crs=crs)
        return empty_nodes, empty_edges

    edges = edges_gdf.to_crs(crs).copy()
    edges["key"] = 0

    new_columns: list[str] = []
    for key, value in dict_columns.items():
        if value is not None:
            if value not in edges.columns:
                raise ValueError(f"dict_columns maps {key!r} to missing column {value!r}")
            edges[key] = edges[value]
            new_columns.append(key)

    missing_other = [column for column in other_columns if column not in edges.columns]
    if missing_other:
        raise ValueError(f"other_columns contains missing columns: {missing_other}")

    edges = edges[["geometry", "key", *new_columns, *other_columns]].copy()
    edges = fix_multiparts_LineString_gdf(edges)
    edges = edges.reset_index(drop=True)
    edges["edgeID"] = edges.index.to_numpy(dtype="int64")

    nodes = obtain_nodes_gdf(edges, crs)
    nodes["nodeID"] = nodes.index.to_numpy(dtype="int64")
    nodes, edges = join_nodes_edges_by_coordinates(nodes, edges)
    edges["length"] = edges.geometry.length

    if "z" not in nodes.columns:
        nodes["z"] = 2.0

    nodes = nodes[nodes.nodeID.isin(np.unique(edges[["u", "v"]].values))].copy()
    return nodes, edges


def obtain_nodes_gdf(edges_gdf: gpd.GeoDataFrame, crs: Any) -> gpd.GeoDataFrame:
    """Create a node GeoDataFrame from unique line start/end coordinates."""
    if edges_gdf.empty:
        return gpd.GeoDataFrame(columns=["nodeID", "x", "y"], geometry=[], crs=crs)

    unique_nodes = pd.concat(
        [
            edges_gdf.geometry.apply(lambda row: row.coords[0]),
            edges_gdf.geometry.apply(lambda row: row.coords[-1]),
        ]
    ).unique()

    if len(edges_gdf.geometry.iloc[0].coords[0]) > 2:
        nodes_data = pd.DataFrame(list(unique_nodes), columns=["x", "y", "z"]).astype("float")
    else:
        nodes_data = pd.DataFrame(list(unique_nodes), columns=["x", "y"]).astype("float")

    geometry = [Point(xy) for xy in zip(nodes_data.x, nodes_data.y, strict=False)]
    nodes_gdf = gpd.GeoDataFrame(nodes_data, crs=crs, geometry=geometry)
    nodes_gdf = nodes_gdf.reset_index(drop=True)
    nodes_gdf["nodeID"] = nodes_gdf.index.to_numpy(dtype="int64")
    return nodes_gdf


def join_nodes_edges_by_coordinates(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Assign edge ``u``/``v`` IDs by matching endpoint coordinates to nodes."""
    nodes = nodes_gdf.copy()
    edges = edges_gdf.copy()

    if "nodeID" not in nodes.columns:
        nodes["nodeID"] = nodes.index.to_numpy(dtype="int64")

    nodes["coordinates"] = list(zip(nodes.x, nodes.y, strict=False))
    node_lookup = nodes.set_index("coordinates").nodeID
    edges["u"] = edges.geometry.apply(lambda row: row.coords[0]).map(node_lookup)
    edges["v"] = edges.geometry.apply(lambda row: row.coords[-1]).map(node_lookup)
    nodes = nodes.drop(columns="coordinates")
    return nodes, edges


def reset_index_graph_gdfs(
    nodes_gdf: gpd.GeoDataFrame,
    edges_gdf: gpd.GeoDataFrame,
    nodeID: str = "nodeID",
    edgeID: str = "edgeID",
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Reset node/edge IDs while preserving the edge-node relationship."""
    nodes = nodes_gdf.copy()
    edges = edges_gdf.copy()

    edges["u"], edges["v"] = edges["u"].astype("int64"), edges["v"].astype("int64")
    edges = edges.rename(columns={"u": "old_u", "v": "old_v"})
    nodes["old_nodeID"] = nodes[nodeID].to_numpy(dtype="int64")
    nodes = nodes.reset_index(drop=True)
    nodes[nodeID] = nodes.index.to_numpy(dtype="int64")

    edges = pd.merge(
        edges,
        nodes[["old_nodeID", nodeID]],
        how="left",
        left_on="old_u",
        right_on="old_nodeID",
    ).rename(columns={nodeID: "u"})
    edges = pd.merge(
        edges,
        nodes[["old_nodeID", nodeID]],
        how="left",
        left_on="old_v",
        right_on="old_nodeID",
    ).rename(columns={nodeID: "v"})

    edges = edges.drop(columns=["old_u", "old_nodeID_x", "old_nodeID_y", "old_v"])
    nodes = nodes.drop(columns=["old_nodeID", "index"], errors="ignore")
    edges = edges.reset_index(drop=True)
    edges[edgeID] = edges.index.to_numpy(dtype="int64")
    return nodes, edges
