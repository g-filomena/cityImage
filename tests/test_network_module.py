"""Tests for the hard network adapter split."""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString

import cityImage as ci


def test_network_from_lines_builds_cityimage_nodes_and_edges():
    lines = gpd.GeoDataFrame(
        {"road_name": ["a", "b"]},
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])],
        crs="EPSG:3857",
    )

    nodes, edges = ci.network_from_lines(
        lines,
        "EPSG:3857",
        dict_columns={"name": "road_name"},
    )

    assert nodes.sort_values("nodeID")["nodeID"].tolist() == [0, 1, 2]
    assert edges.sort_values("edgeID")[["edgeID", "u", "v", "name"]].to_records(index=False).tolist() == [
        (0, 0, 1, "a"),
        (1, 1, 2, "b"),
    ]
    assert edges.sort_values("edgeID")["length"].tolist() == [1.0, 1.0]


def test_reset_index_graph_gdfs_preserves_edge_node_relationships():
    nodes = gpd.GeoDataFrame(
        {"nodeID": [10, 20], "x": [0.0, 1.0], "y": [0.0, 0.0]},
        geometry=[LineString([(0, 0), (0, 0)]).centroid, LineString([(1, 0), (1, 0)]).centroid],
        crs="EPSG:3857",
    )
    edges = gpd.GeoDataFrame(
        {"edgeID": [99], "u": [10], "v": [20], "length": [1.0]},
        geometry=[LineString([(0, 0), (1, 0)])],
        crs="EPSG:3857",
    )

    nodes_out, edges_out = ci.reset_index_graph_gdfs(nodes, edges)

    assert nodes_out["nodeID"].tolist() == [0, 1]
    assert edges_out[["edgeID", "u", "v"]].to_records(index=False).tolist() == [(0, 0, 1)]
