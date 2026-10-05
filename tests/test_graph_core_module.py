"""Tests for the hard graph core boundary."""

from __future__ import annotations

import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import LineString, Point

import cityImage as ci
from tests.fixtures.cityimage_minimal import CRS, minimal_network


def test_graph_from_gdf_preserves_node_and_edge_attributes_without_mutating_inputs():
    nodes, edges = minimal_network()
    nodes["list_attr"] = [[1], [2], [3], [4]]
    edges["list_edge_attr"] = [["a"], ["b"], ["c"], ["d"]]
    original_node_index = nodes.index.copy()

    graph = ci.graph_fromGDF(nodes, edges)

    assert sorted(graph.nodes()) == [1, 2, 3, 4]
    assert sorted(graph.edges()) == [(1, 2), (1, 4), (2, 3), (3, 4)]
    assert graph.nodes[1]["nodeID"] == 1
    assert "list_attr" not in graph.nodes[1]
    assert graph[1][2]["edgeID"] == 101
    assert graph[1][2]["list_edge_attr"] == ["a"]
    assert nodes.index.equals(original_node_index)


def test_graph_from_gdf_keeps_the_shortest_of_parallel_edges():
    nodes, edges = minimal_network()
    # A second, longer street between nodes 1 and 2, bowing 50 m off the first.
    detour = edges[edges["edgeID"] == 101].copy()
    start, end = detour.geometry.iloc[0].coords[0], detour.geometry.iloc[0].coords[-1]
    middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 50)
    detour["edgeID"] = 999
    detour["geometry"] = [LineString([start, middle, end])]

    for frames in ([edges, detour], [detour, edges]):  # whichever comes first
        graph = ci.graph_fromGDF(nodes, pd.concat(frames))
        assert graph[1][2]["edgeID"] == 101


def test_multigraph_from_gdf_preserves_parallel_edge_keys_when_present():
    nodes, edges = minimal_network()
    edges = edges.iloc[[0, 0]].copy()
    edges["edgeID"] = [1, 2]
    edges["key"] = [0, 1]

    graph = ci.multiGraph_fromGDF(nodes, edges)

    assert isinstance(graph, nx.MultiGraph)
    assert graph.number_of_edges(1, 2) == 2
    assert sorted(graph[1][2].keys()) == [0, 1]


def test_dual_gdf_and_dual_graph_preserve_imageability_semantics():
    nodes, edges = minimal_network()

    nodes_dual, edges_dual = ci.dual_gdf(nodes, edges, CRS, angle="degree")
    dual_graph = ci.dual_graph_fromGDF(nodes_dual, edges_dual)

    assert nodes_dual.sort_values("edgeID")["edgeID"].tolist() == [101, 102, 103, 104]
    assert edges_dual.sort_values(["u", "v"])[["u", "v"]].to_records(index=False).tolist() == [
        (101, 102),
        (101, 104),
        (102, 103),
        (103, 104),
    ]
    assert edges_dual.sort_values(["u", "v"])["deg"].tolist() == pytest.approx(
        [90.0, 90.0, 90.0, 90.0]
    )
    assert sorted(dual_graph.nodes()) == [101, 102, 103, 104]


def test_nodes_degree_and_dual_id_dict_helpers_remain_available():
    nodes, edges = minimal_network()
    nodes_dual, edges_dual = ci.dual_gdf(nodes, edges, CRS)
    graph = ci.dual_graph_fromGDF(nodes_dual, edges_dual)

    assert ci.nodes_degree(edges) == {1: 2, 2: 2, 3: 2, 4: 2}
    assert ci.dual_id_dict({101: 7, 102: 8}, graph, "edgeID") == {101: 7, 102: 8}


def test_from_nx_to_gdf_is_only_a_geometry_bearing_graph_adapter():
    graph = nx.Graph()
    graph.add_node(1, geometry=Point(0, 0))
    graph.add_node(2, geometry=Point(1, 0))
    graph.add_edge(1, 2, edgeID=10, geometry=LineString([(0, 0), (1, 0)]))

    nodes, edges = ci.from_nx_to_gdf(graph, CRS)

    assert nodes.sort_values("nodeID")["nodeID"].tolist() == [1, 2]
    assert edges["edgeID"].tolist() == [10]
    assert nodes.crs == edges.crs


def test_graph_edges_keep_their_u_and_v():
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    import cityImage as ci

    nodes = gpd.GeoDataFrame(
        {"nodeID": [5, 7]}, geometry=[Point(0, 0), Point(10, 0)], crs="EPSG:3857"
    )
    # Drawn from node 7 to node 5, one-way that way.
    edges = gpd.GeoDataFrame(
        {"edgeID": [0], "u": [7], "v": [5], "oneway": [True], "length": [10.0]},
        geometry=[LineString([(10, 0), (0, 0)])],
        crs="EPSG:3857",
    )

    for graph in (ci.graph_fromGDF(nodes, edges), ci.multiGraph_fromGDF(nodes, edges)):
        data = next(iter(graph.edges(data=True)))[2]
        assert (data["u"], data["v"], data["oneway"]) == (7, 5, True)


@pytest.mark.parametrize("angle", [None, "radians"])
def test_dual_gdf_of_a_single_street_has_no_dual_edges(angle):
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    import cityImage as ci

    nodes = gpd.GeoDataFrame(
        {"nodeID": [0, 1]}, geometry=[Point(0, 0), Point(10, 0)], crs="EPSG:3857"
    )
    edges = gpd.GeoDataFrame(
        {"edgeID": [0], "u": [0], "v": [1], "length": [10.0]},
        geometry=[LineString([(0, 0), (10, 0)])],
        crs="EPSG:3857",
    )

    nodes_dual, edges_dual = ci.dual_gdf(nodes, edges, "EPSG:3857", angle=angle)

    assert nodes_dual["edgeID"].tolist() == [0]
    assert edges_dual.empty
    assert {"deg", "rad"} <= set(edges_dual.columns)


@pytest.mark.parametrize("angle", [None, "degree", "radians"])
def test_dual_gdf_writes_the_deflection_in_degrees_and_radians(angle):
    import math

    import geopandas as gpd
    from shapely.geometry import LineString, Point

    import cityImage as ci

    nodes = gpd.GeoDataFrame(
        {"nodeID": [0, 1, 2]}, geometry=[Point(0, 0), Point(10, 0), Point(10, 10)], crs="EPSG:3857"
    )
    edges = gpd.GeoDataFrame(
        {"edgeID": [0, 1], "u": [0, 1], "v": [1, 2], "length": [10.0, 10.0]},
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(10, 0), (10, 10)])],
        crs="EPSG:3857",
    )

    _, edges_dual = ci.dual_gdf(nodes, edges, "EPSG:3857", angle=angle)

    assert edges_dual["deg"].tolist() == pytest.approx([90.0])
    assert edges_dual["rad"].tolist() == pytest.approx([math.pi / 2])
