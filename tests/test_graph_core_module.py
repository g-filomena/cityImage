"""Tests for the hard graph core boundary."""

from __future__ import annotations

import networkx as nx
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
