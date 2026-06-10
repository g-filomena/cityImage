"""Behaviour-lock tests for centrality semantics used downstream by scoring."""

from __future__ import annotations

import pytest

import cityImage as ci
from tests.fixtures.cityimage_minimal import path_network_with_opportunities


def test_weight_nodes_assigns_local_opportunity_counts_when_centrality_extra_is_available():
    pytest.importorskip("igraph")

    nodes_gdf, edges_gdf = path_network_with_opportunities()
    graph = ci.graph_fromGDF(nodes_gdf.copy(), edges_gdf.copy())
    nodes_indexed = nodes_gdf.copy().set_index("nodeID", drop=False)
    services = nodes_gdf.copy()

    weighted = ci.weight_nodes(nodes_indexed, services, graph, field_name="n_services", radius=0.25)

    assert {node: weighted.nodes[node]["n_services"] for node in weighted.nodes()} == {
        1: 1,
        2: 1,
        3: 1,
    }


def test_reach_centrality_preserves_node_id_mapping_when_centrality_extra_is_available():
    pytest.importorskip("igraph")

    nodes_gdf, edges_gdf = path_network_with_opportunities()
    graph = ci.graph_fromGDF(nodes_gdf.copy(), edges_gdf.copy())

    reach = ci.calculate_centrality(
        graph,
        measure="reach",
        weight="weight",
        radius=1.0,
        attribute="opportunity",
    )

    assert reach == {1: 20.0, 2: 40.0, 3: 20.0}


def test_straightness_centrality_preserves_path_symmetry_when_centrality_extra_is_available():
    pytest.importorskip("igraph")

    nodes_gdf, edges_gdf = path_network_with_opportunities()
    graph = ci.graph_fromGDF(nodes_gdf.copy(), edges_gdf.copy())

    straightness = ci.calculate_centrality(
        graph,
        measure="straightness",
        weight="weight",
        normalized=True,
    )

    assert straightness[1] == pytest.approx(1.0)
    assert straightness[2] == pytest.approx(1.0)
    assert straightness[3] == pytest.approx(1.0)
