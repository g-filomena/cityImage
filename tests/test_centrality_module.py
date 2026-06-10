"""Tests for centrality delegation."""

from __future__ import annotations

import networkx as nx
import pytest

import cityImage as ci


def _path_graph():
    graph = nx.Graph()
    graph.add_node(1, x=0.0, y=0.0, opportunity=10.0)
    graph.add_node(2, x=1.0, y=0.0, opportunity=20.0)
    graph.add_node(3, x=2.0, y=0.0, opportunity=30.0)
    graph.add_edge(1, 2, weight=1.0, edgeID=101)
    graph.add_edge(2, 3, weight=1.0, edgeID=102)
    return graph


def test_standard_centrality_can_delegate_to_networkx_without_igraph():
    graph = _path_graph()

    betweenness = ci.calculate_centrality(
        graph,
        measure="betweenness",
        weight="weight",
        normalized=True,
        backend="networkx",
    )
    closeness = ci.calculate_centrality(
        graph,
        measure="closeness",
        weight="weight",
        backend="networkx",
    )

    assert betweenness == {1: 0.0, 2: 1.0, 3: 0.0}
    assert closeness[2] > closeness[1]
    assert closeness[2] > closeness[3]


def test_igraph_backend_preserves_reach_semantics_when_extra_is_available():
    pytest.importorskip("igraph")

    graph = _path_graph()
    reach = ci.calculate_centrality(
        graph,
        measure="reach",
        weight="weight",
        radius=1.0,
        attribute="opportunity",
    )

    assert reach == {1: 20.0, 2: 40.0, 3: 20.0}


def test_igraph_backend_preserves_straightness_semantics_when_extra_is_available():
    pytest.importorskip("igraph")

    graph = _path_graph()
    straightness = ci.calculate_centrality(
        graph,
        measure="straightness",
        weight="weight",
        normalized=True,
    )

    assert straightness == {
        1: pytest.approx(1.0),
        2: pytest.approx(1.0),
        3: pytest.approx(1.0),
    }


def test_invalid_backend_raises_clear_error():
    graph = _path_graph()

    with pytest.raises(ValueError, match="backend"):
        ci.calculate_centrality(graph, backend="not-a-backend")
