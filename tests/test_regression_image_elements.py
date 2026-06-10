"""Behaviour-lock tests for Lynchian image elements.

This file locks angle/dual-graph/barrier/district semantics before we delegate
more infrastructure work to external libraries.
"""

from __future__ import annotations

import pytest
from shapely.geometry import LineString

import cityImage as ci
from tests.fixtures.cityimage_minimal import CRS, minimal_network, structuring_barriers


def test_angle_semantics_for_turns_and_straight_continuation():
    first = LineString([(0, 0), (10, 0)])
    right_turn = LineString([(10, 0), (10, 10)])
    straight = LineString([(10, 0), (20, 0)])

    assert ci.angle_line_geometries(
        first, right_turn, degree=True, calculation_type="vectors"
    ) == pytest.approx(90.0)
    assert ci.angle_line_geometries(
        first, right_turn, degree=True, calculation_type="deflection"
    ) == pytest.approx(90.0)
    assert ci.angle_line_geometries(
        first, right_turn, degree=True, calculation_type="angular_change"
    ) == pytest.approx(90.0)

    assert ci.angle_line_geometries(
        first, straight, degree=True, calculation_type="vectors"
    ) == pytest.approx(180.0)
    assert ci.angle_line_geometries(
        first, straight, degree=True, calculation_type="deflection"
    ) == pytest.approx(0.0)
    assert ci.angle_line_geometries(
        first, straight, degree=True, calculation_type="angular_change"
    ) == pytest.approx(0.0)

    assert ci.get_coord_angle((0, 0), 10, 90) == pytest.approx((10.0, 0.0))
    assert ci.get_coord_angle((0, 0), 10, 0) == pytest.approx((0.0, 10.0))


def test_dual_graph_semantics_preserve_edge_ids_and_deflection_angles():
    nodes_gdf, edges_gdf = minimal_network()

    graph = ci.graph_fromGDF(nodes_gdf.copy(), edges_gdf.copy())
    assert sorted(graph.nodes()) == [1, 2, 3, 4]
    assert sorted(graph.edges()) == [(1, 2), (1, 4), (2, 3), (3, 4)]
    assert {node: graph.degree[node] for node in graph.nodes()} == {1: 2, 2: 2, 3: 2, 4: 2}

    nodes_dual, edges_dual = ci.dual_gdf(nodes_gdf.copy(), edges_gdf.copy(), CRS, angle="degree")

    assert nodes_dual.sort_values("edgeID")["edgeID"].tolist() == [101, 102, 103, 104]
    assert edges_dual.sort_values(["u", "v"])[["u", "v"]].to_records(index=False).tolist() == [
        (101, 102),
        (101, 104),
        (102, 103),
        (103, 104),
    ]
    assert edges_dual.sort_values(["u", "v"])["length"].tolist() == pytest.approx(
        [10.0, 10.0, 10.0, 10.0]
    )
    assert edges_dual.sort_values(["u", "v"])["deg"].tolist() == pytest.approx(
        [90.0, 90.0, 90.0, 90.0]
    )

    dual_graph = ci.dual_graph_fromGDF(nodes_dual.copy(), edges_dual.copy())
    assert sorted(dual_graph.nodes()) == [101, 102, 103, 104]
    assert sorted(dual_graph.edges()) == [(101, 102), (101, 104), (102, 103), (103, 104)]
    assert {node: dual_graph.nodes[node]["edgeID"] for node in dual_graph.nodes()} == {
        101: 101,
        102: 102,
        103: 103,
        104: 104,
    }


def test_structuring_barrier_assignment_preserves_boolean_crossing_semantics():
    _, edges_gdf = minimal_network()
    assigned = ci.assign_structuring_barriers(edges_gdf.copy(), structuring_barriers())

    assert "sep_barr" in assigned.columns
    assert assigned.sort_values("edgeID")["sep_barr"].tolist() == [True, True, True, True]


def test_district_and_gateway_semantics_are_preserved_when_regions_extra_is_available():
    pytest.importorskip("community")

    nodes_gdf, edges_gdf = minimal_network()
    edges_with_districts = edges_gdf.copy()
    edges_with_districts["district"] = [1, 1, 2, 2]

    nodes_with_districts = ci.district_to_nodes_from_edges(
        nodes_gdf.copy(),
        edges_with_districts.copy(),
        "district",
    )
    assert nodes_with_districts.sort_values("nodeID")["district"].tolist() == [1, 1, 1, 2]

    nodes_indexed = nodes_with_districts.copy().set_index("nodeID", drop=False)
    edges_indexed = edges_gdf.copy().set_index("edgeID", drop=False)
    edges_from_nodes = ci.districts_to_edges_from_nodes(
        nodes_indexed.copy(),
        edges_indexed.copy(),
        "district",
    )
    edges_from_nodes = edges_from_nodes.reset_index(drop=True).sort_values("edgeID")

    assert edges_from_nodes["district_uv"].tolist() == [1, 1, 999999, 999999]
    assert edges_from_nodes["district_u"].tolist() == [1, 1, 1, 2]
    assert edges_from_nodes["district_v"].tolist() == [1, 1, 2, 1]

    gateways = ci.find_gateways(nodes_indexed.copy(), edges_indexed.copy(), "district")
    gateways = gateways.reset_index(drop=True).sort_values("nodeID")
    assert gateways["gateway"].tolist() == [1, 0, 1, 1]
