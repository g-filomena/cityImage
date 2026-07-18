"""Tests for delegated region/district semantics."""

from __future__ import annotations

import networkx as nx
import pytest

import cityImage as ci
from tests.fixtures.cityimage_minimal import CRS, minimal_network, york_network


def test_regions_from_dual_partition_maps_dual_partition_to_primal_edges():
    _, edges_gdf = minimal_network()

    dual_graph = nx.Graph()
    dual_graph.add_node("a", edgeID=101)
    dual_graph.add_node("b", edgeID=102)
    dual_graph.add_node("c", edgeID=103)
    dual_graph.add_node("d", edgeID=104)
    partition = {"a": 7, "b": 7, "c": 8, "d": 8}

    regions = ci.regions_from_dual_partition(partition, dual_graph, edges_gdf, column="district")

    assert regions.sort_values("edgeID")["district"].tolist() == [7, 7, 8, 8]


def test_regions_from_primal_partition_maps_partition_to_nodes():
    nodes_gdf, _ = minimal_network()
    partition = {1: 10, 2: 10, 3: 11, 4: 11}

    regions = ci.regions_from_primal_partition(partition, nodes_gdf, column="district")

    assert regions.sort_values("nodeID")["district"].tolist() == [10, 10, 11, 11]


def test_polygonise_partitions_preserves_polygonising_buffer_semantics():
    _, edges_gdf = minimal_network()
    edges_gdf = edges_gdf.copy()
    edges_gdf["district"] = [1, 1, 2, 2]

    polygons = ci.polygonise_partitions(edges_gdf, "district", convex_hull=True, buffer=1)

    assert sorted(polygons["district"].tolist()) == [1, 2]
    assert all(geom.geom_type in {"Polygon", "MultiPolygon"} for geom in polygons.geometry)
    assert all(geom.area > 0 for geom in polygons.geometry)


def test_district_gateway_semantics_remain_stable():
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


def test_identify_regions_convenience_delegates_to_python_louvain_when_available():
    pytest.importorskip("community")
    nodes_gdf, edges_gdf = minimal_network()
    nodes_dual, edges_dual = ci.dual_gdf(nodes_gdf.copy(), edges_gdf.copy(), CRS, angle="degree")
    dual_graph = ci.dual_graph_fromGDF(nodes_dual.copy(), edges_dual.copy())

    regions = ci.identify_regions(dual_graph, edges_gdf.copy())

    assert "p_topo" in regions.columns
    assert len(regions) == len(edges_gdf)


def test_amend_nodes_membership_reassigns_small_district_and_survives_connectivity_check():
    # Regression: amend_nodes_membership must run the connectivity check (which builds a graph via
    # _graph_from_gdfs -> graph_fromGDF) for a large-enough district and reassign a too-small one to
    # a neighbouring district. Uses the real York subset (non-contiguous nodeID on a plain
    # RangeIndex), which is where the .loc-by-nodeID lookups used to KeyError.
    nodes_gdf, edges_gdf = york_network()
    nodes_gdf["district"] = 1
    lone = int(nodes_gdf["nodeID"].iloc[0])
    nodes_gdf.loc[nodes_gdf["nodeID"] == lone, "district"] = 2  # a single too-small district

    out = ci.amend_nodes_membership(nodes_gdf, edges_gdf, "district", min_size_district=2)

    # The lone district is absorbed and the connectivity pass keeps everything in one district.
    assert set(out["district"]) == {1}
    assert len(out) == len(nodes_gdf)
