"""Tests for delegated region/district semantics."""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import pytest
from shapely.geometry import LineString, Point

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


def test_identify_regions_resolution_sets_the_scale_of_the_regions():
    nodes, edges = york_network()
    nodes_dual, edges_dual = ci.dual_gdf(nodes, edges, edges.crs)
    dual = ci.dual_graph_fromGDF(nodes_dual, edges_dual)

    def regions(resolution):
        partition = ci.identify_regions(dual, edges, weight="rad", resolution=resolution)
        return partition["p_rad"].nunique()

    assert regions(0.3) < regions(1.0) < regions(3.0)


def _district_with_bridge_and_strip():
    """Two 100 m blocks; a 300 m street out of them (a bridge); a street leading to a 6 m wide
    strip between two carriageways."""
    lines = [
        LineString([(0, 0), (200, 0), (200, 100), (0, 100), (0, 0)]),
        LineString([(100, 0), (100, 100)]),
        LineString([(200, 50), (500, 50)]),
        LineString([(100, 100), (100, 400)]),
        LineString([(0, 400), (200, 400), (200, 406), (0, 406), (0, 400)]),
    ]
    return gpd.GeoDataFrame({"district": [1] * len(lines)}, geometry=lines, crs=CRS)


def test_polygonise_partitions_blocks_leave_out_streets_that_close_no_block():
    edges = _district_with_bridge_and_strip()

    district = ci.polygonise_partitions(
        edges, "district", convex_hull=False, blocks=True
    ).geometry.iloc[0]

    assert district.contains(Point(50, 50)) and district.contains(Point(150, 50))
    assert not district.contains(Point(450, 50))  # the bridge
    assert not district.contains(Point(100, 403))  # the strip between the carriageways


def test_polygonise_partitions_convex_hull_reaches_the_end_of_every_street():
    edges = _district_with_bridge_and_strip()

    district = ci.polygonise_partitions(edges, "district").geometry.iloc[0]

    assert district.contains(Point(450, 50)) and district.contains(Point(100, 403))


def test_polygonise_partitions_blocks_keep_the_streets_of_a_partition_without_blocks():
    edges = gpd.GeoDataFrame({"district": [1]}, geometry=[LineString([(0, 0), (300, 0)])], crs=CRS)

    district = ci.polygonise_partitions(
        edges, "district", convex_hull=False, blocks=True
    ).geometry.iloc[0]

    assert district.contains(Point(150, 0)) and district.area > 0


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


def test_identify_regions_gives_every_edge_a_region():
    nodes_gdf, edges_gdf = minimal_network()
    nodes_dual, edges_dual = ci.dual_gdf(nodes_gdf.copy(), edges_gdf.copy(), CRS, angle="degree")
    dual_graph = ci.dual_graph_fromGDF(nodes_dual.copy(), edges_dual.copy())

    regions = ci.identify_regions(dual_graph, edges_gdf.copy())

    assert "p_topo" in regions.columns
    assert len(regions) == len(edges_gdf)


def test_identify_regions_is_reproducible_by_default():
    nodes_gdf, edges_gdf = york_network()
    graph = ci.graph_fromGDF(nodes_gdf.copy(), edges_gdf.copy())

    first = ci.identify_regions_primal(graph, nodes_gdf.copy(), weight="length")
    second = ci.identify_regions_primal(graph, nodes_gdf.copy(), weight="length")

    assert first["p_length"].tolist() == second["p_length"].tolist()


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


def _york_dual_graph():
    nodes_gdf, edges_gdf = york_network()
    nodes_dual, edges_dual = ci.dual_gdf(nodes_gdf, edges_gdf, "EPSG:2019")
    return ci.dual_graph_fromGDF(nodes_dual, edges_dual), edges_gdf


def test_identify_regions_weights_by_the_angle_the_dual_graph_carries():
    dual_graph, edges_gdf = _york_dual_graph()

    angular = ci.identify_regions(dual_graph, edges_gdf.copy(), weight="rad")
    topological = ci.identify_regions(dual_graph, edges_gdf.copy())

    # The weight is read: an angular partition is not the topological one under another name.
    assert angular["p_rad"].tolist() != topological["p_topo"].tolist()


@pytest.mark.parametrize("weight", ["radians", "angle"])
def test_identify_regions_rejects_a_weight_no_edge_has(weight):
    dual_graph, edges_gdf = _york_dual_graph()

    with pytest.raises(ValueError, match="no edge of the graph has"):
        ci.identify_regions(dual_graph, edges_gdf.copy(), weight=weight)


def test_identify_regions_primal_rejects_a_weight_no_edge_has():
    nodes_gdf, edges_gdf = york_network()
    graph = ci.graph_fromGDF(nodes_gdf, edges_gdf)

    with pytest.raises(ValueError, match="'rad'"):
        ci.identify_regions_primal(graph, nodes_gdf, weight="rad")
