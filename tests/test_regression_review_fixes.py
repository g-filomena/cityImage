"""Behaviours that are easy to get wrong: districts, visibility, networks, centrality, barriers,
buildings and heights.

Each test pins one behaviour with the smallest realistic input.
"""

from __future__ import annotations

import logging
import math
import os

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon, box

import cityImage as ci
from cityImage import barriers, centrality, height, regions, visibility2d

CRS = "EPSG:27700"


def _grid_lines(n, spacing=100.0, x0=0.0):
    lines = []
    for i in range(n):
        for j in range(n):
            if i < n - 1:
                lines.append(
                    LineString(
                        [(x0 + i * spacing, j * spacing), (x0 + (i + 1) * spacing, j * spacing)]
                    )
                )
            if j < n - 1:
                lines.append(
                    LineString(
                        [(x0 + i * spacing, j * spacing), (x0 + i * spacing, (j + 1) * spacing)]
                    )
                )
    return gpd.GeoDataFrame(geometry=lines, crs=CRS)


def _scored_buildings(heights):
    data = {
        "buildingID": [0, 1, 2],
        "area": [1.0, 2.0, 3.0],
        "fac": [1.0, 0.0, 3.0],
        "3dvis": [0.0, 1.0, 0.5],
        "neigh": [1, 2, 3],
        "road": [1.0, 2.0, 3.0],
        "2dvis": [1.0, 2.0, 3.0],
        "cult": [0.0, 1.0, 0.0],
        "prag": [0.1, 0.2, 0.3],
    }
    if heights is not None:
        data["height"] = heights
    return gpd.GeoDataFrame(
        data, geometry=[box(i * 20, 0, i * 20 + 10, 10) for i in range(3)], crs=CRS
    )


# --- Districts ------------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_amend_nodes_membership_refuses_a_network_with_islands():
    lines = pd.concat([_grid_lines(6), _grid_lines(2, x0=5000)], ignore_index=True)
    nodes, edges = ci.network_from_lines(gpd.GeoDataFrame(lines, crs=CRS), CRS)
    nodes["d"] = np.where(nodes.geometry.x < 4000, 0, 1)

    with pytest.raises(ValueError, match="not connected"):
        regions.amend_nodes_membership(nodes, edges, "d", min_size_district=10)


@pytest.mark.timeout(30)
def test_amend_nodes_membership_refuses_a_network_smaller_than_min_size():
    nodes, edges = ci.network_from_lines(_grid_lines(2), CRS)
    nodes["d"] = 0

    with pytest.raises(ValueError, match="No district reaches"):
        regions.amend_nodes_membership(nodes, edges, "d", min_size_district=10)


@pytest.mark.timeout(30)
def test_amend_nodes_membership_refuses_when_no_district_is_large_enough():
    nodes, edges = ci.network_from_lines(_grid_lines(5), CRS)  # 25 nodes
    nodes["d"] = (nodes.geometry.x // 100).astype(int)  # 5 districts of 5 nodes

    with pytest.raises(ValueError, match="No district reaches"):
        regions.amend_nodes_membership(nodes, edges, "d", min_size_district=10)


@pytest.mark.timeout(30)
def test_amend_nodes_membership_amends_a_small_district():
    nodes, edges = ci.network_from_lines(_grid_lines(6), CRS)  # 36 nodes
    nodes["d"] = 0
    corner = (nodes.geometry.x == 0) & (nodes.geometry.y == 0)
    nodes.loc[corner, "d"] = 1  # a one-node district

    out = regions.amend_nodes_membership(nodes, edges, "d", min_size_district=10)

    assert set(out["d"]) == {0}


@pytest.mark.timeout(30)
def test_amend_nodes_membership_stops_when_nodes_trade_districts(monkeypatch):
    nodes, edges = ci.network_from_lines(_grid_lines(6), CRS)  # 36 nodes
    first, second = nodes["nodeID"].iloc[0], nodes["nodeID"].iloc[1]
    nodes["d"] = np.where(nodes["nodeID"] == first, regions.INVALID_DISTRICT, 0)

    def trade(node_id, nodes_gdf, edges_gdf, column):
        # The two nodes swap between a valid district and none, out of phase, on every pass.
        value = nodes_gdf.loc[node_id, column]
        if node_id in (first, second):
            return 0 if value == regions.INVALID_DISTRICT else regions.INVALID_DISTRICT
        return value

    monkeypatch.setattr(regions, "_amend_node_membership", trade)
    monkeypatch.setattr(regions, "_check_disconnected_districts", lambda nodes, *args: nodes)

    with pytest.raises(ValueError, match="did not settle within 36 passes"):
        regions.amend_nodes_membership(nodes, edges, "d", min_size_district=10)


def test_district_to_nodes_from_edges_looks_beyond_100_m():
    nodes = gpd.GeoDataFrame({"nodeID": [0]}, geometry=[Point(0, 500)], crs=CRS)
    edges = gpd.GeoDataFrame(
        {"edgeID": [0], "u": [1], "v": [2], "p": [3]},
        geometry=[LineString([(0, 0), (10, 0)])],
        crs=CRS,
    )

    assert regions.district_to_nodes_from_edges(nodes, edges, "p")["p"].tolist() == [3]


def test_districts_to_edges_from_nodes_looks_nodes_up_by_node_id():
    # Index 0..2, nodeIDs 0, 2, 3: index label 2 is node 3, in district 7.
    nodes = gpd.GeoDataFrame(
        {"nodeID": [0, 2, 3], "d": [1, 1, 7]},
        geometry=[Point(0, 0), Point(10, 0), Point(20, 0)],
        crs=CRS,
    )
    edges = gpd.GeoDataFrame(
        {"edgeID": [0], "u": [0], "v": [2]}, geometry=[LineString([(0, 0), (10, 0)])], crs=CRS
    )

    out = regions.districts_to_edges_from_nodes(nodes, edges, "d")

    assert (out["d_u"].iloc[0], out["d_v"].iloc[0], out["d_uv"].iloc[0]) == (1, 1, 1)


# --- Visibility -----------------------------------------------------------------------------


def test_visibility_polygon2d_covers_the_whole_ring():
    building = box(0, 0, 10, 10)
    obstructions = gpd.GeoDataFrame({"buildingID": [0]}, geometry=[building], crs=CRS)

    area = visibility2d.visibility_polygon2d(building, obstructions, None, 100)

    radius = 100 + Point(5, 5).distance(building.envelope.exterior)
    tips = [ci.get_coord_angle((5, 5), radius, angle) for angle in range(0, 360, 10)]
    assert area == pytest.approx(Polygon(tips).difference(building).area)


def test_compute_3d_sight_lines_leaves_the_caller_frame_and_no_files(tmp_path):
    pytest.importorskip("dask")
    pytest.importorskip("psutil")
    nodes = gpd.GeoDataFrame(
        {"nodeID": [0, 1], "x": [0.0, 0.0], "y": [-400.0, -500.0]},
        geometry=[Point(0, -400), Point(0, -500)],
        crs=CRS,
    )
    buildings = gpd.GeoDataFrame(
        {"buildingID": [0, 1], "height": [20.0, 30.0], "base": [0.3, 0.5]},
        geometry=[box(-10, 0, 10, 20), box(40, 0, 60, 20)],
        crs=CRS,
    )
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        sight_lines = ci.compute_3d_sight_lines(
            nodes, buildings, buildings, None, "t", num_workers=1
        )
    finally:
        os.chdir(cwd)

    assert len(sight_lines) > 0
    assert buildings["base"].tolist() == [0.3, 0.5]
    assert list(tmp_path.iterdir()) == []

    chunks = tmp_path / "chunks"
    chunks.mkdir()
    ci.compute_3d_sight_lines(nodes, buildings, buildings, None, "t", num_workers=1, tmp_dir=chunks)
    assert list(chunks.iterdir()) == []


def test_verbose_sight_lines_show_their_progress(monkeypatch):
    from cityImage import visibility3d

    logger = logging.getLogger("cityImage.visibility3d")
    monkeypatch.setattr(logger, "level", logging.NOTSET)
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "propagate", False)  # as if no logging were configured

    visibility3d._show_progress_logs()
    visibility3d._show_progress_logs()

    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1


# --- Networks -------------------------------------------------------------------------------


def test_sight_line_progress_logs_once_per_bar_step(caplog):
    from cityImage.visibility3d import _ProgressLogger

    progress = _ProgressLogger(enabled=True)
    progress.n_chunks = 240

    with caplog.at_level("INFO", logger="cityImage.visibility3d"):
        for done in range(1, 241):
            progress.chunk(done, 10, 5, 1.0, 1.0)

    assert len(caplog.records) == _ProgressLogger.BAR_WIDTH + 1  # steps 0 to 24
    assert caplog.records[-1].getMessage().startswith("chunk 240/240")


def test_network_from_lines_joins_3d_lines_to_their_nodes():
    lines = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0, 1), (100, 0, 2)]), LineString([(100, 0, 2), (200, 0, 3)])],
        crs=CRS,
    )

    nodes, edges = ci.network_from_lines(lines, CRS)

    assert len(nodes) == 3
    assert edges["u"].notna().all() and edges["v"].notna().all()
    assert edges["v"].iloc[0] == edges["u"].iloc[1]


def test_clean_network_keeps_3d_lines_3d():
    nodes = gpd.GeoDataFrame(
        {"nodeID": [0, 1, 2, 3]},
        geometry=[Point(0, 0, 1), Point(10, 0, 2), Point(20, 0, 3), Point(10, 10, 4)],
        crs=CRS,
    )
    edges = gpd.GeoDataFrame(
        {"edgeID": [0, 1, 2], "u": [0, 1, 1], "v": [1, 2, 3]},
        geometry=[
            LineString([(0, 0, 1), (5, 1, 1.5), (10, 0, 2)]),
            LineString([(10, 0, 2), (20, 0, 3)]),
            LineString([(10, 0, 2), (10, 10, 4)]),
        ],
        crs=CRS,
    )

    _, cleaned = ci.clean_network(nodes, edges)

    assert len(cleaned) == 3
    assert cleaned.geometry.has_z.all()


def test_remove_disconnected_islands_accepts_an_empty_network():
    nodes = gpd.GeoDataFrame({"nodeID": []}, geometry=[], crs=CRS)
    edges = gpd.GeoDataFrame({"edgeID": [], "u": [], "v": []}, geometry=[], crs=CRS)

    out_nodes, out_edges = ci.remove_disconnected_islands(nodes, edges)

    assert out_nodes.empty and out_edges.empty


def test_dual_graph_respects_one_way_streets():
    nodes = gpd.GeoDataFrame(
        {"nodeID": [0, 1, 2]}, geometry=[Point(0, 0), Point(10, 0), Point(20, 0)], crs=CRS
    )
    edges = gpd.GeoDataFrame(
        {"edgeID": [0, 1], "u": [0, 1], "v": [1, 2], "oneway": [1, 1], "length": [10.0, 10.0]},
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(10, 0), (20, 0)])],
        crs=CRS,
    )

    nodes_dual, edges_dual = ci.dual_gdf(nodes, edges, CRS, oneway=True)
    dual_graph = ci.dual_graph_fromGDF(nodes_dual, edges_dual)

    assert edges_dual[["u", "v", "oneway"]].values.tolist() == [[0, 1, 1]]
    assert not dual_graph.is_directed()
    # The undirected edge keeps the allowed direction, 0 -> 1, for the modeller to follow.
    data = dual_graph.edges[1, 0]
    assert (data["u"], data["v"], data["oneway"]) == (0, 1, 1)
    _, edges_back = ci.from_nx_to_gdf(dual_graph, CRS)
    assert edges_back[["u", "v", "oneway"]].values.tolist() == [[0, 1, 1]]


def test_a_two_way_pair_is_one_row_and_both_moves():
    nodes = gpd.GeoDataFrame(
        {"nodeID": [0, 1, 2, 3]},
        geometry=[Point(0, 0), Point(10, 0), Point(20, 0), Point(30, 0)],
        crs=CRS,
    )
    edges = gpd.GeoDataFrame(
        {"edgeID": [0, 1, 2], "u": [0, 1, 2], "v": [1, 2, 3], "oneway": [0, 0, 1]},
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(10, 0), (20, 0)]),
            LineString([(20, 0), (30, 0)]),
        ],
        crs=CRS,
    )
    edges["length"] = edges.geometry.length

    nodes_dual, edges_dual = ci.dual_gdf(nodes, edges, CRS, oneway=True)
    dual_graph = ci.dual_graph_fromGDF(nodes_dual, edges_dual)

    # Segments 0-1 are two-way: one row, both moves. Segment 2 runs one way, away from 1.
    assert sorted(edges_dual[["u", "v", "oneway"]].values.tolist()) == [[0, 1, 0], [1, 2, 1]]
    assert dual_graph.number_of_edges() == 2
    assert dual_graph.edges[0, 1]["oneway"] == 0
    assert dual_graph.edges[1, 2]["oneway"] == 1


def _three_segments(oneway):
    nodes = gpd.GeoDataFrame(
        {"nodeID": [0, 1, 2, 3]},
        geometry=[Point(0, 0), Point(10, 0), Point(20, 0), Point(30, 0)],
        crs=CRS,
    )
    edges = gpd.GeoDataFrame(
        {"edgeID": [0, 1, 2], "u": [0, 1, 2], "v": [1, 2, 3], "oneway": oneway},
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(10, 0), (20, 0)]),
            LineString([(20, 0), (30, 0)]),
        ],
        crs=CRS,
    )
    edges["length"] = edges.geometry.length
    return nodes, edges


@pytest.mark.parametrize(
    "oneway",
    [
        [False, False, True],
        ["no", "No", "yes"],
        ["false", None, "TRUE"],
        [0.0, np.nan, 1.0],
    ],
)
def test_dual_gdf_reads_booleans_and_yes_no_as_oneway(oneway):
    nodes, edges = _three_segments(oneway)

    _, edges_dual = ci.dual_gdf(nodes, edges, CRS, oneway=True)

    assert sorted(edges_dual[["u", "v", "oneway"]].values.tolist()) == [[0, 1, 0], [1, 2, 1]]


def test_dual_gdf_raises_on_an_unreadable_oneway():
    nodes, edges = _three_segments(["no", "-1", "reversible"])

    with pytest.raises(ValueError, match="-1.*reversible"):
        ci.dual_gdf(nodes, edges, CRS, oneway=True)


def test_dual_graph_without_oneway_stays_undirected():
    nodes = gpd.GeoDataFrame(
        {"nodeID": [0, 1, 2]}, geometry=[Point(0, 0), Point(10, 0), Point(20, 0)], crs=CRS
    )
    edges = gpd.GeoDataFrame(
        {"edgeID": [0, 1], "u": [0, 1], "v": [1, 2], "length": [10.0, 10.0]},
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(10, 0), (20, 0)])],
        crs=CRS,
    )

    nodes_dual, edges_dual = ci.dual_gdf(nodes, edges, CRS)

    assert not ci.dual_graph_fromGDF(nodes_dual, edges_dual).is_directed()
    assert edges_dual["oneway"].tolist() == [0]


def test_network_from_osm_walk_projects_when_no_crs_is_given(monkeypatch):
    import cityImage.pedestrian as pedestrian

    features = gpd.GeoDataFrame(
        {"highway": ["footway"]},
        geometry=[LineString([(-0.100, 51.500), (-0.099, 51.500)])],
        crs="EPSG:4326",
    )
    calls = {}

    def fake_features(ox, method, *args, **kwargs):
        calls.update(kwargs)
        return features

    monkeypatch.setattr(pedestrian, "_call_osmnx_features", fake_features)

    nodes, edges = ci.network_from_osm(
        (51.5, -0.1), download_method="distance_from_point", network_type="walk", distance=300
    )

    assert edges.crs.is_projected
    assert edges["length"].iloc[0] == pytest.approx(69, abs=2)
    assert calls["dist"] == 300


def test_network_from_osm_walk_requires_a_distance_for_distance_methods():
    with pytest.raises(ValueError, match="distance is required"):
        ci.network_from_osm(
            (51.5, -0.1), download_method="distance_from_point", network_type="walk"
        )


# --- Centrality helpers ---------------------------------------------------------------------


def test_weight_nodes_matches_graph_nodes_by_node_id():
    nodes = gpd.GeoDataFrame({"nodeID": [10, 20]}, geometry=[Point(0, 0), Point(1000, 0)], crs=CRS)
    edges = gpd.GeoDataFrame(
        {"edgeID": [0], "u": [10], "v": [20]}, geometry=[LineString([(0, 0), (1000, 0)])], crs=CRS
    )
    graph = ci.graph_fromGDF(nodes, edges)
    services = gpd.GeoDataFrame(geometry=[Point(1, 1)], crs=CRS)

    centrality.weight_nodes(nodes, services, graph, "w", 50)

    assert graph.nodes[10]["w"] == 1
    assert graph.nodes[20]["w"] == 0


def test_append_edges_metrics_adds_no_rows_when_the_index_is_not_the_edge_id():
    nodes = gpd.GeoDataFrame(
        {"nodeID": [0, 1, 2]}, geometry=[Point(0, 0), Point(10, 0), Point(20, 0)], crs=CRS
    )
    edges = gpd.GeoDataFrame(
        {"edgeID": [7, 9], "u": [0, 1], "v": [1, 2]},
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(10, 0), (20, 0)])],
        crs=CRS,
    )
    graph = ci.graph_fromGDF(nodes, edges)

    out = centrality.append_edges_metrics(
        edges, graph, [nx.edge_betweenness_centrality(graph)], ["Eb"]
    )

    assert list(out.index) == [7, 9]
    assert out["Eb"].notna().all()


def _crescent_beside_a_straight_road():
    """Nodes 0-1-2 on a line, with a second, longer street (a crescent) also joining 0 and 1."""
    nodes = gpd.GeoDataFrame(
        {"nodeID": [0, 1, 2]}, geometry=[Point(0, 0), Point(10, 0), Point(20, 0)], crs=CRS
    )
    edges = gpd.GeoDataFrame(
        {"edgeID": [7, 8, 9], "u": [0, 0, 1], "v": [1, 1, 2], "key": [0, 0, 0]},
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(0, 0), (5, 8), (10, 0)]),
            LineString([(10, 0), (20, 0)]),
        ],
        crs=CRS,
    )
    edges["length"] = edges.geometry.length
    return nodes, edges


def test_multigraph_keeps_parallel_streets_sharing_a_key():
    nodes, edges = _crescent_beside_a_straight_road()

    multigraph = ci.multiGraph_fromGDF(nodes, edges)

    assert multigraph.number_of_edges(0, 1) == 2
    assert {data["edgeID"] for data in multigraph[0][1].values()} == {7, 8}


def test_edge_metrics_on_a_multigraph_give_every_street_a_value():
    nodes, edges = _crescent_beside_a_straight_road()
    multigraph = ci.multiGraph_fromGDF(nodes, edges)
    betweenness = nx.edge_betweenness_centrality(multigraph, weight="length", normalized=False)

    out = centrality.append_edges_metrics(edges, multigraph, [betweenness], ["Eb"])

    # Shortest paths take the straight road, so the crescent carries none of them.
    assert out.loc[7, "Eb"] > 0.0
    assert out.loc[8, "Eb"] == 0.0
    assert out.loc[9, "Eb"] > 0.0


def test_edge_metrics_on_a_graph_match_the_multigraph():
    nodes, edges = _crescent_beside_a_straight_road()
    graph = ci.graph_fromGDF(nodes, edges)
    multigraph = ci.multiGraph_fromGDF(nodes, edges)

    on_graph = centrality.append_edges_metrics(
        edges, graph, [nx.edge_betweenness_centrality(graph, weight="length")], ["Eb"]
    )
    on_multigraph = centrality.append_edges_metrics(
        edges, multigraph, [nx.edge_betweenness_centrality(multigraph, weight="length")], ["Eb"]
    )

    assert on_graph.loc[8, "Eb"] == 0.0  # the crescent, left out of the Graph
    assert on_graph["Eb"].tolist() == pytest.approx(on_multigraph["Eb"].tolist())


def test_node_centrality_on_a_multigraph_matches_the_graph():
    nodes, edges = _crescent_beside_a_straight_road()

    on_graph = ci.calculate_centrality(ci.graph_fromGDF(nodes, edges), weight="length")
    on_multigraph = ci.calculate_centrality(ci.multiGraph_fromGDF(nodes, edges), weight="length")

    assert on_multigraph == on_graph


# --- Barriers -------------------------------------------------------------------------------


def test_barriers_are_projected_when_no_crs_is_given():
    rail = gpd.GeoDataFrame(
        {"railway": ["rail"]},
        geometry=[LineString([(-0.10, 51.50), (-0.08, 51.50)])],
        crs="EPSG:4326",
    )
    lake = gpd.GeoDataFrame(
        {"natural": ["water"]}, geometry=[box(-0.10, 51.50, -0.09, 51.505)], crs="EPSG:4326"
    )

    out = barriers.barriers_from_osm_features(railways_gdf=rail, water_gdf=lake, crs=None)

    assert out.crs.is_projected
    assert sorted(out["barrier_type"]) == ["railway", "water"]
    width = out[out["barrier_type"] == "railway"].total_bounds
    assert width[2] - width[0] < 2000  # metres, not a 20-degree buffer


def test_barriers_without_a_crs_keep_their_own_units():
    rail = gpd.GeoDataFrame(
        {"railway": ["rail"]}, geometry=[LineString([(350000, 400000), (352000, 400000)])]
    )

    out = barriers.railway_barriers_from_osm_features(rail)

    assert out.crs is None
    assert out.total_bounds[0] == pytest.approx(350000, abs=50)


def test_along_within_parks_leaves_the_caller_frame():
    edges = gpd.GeoDataFrame({"edgeID": [0]}, geometry=[LineString([(0, 0), (10, 0)])], crs=CRS)
    parks = gpd.GeoDataFrame(
        {"barrierID": [0], "barrier_type": ["park"]},
        geometry=[box(-5, -5, 20, 5).boundary],
        crs=CRS,
    )

    out = barriers.along_within_parks(edges, parks)

    assert "w_parks" in out.columns
    assert "w_parks" not in edges.columns


# --- Buildings and heights ------------------------------------------------------------------


def test_gdf_multipolygon_to_polygon_keeps_ids_when_nothing_is_split():
    gdf = gpd.GeoDataFrame(
        {"buildingID": [101, 205]}, geometry=[box(0, 0, 1, 1), box(5, 5, 6, 6)], crs=CRS
    )

    assert ci.gdf_multipolygon_to_polygon(gdf)["buildingID"].tolist() == [101, 205]


def test_buildings_from_file_keeps_the_file_ids(tmp_path):
    path = tmp_path / "b.gpkg"
    gpd.GeoDataFrame(
        {"buildingID": [101, 205], "height": [10.0, 12.0]},
        geometry=[box(0, 0, 20, 20), box(50, 0, 70, 20)],
        crs=CRS,
    ).to_file(path)

    assert ci.buildings_from_file(str(path), CRS)["buildingID"].tolist() == [101, 205]


def test_buildings_from_file_drops_only_buildings_known_to_be_below_min_height(tmp_path):
    path = tmp_path / "b.gpkg"
    gpd.GeoDataFrame(
        {"buildingID": [0, 1, 2, 3, 4, 5], "h": [12.0, np.nan, 0.0, 3.0, 20.0, 5.0]},
        geometry=[box(i * 50, 0, i * 50 + 20, 20) for i in range(6)],
        crs=CRS,
    ).to_file(path)

    out = ci.buildings_from_file(str(path), CRS, height_field="h", min_height=5)

    assert out["buildingID"].tolist() == [0, 1, 2, 4, 5]  # only the 3 m building is dropped
    heights = out.set_index("buildingID")["height"]
    assert heights[[0, 4, 5]].tolist() == [12.0, 20.0, 5.0]
    assert heights[[1, 2]].isna().all()  # missing and zero heights are unknown


def test_the_building_schema_reads_heights_as_metres_above_zero():
    buildings = gpd.GeoDataFrame(
        {"buildingID": range(6), "height": ["12 m", "7,5", 0, -3.0, "bad", None]},
        geometry=[box(i * 50, 0, i * 50 + 20, 20) for i in range(6)],
        crs=CRS,
    )

    out = ci.standardize_buildings_gdf(buildings)

    assert out["height"].dtype == float
    assert out["height"].tolist()[:2] == [12.0, 7.5]
    assert out["height"].iloc[2:].isna().all()


def test_visibility_score_leaves_a_building_without_a_height_out_of_fac_and_3dvis():
    buildings = gpd.GeoDataFrame(
        {"buildingID": [0, 1, 2, 3], "height": [10.0, 0.0, -4.0, np.nan]},
        geometry=[box(i * 50, 0, i * 50 + 10, 10) for i in range(4)],
        crs=CRS,
    )
    sight_lines = gpd.GeoDataFrame(
        {"nodeID": [1, 1], "buildingID": [0, 1]},
        geometry=[LineString([(0, 50), (5, 10)]), LineString([(50, 50), (55, 10)])],
        crs=CRS,
    )

    out = ci.visibility_score(buildings, sight_lines=sight_lines)

    assert out.loc[0, "fac"] == 100.0
    assert out.loc[0, "3dvis"] >= 0.0
    assert out.loc[1:, ["fac", "3dvis"]].isna().all().all()  # no negative or zero facade area


def test_zero_and_negative_heights_do_not_stretch_the_height_rescaling():
    with_bad = ci.score_buildings_global(_scored_buildings([10.0, -20.0, 30.0]))
    without = ci.score_buildings_global(_scored_buildings([10.0, np.nan, 30.0]))

    assert with_bad.loc[[0, 2], "height_sc"].tolist() == [0.0, 1.0]
    pd.testing.assert_series_equal(with_bad["vScore"], without["vScore"])
    assert math.isnan(with_bad.loc[1, "vScore"])


def test_a_building_without_a_height_does_not_move_the_visual_rescaling():
    # Building 1 has no height but carries fac and 3dvis values, as a hand-built frame may.
    scored = ci.score_buildings_global(_scored_buildings([5.0, np.nan, 7.0]))
    known = ci.score_buildings_global(_scored_buildings([5.0, np.nan, 7.0]).drop(index=1))

    for column in ("fac_sc", "3dvis_sc", "vScore", "vScore_sc"):
        assert scored.loc[[0, 2], column].tolist() == known[column].tolist()
        assert math.isnan(scored.loc[1, column])


def test_scaling_keeps_nan_when_the_known_values_are_equal():
    scaled = ci.scaling_columnDF(pd.Series([4.0, np.nan, 4.0]))

    assert scaled.tolist()[0::2] == [0.0, 0.0]
    assert math.isnan(scaled[1])


def test_global_and_local_scores_are_not_rounded():
    buildings = _scored_buildings([5.0, 6.0, 7.0])
    buildings["area"] = [1.0, 2.0, 7.0]  # thirds, which rounding would cut

    global_scores = ci.score_buildings_global(buildings)
    local_scores = ci.score_buildings_local(buildings)

    for column in (global_scores["gScore"], local_scores["lScore"]):
        assert column.tolist() != column.round(3).tolist()


def test_3dvis_is_not_computed_without_sight_lines():
    buildings = _scored_buildings([5.0, 6.0, 7.0]).drop(columns=["fac", "3dvis"])

    assert ci.visibility_score(buildings)["3dvis"].isna().all()
    # Sight lines that reach no building: nothing to score either.
    empty = gpd.GeoDataFrame({"nodeID": [], "buildingID": []}, geometry=[], crs=CRS)
    assert ci.visibility_score(buildings, sight_lines=empty)["3dvis"].isna().all()
    elsewhere = gpd.GeoDataFrame(
        {"nodeID": [0], "buildingID": [99]}, geometry=[LineString([(0, 0), (0, 50)])], crs=CRS
    )
    assert ci.visibility_score(buildings, sight_lines=elsewhere)["3dvis"].isna().all()


def test_3dvis_is_0_for_an_unreached_building_when_another_is_reached():
    buildings = _scored_buildings([5.0, 6.0, np.nan]).drop(columns=["fac", "3dvis"])
    lines = gpd.GeoDataFrame(
        {"nodeID": [0], "buildingID": [0]}, geometry=[LineString([(0, 0), (0, 50)])], crs=CRS
    )

    out = ci.visibility_score(buildings, sight_lines=lines)["3dvis"]

    # One reached building scales to 0 on its own, like the unreached one; no height stays NaN.
    assert out.tolist()[:2] == [0.0, 0.0]
    assert math.isnan(out.tolist()[2])


def test_cult_is_not_computed_without_a_historic_layer():
    buildings = _scored_buildings(None).drop(columns="cult")
    historic = gpd.GeoDataFrame(geometry=[box(0, 0, 5, 5)], crs=CRS)

    assert ci.cultural_score(buildings)["cult"].isna().all()
    with_layer = ci.cultural_score(buildings, historic_elements_gdf=historic)
    assert with_layer["cult"].tolist() == [1.0, 0.0, 0.0]


def test_cult_is_nan_only_when_no_building_has_anything():
    buildings = _scored_buildings(None).drop(columns="cult")
    far = gpd.GeoDataFrame({"grade": [2.0]}, geometry=[box(900, 900, 905, 905)], crs=CRS)
    zero = gpd.GeoDataFrame({"grade": [0.0]}, geometry=[box(0, 0, 5, 5)], crs=CRS)
    untagged = buildings.assign(historic=[None, "no", ""])

    # A layer no building touches, sums that are all 0, no historic tag: nothing to score.
    assert ci.cultural_score(buildings, historic_elements_gdf=far)["cult"].isna().all()
    assert ci.cultural_score(buildings, zero, score_column="grade")["cult"].isna().all()
    assert ci.cultural_score(untagged, from_OSM=True)["cult"].isna().all()
    # One building with something: the others get 0.
    tagged = buildings.assign(historic=[None, "castle", ""])
    assert ci.cultural_score(tagged, from_OSM=True)["cult"].tolist() == [0.0, 1.0, 0.0]


def test_a_building_with_no_neighbour_is_as_unexpected_as_can_be():
    buildings = _scored_buildings(None)
    buildings["land_uses"] = [["residential"]] * 3

    prag = ci.pragmatic_score(buildings, search_radius=5)["prag"]  # 10 m apart

    assert prag.tolist() == [1.0, 1.0, 1.0]


def test_global_scores_write_only_the_computed_components():
    buildings = _scored_buildings([5.0, 6.0, 7.0]).drop(columns=["cult", "prag"])

    scored = ci.score_buildings_global(buildings)

    assert {"vScore", "sScore"} <= set(scored.columns)
    assert not {"cScore", "cScore_sc", "pScore", "pScore_sc"} & set(scored.columns)


@pytest.mark.parametrize("heights", [None, [np.nan, 0.0, np.nan]], ids=["no-column", "unknown"])
def test_scores_without_heights_write_no_visual_columns(heights):
    buildings = _scored_buildings(heights)
    visual = {"vScore", "vScore_sc", "vScore_l", "fac_sc", "height_sc", "3dvis_sc"}

    global_scores = ci.score_buildings_global(buildings)
    local_scores = ci.score_buildings_local(buildings)

    for scored in (global_scores, local_scores):
        assert not visual & set(scored.columns)
        assert ("height" in scored.columns) == (heights is not None)  # never added
    assert global_scores["gScore"].notna().all()
    assert local_scores["lScore"].notna().all()


def test_scores_read_height_strings():
    scored = ci.score_buildings_global(_scored_buildings(["10 m", "bad", "30 m"]))

    assert scored["height"].tolist()[0::2] == [10.0, 30.0]
    assert math.isnan(scored.loc[1, "vScore"])


def test_buildings_from_file_with_an_empty_height_field_scores_without_heights(tmp_path):
    path = tmp_path / "b.gpkg"
    gpd.GeoDataFrame(
        {"buildingID": [0, 1, 2], "h": [np.nan, 0.0, np.nan]},
        geometry=[box(i * 50, 0, i * 50 + 20, 20) for i in range(3)],
        crs=CRS,
    ).to_file(path)

    out = ci.buildings_from_file(str(path), CRS, height_field="h")
    scored = ci.score_buildings_global(out)

    assert out["buildingID"].tolist() == [0, 1, 2]
    assert out["height"].isna().all()
    assert "vScore" not in scored.columns
    assert scored["gScore"].notna().all()


def test_buildings_from_file_without_heights_keeps_every_building(tmp_path):
    path = tmp_path / "b.gpkg"
    gpd.GeoDataFrame(
        {"buildingID": [0, 1]}, geometry=[box(0, 0, 20, 20), box(50, 0, 70, 20)], crs=CRS
    ).to_file(path)

    out = ci.buildings_from_file(str(path), CRS, min_height=5)

    assert out["buildingID"].tolist() == [0, 1]
    assert out["height"].isna().all()


def test_buildings_without_a_height_are_not_3d_obstructions():
    from cityImage import visibility3d

    buildings = gpd.GeoDataFrame(
        {"buildingID": [0, 1, 2, 3], "height": [12.0, 0.0, np.nan, "bad"]},
        geometry=[box(i * 50, 0, i * 50 + 20, 20) for i in range(4)],
        crs=CRS,
    )

    prepared = visibility3d._prepare_buildings_gdf(buildings)

    assert prepared["buildingID"].tolist() == [0]
    assert prepared["height"].tolist() == [12.0]


def test_3d_building_base_is_used_as_given_and_zero_when_missing():
    from cityImage import visibility3d

    buildings = gpd.GeoDataFrame(
        {"buildingID": [0, 1, 2], "height": [12.0, 12.0, 12.0], "base": [0.3, np.nan, -2.0]},
        geometry=[box(i * 50, 0, i * 50 + 20, 20) for i in range(3)],
        crs=CRS,
    )

    prepared = visibility3d._prepare_buildings_gdf(buildings)

    assert prepared["base"].tolist() == [0.3, 0.0, -2.0]
    assert visibility3d._prepare_buildings_gdf(buildings.drop(columns="base"))["base"].eq(0).all()


def test_3d_targets_are_the_buildings_at_least_min_target_height_tall():
    from cityImage import visibility3d

    nodes = gpd.GeoDataFrame(
        {"nodeID": [0], "x": [0.0], "y": [-400.0]}, geometry=[Point(0, -400)], crs=CRS
    )
    buildings = gpd.GeoDataFrame(
        {"buildingID": [0, 1, 2], "height": [4.0, 5.0, 12.0]},
        geometry=[box(i * 50, 0, i * 50 + 20, 20) for i in range(3)],
        crs=CRS,
    )

    def targets(**kwargs):
        _, target_points, obstructions = visibility3d._prepare_3d_sight_lines(
            nodes, buildings, buildings, **kwargs
        )
        assert sorted(obstructions["buildingID"]) == [0, 1, 2]  # every height obstructs
        return sorted(target_points["buildingID"].unique())

    assert targets() == [1, 2]  # the 5 m default, inclusive
    assert targets(min_target_height=10.0) == [2]


def _observer_nodes(z):
    nodes = gpd.GeoDataFrame(
        {"nodeID": [0, 1, 2], "x": [0.0, 30.0, 60.0], "y": [-400.0] * 3},
        geometry=[Point(x, -400) for x in (0, 30, 60)],
        crs=CRS,
    )
    if z is not None:
        nodes["z"] = z
    return nodes


def test_3d_observers_stand_on_their_z_as_given():
    from cityImage import visibility3d

    eyes = visibility3d._observer_eyes(_observer_nodes([-80.0, 0.0, 12.0]), observer_height=1.6)

    assert [point.z for point in eyes] == pytest.approx([-78.4, 1.6, 13.6])  # no nodata guess


@pytest.mark.parametrize("z", [None, [np.nan] * 3], ids=["no-column", "all-missing"])
def test_3d_observers_without_elevations_stand_at_zero(z):
    from cityImage import visibility3d

    eyes = visibility3d._observer_eyes(_observer_nodes(z), observer_height=1.6)

    assert [point.z for point in eyes] == pytest.approx([1.6] * 3)


def test_3d_observers_without_a_z_are_left_out_where_others_have_one(caplog):
    from cityImage import visibility3d

    buildings = gpd.GeoDataFrame(
        {"buildingID": [0], "height": [12.0], "base": [10.0]}, geometry=[box(0, 0, 20, 20)], crs=CRS
    )

    with caplog.at_level(logging.INFO, logger="cityImage.visibility3d"):
        observers, _, _ = visibility3d._prepare_3d_sight_lines(
            _observer_nodes([10.0, np.nan, 11.0]), buildings, buildings
        )

    assert observers["nodeID"].tolist() == [0, 2]
    assert "Left out 1 observer node(s)" in caplog.text
    assert "different grounds" not in caplog.text


def test_3d_sight_lines_warn_when_only_one_side_has_elevations(caplog):
    from cityImage import visibility3d

    buildings = gpd.GeoDataFrame(
        {"buildingID": [0], "height": [12.0]}, geometry=[box(0, 0, 20, 20)], crs=CRS
    )

    with caplog.at_level(logging.WARNING, logger="cityImage.visibility3d"):
        visibility3d._prepare_3d_sight_lines(_observer_nodes([50.0] * 3), buildings, buildings)

    assert "nodes have elevations but the buildings (base) do not" in caplog.text


def test_2d_networks_are_loaded_at_ground_level():
    lines = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (100, 0)])], crs=CRS)

    nodes, _ = ci.network_from_lines(lines, CRS)

    assert nodes["z"].eq(0.0).all()


def test_3d_sight_lines_need_a_building_with_a_height():
    from cityImage import visibility3d

    buildings = gpd.GeoDataFrame(
        {"buildingID": [0, 1], "height": [np.nan, 0.0]},
        geometry=[box(0, 0, 20, 20), box(50, 0, 70, 20)],
        crs=CRS,
    )

    with pytest.raises(ValueError, match="no building has one"):
        visibility3d._prepare_buildings_gdf(buildings)
    with pytest.raises(ValueError, match="no building has one"):
        visibility3d._prepare_buildings_gdf(buildings.drop(columns="height"))


def test_buildings_from_osm_keeps_height_tags_only_when_asked(monkeypatch):
    from cityImage import osm

    features = gpd.GeoDataFrame(
        {"building": ["yes", "yes"], "height": ["12 m", None]},
        geometry=[box(0, 0, 20, 20), box(50, 0, 70, 20)],
        crs=CRS,
    )
    monkeypatch.setattr(osm, "features_from_osm", lambda *args, **kwargs: features.copy())

    without = ci.buildings_from_osm("anywhere", crs=CRS)
    tagged = ci.buildings_from_osm("anywhere", crs=CRS, keep_osm_heights=True)

    assert without["height"].isna().all()
    assert tagged["height"].iloc[0] == 12.0
    assert math.isnan(tagged["height"].iloc[1])


def test_scores_give_a_building_without_a_height_no_visual_score():
    buildings = _scored_buildings([5.0, np.nan, 7.0])  # building 1 is the most visible in 3D

    global_scores = ci.score_buildings_global(buildings)
    local_scores = ci.score_buildings_local(buildings)

    assert global_scores["buildingID"].tolist() == [0, 1, 2]
    assert math.isnan(global_scores.loc[1, "vScore"])  # no visual score, adds nothing to gScore
    assert global_scores["gScore"].notna().all()
    assert local_scores["buildingID"].tolist() == [0, 1, 2]
    assert local_scores["lScore"].notna().all()


def test_scores_without_heights_leave_out_the_visual_component():
    scored = ci.score_buildings_global(_scored_buildings(None))

    assert "vScore" not in scored.columns
    assert scored["gScore"].notna().all()


def test_local_scores_write_only_the_local_score():
    buildings = _scored_buildings([5.0, 6.0, 7.0])

    out = ci.score_buildings_local(buildings)

    assert set(out.columns) - set(buildings.columns) == {"lScore", "lScore_sc"}


def test_score_buildings_local_leaves_the_caller_frame():
    buildings = _scored_buildings([5.0, 6.0, 7.0])
    before = list(buildings.columns)

    out = ci.score_buildings_local(buildings)

    assert list(buildings.columns) == before
    assert "lScore" in out.columns


def test_visibility_score_keeps_the_caller_index():
    buildings = gpd.GeoDataFrame(
        {"buildingID": [0, 1], "height": [10.0, 12.0]},
        geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10)],
        crs=CRS,
        index=[50, 60],
    )
    sight_lines = gpd.GeoDataFrame(
        {"nodeID": [1], "buildingID": [1]}, geometry=[LineString([(0, 50), (25, 10)])], crs=CRS
    )

    out = ci.visibility_score(buildings, sight_lines=sight_lines)

    assert list(out.index) == [50, 60]
    assert out["3dvis"].notna().all()


def test_one_detailed_building_gives_its_height_to_its_best_match_only():
    buildings = gpd.GeoDataFrame(
        {"buildingID": [0, 1]}, geometry=[box(0, 0, 100, 10), box(100, 0, 200, 10)], crs=CRS
    )
    detailed = gpd.GeoDataFrame(
        {"base": [0.0], "height": [30.0]}, geometry=[box(94, 0, 104, 10)], crs=CRS
    )  # 60% on building 0, 40% on building 1

    out = height.assign_building_heights_from_other_gdf(buildings, detailed, CRS, min_overlap=0.4)

    assert out["height"].iloc[0] == 30.0
    assert math.isnan(out["height"].iloc[1])
