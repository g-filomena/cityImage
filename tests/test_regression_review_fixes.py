"""Behaviours that are easy to get wrong: districts, visibility, networks, centrality, barriers,
buildings and heights.

Each test pins one behaviour with the smallest realistic input.
"""

from __future__ import annotations

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

    with pytest.raises(ValueError, match="larger than the network"):
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
    dual_graph = ci.dual_graph_fromGDF(nodes_dual, edges_dual, directed=True)

    assert dual_graph.is_directed()
    assert not ci.dual_graph_fromGDF(nodes_dual, edges_dual).is_directed()
    assert nx.has_path(dual_graph, 0, 1)
    assert not nx.has_path(dual_graph, 1, 0)


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
    assert len(edges_dual) == 1


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


def test_edge_metrics_refuse_a_graph_missing_parallel_streets():
    nodes, edges = _crescent_beside_a_straight_road()
    graph = ci.graph_fromGDF(nodes, edges)

    with pytest.raises(ValueError, match="multiGraph_fromGDF"):
        centrality.append_edges_metrics(
            edges, graph, [nx.edge_betweenness_centrality(graph, weight="length")], ["Eb"]
        )


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


def test_buildings_from_file_drops_buildings_below_min_height(tmp_path):
    path = tmp_path / "b.gpkg"
    gpd.GeoDataFrame(
        {"buildingID": [0, 1, 2, 3, 4], "h": [12.0, np.nan, 0.0, 3.0, 20.0]},
        geometry=[box(i * 50, 0, i * 50 + 20, 20) for i in range(5)],
        crs=CRS,
    ).to_file(path)

    out = ci.buildings_from_file(str(path), CRS, height_field="h")

    assert out["buildingID"].tolist() == [0, 4]  # missing, zero and 3 m dropped


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
    assert out["height"].tolist() == [5.0, 5.0]


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


def test_scores_leave_out_buildings_without_a_height():
    buildings = _scored_buildings([5.0, np.nan, 7.0])

    global_scores = ci.score_buildings_global(buildings)
    local_scores = ci.score_buildings_local(buildings)

    assert global_scores["buildingID"].tolist() == [0, 2]
    assert global_scores["gScore"].notna().all()
    assert local_scores["buildingID"].tolist() == [0, 2]
    assert local_scores["lScore"].notna().all()


def test_scores_without_heights_leave_out_the_visual_component():
    scored = ci.score_buildings_global(_scored_buildings(None))

    assert "vScore" not in scored.columns
    assert scored["gScore"].notna().all()


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
