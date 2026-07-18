"""End-to-end offline test for ``compute_3d_sight_lines``.

Drives the full 3D sight-line orchestrator (distance filter -> 2D obstruction
split via dask -> 3D PyVista ray casting -> chunked GeoPackage merge) on a tiny
synthetic city. No live data; the chunk temp dir is redirected into tmp_path.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

import cityImage as ci

pytest.importorskip("pyvista")
pytest.importorskip("dask")
pytest.importorskip("psutil")

CRS = "EPSG:3857"


def _nodes():
    # Two observers well to the east of the buildings (distance ~490-500 units).
    return gpd.GeoDataFrame(
        {"nodeID": [1, 2], "x": [500.0, 500.0], "y": [5.0, 25.0], "z": [2.0, 2.0]},
        geometry=[Point(500, 5), Point(500, 25)],
        crs=CRS,
    )


def _buildings():
    return gpd.GeoDataFrame(
        {"buildingID": [1, 2], "height": [20.0, 25.0], "base": [1.0, 1.0]},
        geometry=[
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(0, 20), (10, 20), (10, 30), (0, 30)]),
        ],
        crs=CRS,
    )


def _edges():
    return gpd.GeoDataFrame(
        {"edgeID": [1], "u": [1], "v": [2]},
        geometry=[LineString([(500, 5), (500, 25)])],
        crs=CRS,
    )


def test_compute_3d_sight_lines_end_to_end(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # chunk files go to tmp_path/sight_lines_tmp
    buildings = _buildings()

    out = ci.compute_3d_sight_lines(
        nodes_gdf=_nodes(),
        target_buildings_gdf=buildings.copy(),
        obstructions_buildings_gdf=buildings.copy(),
        simplified_target_buildings=None,
        edges_gdf=_edges(),
        city_name="Test",
        distance_along=5,
        min_observer_target_distance=100,
        num_workers=1,
    )

    assert isinstance(out, gpd.GeoDataFrame)
    # Near-edge roof points are unobstructed, so at least one visible line is found,
    # and every returned row is a genuine 3D LineString.
    assert len(out) > 0
    assert (out.geometry.geom_type == "LineString").all()
    assert out.geometry.iloc[0].has_z


def test_compute_3d_sight_lines_with_consolidation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    buildings = _buildings()

    out = ci.compute_3d_sight_lines(
        nodes_gdf=_nodes(),
        target_buildings_gdf=buildings.copy(),
        obstructions_buildings_gdf=buildings.copy(),
        simplified_target_buildings=None,
        edges_gdf=_edges(),
        city_name="TestC",
        distance_along=5,
        min_observer_target_distance=100,
        consolidate=True,
        consolidate_tolerance=5.0,
        num_workers=1,
    )

    # The consolidate path additionally runs consolidate_nodes and the final _last_check;
    # it must still yield genuine 3D sight lines (observers are 20 units apart, so the
    # 5-unit tolerance merges nothing and visible lines are still found).
    assert isinstance(out, gpd.GeoDataFrame)
    assert len(out) > 0
    assert (out.geometry.geom_type == "LineString").all()
    assert out.geometry.iloc[0].has_z


def test_compute_3d_sight_lines_with_simplified_targets(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    buildings = _buildings()
    # A single simplified outline enclosing both detailed target buildings, which routes the
    # pipeline through _use_simplified_buildings (detailed targets mapped onto simplified geometry).
    simplified = gpd.GeoDataFrame(
        {"simplifiedID": [1]},
        geometry=[Polygon([(-1, -1), (11, -1), (11, 31), (-1, 31)])],
        crs=CRS,
    )

    out = ci.compute_3d_sight_lines(
        nodes_gdf=_nodes(),
        target_buildings_gdf=buildings.copy(),
        obstructions_buildings_gdf=buildings.copy(),
        simplified_target_buildings=simplified,
        edges_gdf=_edges(),
        city_name="TestSimplified",
        distance_along=5,
        min_observer_target_distance=100,
        num_workers=1,
    )

    assert isinstance(out, gpd.GeoDataFrame)
    assert len(out) > 0
    assert (out.geometry.geom_type == "LineString").all()
    assert out.geometry.iloc[0].has_z


def test_compute_3d_sight_lines_no_visible_returns_empty(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    buildings = _buildings()

    # An impossibly large minimum distance drops every candidate pair.
    out = ci.compute_3d_sight_lines(
        nodes_gdf=_nodes(),
        target_buildings_gdf=buildings.copy(),
        obstructions_buildings_gdf=buildings.copy(),
        simplified_target_buildings=None,
        edges_gdf=_edges(),
        city_name="TestEmpty",
        distance_along=5,
        min_observer_target_distance=100000,
        num_workers=1,
    )

    assert isinstance(out, gpd.GeoDataFrame)
    assert out.empty
