"""Offline tests for the tractable helpers in ``cityImage.visibility3d``.

The heavy orchestrators (``compute_3d_sight_lines``, ``obstructions_3d``) drive
dask/pyvista/multiprocessing and are covered by the network-marked integration
run. These tests target the pure/geometry helpers that can run deterministically
offline.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

import cityImage as ci

CRS = "EPSG:3857"


def test_downsample_coords_spacing_and_midpoint_fallback():
    line = LineString([(0, 0), (5, 0), (10, 0), (15, 0)])
    # Spacing of 10 selects roughly every 10 units of accumulated length.
    sampled = ci.downsample_coords(line, distance_along=10)
    assert sampled == [(10.0, 0.0)]

    # When nothing reaches the spacing, the coordinate nearest the midpoint is returned.
    short = LineString([(0, 0), (1, 0)])
    fallback = ci.downsample_coords(short, distance_along=100)
    assert len(fallback) == 1


def test_filter_distance_keeps_pairs_beyond_threshold():
    chunk = gpd.GeoDataFrame(
        {"nodeID": [1], "observer_geo": [Point(0, 0)]},
        geometry=[Point(0, 0)],
        crs=CRS,
    )
    targets = gpd.GeoDataFrame(
        {"buildingID": [1, 2], "target_geo": [Point(100, 0), Point(5, 0)]},
        geometry=[Point(100, 0), Point(5, 0)],
        crs=CRS,
    )

    out = ci.filter_distance(chunk, targets, min_observer_target_distance=50)

    # Only the 100-unit pair survives the 50-unit minimum; the 5-unit pair is dropped.
    assert len(out) == 1
    assert out.iloc[0]["buildingID"] == 1
    assert out.iloc[0]["geometry"].length == pytest.approx(100.0)


def test_merge_gpkg_chunks_to_gdf(tmp_path):
    a = gpd.GeoDataFrame({"v": [1, 2]}, geometry=[Point(0, 0), Point(1, 1)], crs=CRS)
    b = gpd.GeoDataFrame({"v": [3]}, geometry=[Point(2, 2)], crs=CRS)
    fa, fb = tmp_path / "a.gpkg", tmp_path / "b.gpkg"
    a.to_file(fa, driver="GPKG")
    b.to_file(fb, driver="GPKG")

    merged = ci.merge_gpkg_chunks_to_gdf([str(fa), str(fb)], potential_obstructions_column="obs")
    assert len(merged) == 3
    assert sorted(merged["v"]) == [1, 2, 3]


def test_polygon_2d_to_3d_extrusion_flags():
    pv = pytest.importorskip("pyvista")
    footprint = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])

    # From sea level, height measured from base -> z spans [0, base + height].
    solid = ci.polygon_2d_to_3d(
        footprint,
        base=5.0,
        height=20.0,
        extrude_from_sealevel=True,
        height_relative_to_ground=False,
    )
    assert isinstance(solid, pv.PolyData) and solid.n_points > 0
    assert solid.bounds[4] == pytest.approx(0.0)
    assert solid.bounds[5] == pytest.approx(25.0)

    # From the given base, height measured from base -> z spans [base, base + height].
    solid2 = ci.polygon_2d_to_3d(
        footprint,
        base=5.0,
        height=20.0,
        extrude_from_sealevel=False,
        height_relative_to_ground=False,
    )
    assert solid2.bounds[4] == pytest.approx(5.0)
    assert solid2.bounds[5] == pytest.approx(25.0)

    # From the given base, absolute height -> z spans [base, height].
    solid3 = ci.polygon_2d_to_3d(
        footprint,
        base=5.0,
        height=20.0,
        extrude_from_sealevel=False,
        height_relative_to_ground=True,
    )
    assert solid3.bounds[4] == pytest.approx(5.0)
    assert solid3.bounds[5] == pytest.approx(20.0)
