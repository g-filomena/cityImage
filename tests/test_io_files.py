"""Offline tests for file loaders in ``cityImage.io`` (round-tripped through temp GeoPackages)."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

import cityImage as ci

UTM = "EPSG:32633"


def _square(x0, side=20.0):
    return Polygon([(x0, 0), (x0 + side, 0), (x0 + side, side), (x0, side)])


def _write_buildings(path, extra=None):
    cols = {"osm_id": [1, 2]}
    if extra:
        cols.update(extra)
    gdf = gpd.GeoDataFrame(cols, geometry=[_square(0), _square(40)], crs=UTM)  # two 400 m^2 squares
    gdf.to_file(path, driver="GPKG")
    return str(path)


def test_buildings_from_file_applies_height_base_defaults(tmp_path):
    out = ci.buildings_from_file(_write_buildings(tmp_path / "b.gpkg"), crs=UTM, min_area=100)
    assert (out["height"] == 5).all()  # min_height default when no height field/column
    assert (out["base"] == 0.0).all()  # base default
    assert "buildingID" in out.columns


def test_buildings_from_file_reads_named_fields(tmp_path):
    path = _write_buildings(
        tmp_path / "b.gpkg", {"h": [12.0, 20.0], "b": [1.0, 2.0], "lu": ["church", "shop"]}
    )
    out = ci.buildings_from_file(
        path, crs=UTM, min_area=100, height_field="h", base_field="b", land_uses_raw_field="lu"
    )
    assert out.sort_values("buildingID")["height"].tolist() == [12.0, 20.0]
    assert "land_uses_raw" in out.columns


@pytest.mark.parametrize("field", ["height_field", "base_field", "land_uses_raw_field"])
def test_buildings_from_file_missing_field_raises(tmp_path, field):
    path = _write_buildings(tmp_path / "b.gpkg")
    with pytest.raises(ValueError, match="not found"):
        ci.buildings_from_file(path, crs=UTM, min_area=100, **{field: "missing"})


def test_buildings_from_file_case_study_area_and_distance(tmp_path):
    path = _write_buildings(tmp_path / "b.gpkg")
    only_first = Polygon([(-5, -5), (25, -5), (25, 25), (-5, 25)])  # covers building 1 only
    clipped = ci.buildings_from_file(path, crs=UTM, min_area=100, case_study_area=only_first)
    assert len(clipped) == 1

    around_centre = ci.buildings_from_file(path, crs=UTM, min_area=100, distance_from_center=50)
    assert 1 <= len(around_centre) <= 2  # centroid-distance study area


def test_network_from_file_builds_network(tmp_path):
    lines = gpd.GeoDataFrame(
        {"road": ["a", "b"]},
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(10, 0), (10, 10)])],
        crs=UTM,
    )
    path = str(tmp_path / "n.gpkg")
    lines.to_file(path, driver="GPKG")

    nodes, edges = ci.network_from_file(path, UTM, other_columns=["road"])
    assert len(edges) == 2 and len(nodes) == 3
    assert {"u", "v", "edgeID", "road"}.issubset(edges.columns)
