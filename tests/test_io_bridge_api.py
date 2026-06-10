"""Tests for cityImage file IO bridge helpers."""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString, Polygon

from cityImage import io


def test_network_from_file_delegates_file_io_to_geopandas_and_schema_to_cityimage(monkeypatch):
    raw_edges = gpd.GeoDataFrame(
        {"road_name": ["a"]},
        geometry=[LineString([(0, 0), (10, 0)])],
        crs="EPSG:3857",
    )

    monkeypatch.setattr(gpd, "read_file", lambda input_path: raw_edges.copy())

    nodes, edges = io.network_from_file(
        "dummy.gpkg",
        "EPSG:3857",
        other_columns=["road_name"],
    )

    assert nodes["nodeID"].tolist() == [0, 1]
    assert edges["u"].tolist() == [0]
    assert edges["v"].tolist() == [1]
    assert edges["road_name"].tolist() == ["a"]


def test_buildings_from_file_preserves_height_base_and_land_use_fields(monkeypatch):
    raw_buildings = gpd.GeoDataFrame(
        {"h": [12.0], "b": [1.0], "lu": ["retail"]},
        geometry=[Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])],
        crs="EPSG:3857",
    )

    monkeypatch.setattr(gpd, "read_file", lambda input_path: raw_buildings.copy())

    buildings = io.buildings_from_file(
        "dummy.gpkg",
        "EPSG:3857",
        height_field="h",
        base_field="b",
        land_uses_raw_field="lu",
        min_area=1,
    )

    assert buildings.loc[0, "height"] == 12.0
    assert buildings.loc[0, "base"] == 1.0
    assert buildings.loc[0, "land_uses_raw"] == ["retail"]
