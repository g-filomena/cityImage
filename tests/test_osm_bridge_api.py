"""Tests for cityImage OSM bridge helpers."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

from cityImage import osm


def test_buildings_from_osm_converts_raw_features_to_cityimage_schema(monkeypatch):
    raw_buildings = gpd.GeoDataFrame(
        {"building": ["yes"], "amenity": ["school"]},
        geometry=[Polygon([(0, 0), (30, 0), (30, 30), (0, 30)])],
        crs="EPSG:3857",
    )

    monkeypatch.setattr(
        osm,
        "features_from_osm",
        lambda query, tags, download_method="OSMplace", distance=None, crs=None: (
            raw_buildings.copy()
        ),
    )

    buildings = osm.buildings_from_osm(
        "Any place",
        crs="EPSG:3857",
        min_area=1,
    )

    assert buildings["buildingID"].tolist() == [0]
    assert "land_uses_raw" in buildings.columns
    assert "land_uses" in buildings.columns
    assert buildings.geometry.iloc[0].area == pytest.approx(900.0)


def test_features_from_osm_requires_valid_download_method_before_importing_osmnx():
    with pytest.raises(ValueError):
        osm.features_from_osm("Any place", {"building": True}, download_method="bad-method")
