"""Tests for the hard building loading split."""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Polygon

import cityImage as ci


def test_select_buildings_by_study_area_with_polygon():
    buildings = gpd.GeoDataFrame(
        {"buildingID": [1, 2]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(5, 5), (6, 5), (6, 6), (5, 6)]),
        ],
        crs="EPSG:3857",
    )
    study_area = Polygon([(-1, -1), (2, -1), (2, 2), (-1, 2)])

    selected = ci.select_buildings_by_study_area(buildings, method="polygon", polygon=study_area)

    assert selected["buildingID"].tolist() == [1]


def test_standardize_buildings_preserves_raw_land_use_provenance():
    raw = gpd.GeoDataFrame(
        {"height_m": [10.0], "kind": ["retail"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:3857",
    )

    standardized = ci.standardize_buildings_gdf(
        raw,
        land_uses_raw_column="kind",
        add_area=True,
    )

    assert standardized["buildingID"].tolist() == [0]
    assert standardized["land_uses_raw"].tolist() == [["retail"]]
    assert "land_uses" not in standardized.columns
    assert "land_uses_overlap" not in standardized.columns
    assert standardized["area"].tolist() == [1.0]
