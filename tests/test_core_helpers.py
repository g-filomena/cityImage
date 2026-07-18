"""Offline tests for small core helpers: geometry, building selection, scoring, schema."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon

import cityImage as ci
from cityImage.buildings import select_buildings_by_study_area
from cityImage.geometry import (
    center_line,
    fix_multiparts_LineString_gdf,
    gdf_multipolygon_to_polygon,
    split_line_at_MultiPoint,
)
from cityImage.schema import (
    SchemaError,
    require_columns,
    require_geometry,
    require_land_use_lists,
    validate_buildings_gdf,
    validate_edges_gdf,
    validate_nodes_gdf,
)

CRS = "EPSG:3857"


# --------------------------------------------------------------------------- geometry


def test_center_line_averages_and_orients_lines():
    a = LineString([(0, 0), (10, 0)])
    b = LineString([(10, 2), (0, 2)])  # opposite direction -> reversed internally
    center = center_line([a, b])
    assert list(center.coords) == [(0.0, 1.0), (10.0, 1.0)]


def test_center_line_requires_two_lines():
    with pytest.raises(ValueError, match="At least two"):
        center_line([LineString([(0, 0), (1, 1)])])


def test_split_line_at_multipoint_with_and_without_z():
    line = LineString([(0, 0), (10, 0)])
    parts_3d = split_line_at_MultiPoint(line, [Point(5, 0)], z=0.0)
    assert len(parts_3d) == 2
    assert all(len(coord) == 3 for coord in parts_3d[0].coords)  # z added

    parts_2d = split_line_at_MultiPoint(line, [Point(5, 0)], z=None)
    assert all(len(coord) == 2 for coord in parts_2d[0].coords)  # kept 2D


def test_fix_multiparts_linestring_gdf_merges_and_explodes():
    mergeable = MultiLineString([[(0, 0), (1, 0)], [(1, 0), (2, 0)]])  # -> single LineString
    disjoint = MultiLineString([[(0, 5), (1, 5)], [(5, 5), (6, 5)]])  # -> explodes
    gdf = gpd.GeoDataFrame({"edgeID": [1, 2]}, geometry=[mergeable, disjoint], crs=CRS)

    out = fix_multiparts_LineString_gdf(gdf)

    assert set(out.geometry.type.unique()) == {"LineString"}
    assert len(out) >= 3  # disjoint part exploded into two lines


def test_gdf_multipolygon_to_polygon_converts_and_explodes():
    single = MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])
    multi = MultiPolygon(
        [Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])]
    )
    gdf = gpd.GeoDataFrame({"buildingID": [1, 2]}, geometry=[single, multi], crs=CRS)

    out = gdf_multipolygon_to_polygon(gdf)

    assert set(out.geometry.type.unique()) == {"Polygon"}
    assert (out["area"] > 0).all()


# --------------------------------------------------------------------------- building selection


def _grid_buildings():
    return gpd.GeoDataFrame(
        {"buildingID": [1, 2]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(100, 100), (101, 100), (101, 101), (100, 101)]),
        ],
        crs=CRS,
    )


def test_select_buildings_by_polygon_and_distance():
    buildings = _grid_buildings()
    area = Polygon([(-1, -1), (2, -1), (2, 2), (-1, 2)])  # only covers building 1
    by_polygon = select_buildings_by_study_area(buildings, method="polygon", polygon=area)
    assert by_polygon["buildingID"].tolist() == [1]

    by_distance = select_buildings_by_study_area(buildings, method="distance", distance=5)
    assert set(by_distance["buildingID"]).issubset({1, 2})


def test_select_buildings_edge_cases():
    buildings = _grid_buildings()
    assert select_buildings_by_study_area(buildings.iloc[0:0]).empty  # empty input
    assert select_buildings_by_study_area(buildings, method="polygon", polygon=None).empty
    with pytest.raises(ValueError, match="polygon.*distance"):
        select_buildings_by_study_area(buildings, method="bogus")


# --------------------------------------------------------------------------- scoring weights


def test_validate_score_weights_accepts_valid_and_rejects_invalid():
    assert ci.validate_score_weights({"a": 0.5, "b": 0.5}) == {"a": 0.5, "b": 0.5}
    with pytest.raises(ValueError, match="must not be empty"):
        ci.validate_score_weights({})
    with pytest.raises(ValueError, match="non-negative"):
        ci.validate_score_weights({"a": -0.1}, expected_sum=None)
    with pytest.raises(ValueError, match="must sum to"):
        ci.validate_score_weights({"a": 0.2, "b": 0.2})


def test_score_building_components_requires_edges_for_structural():
    buildings = _grid_buildings()
    buildings["area"] = buildings.geometry.area
    with pytest.raises(ValueError, match="requires edges_gdf"):
        ci.score_building_components(buildings, compute_structural=True, edges_gdf=None)


# --------------------------------------------------------------------------- schema


def test_require_columns_and_geometry_raise_schema_errors():
    df = gpd.GeoDataFrame({"a": [1]}, geometry=[Point(0, 0)], crs=CRS)
    with pytest.raises(SchemaError, match="missing required columns"):
        require_columns(df, ["a", "b"])

    with pytest.raises(SchemaError, match="must be a GeoDataFrame"):
        require_geometry(pd.DataFrame({"a": [1]}))

    nan_geom = gpd.GeoDataFrame({"a": [1]}, geometry=[None], crs=CRS)
    with pytest.raises(SchemaError, match="missing geometries"):
        require_geometry(nan_geom)

    empty_geom = gpd.GeoDataFrame({"a": [1]}, geometry=[Polygon()], crs=CRS)
    with pytest.raises(SchemaError, match="empty geometries"):
        require_geometry(empty_geom)


def test_require_land_use_lists_validates_shape():
    ok = gpd.GeoDataFrame(
        {"land_uses": [["retail"]], "land_uses_overlap": [[1.0]]},
        geometry=[Point(0, 0)],
        crs=CRS,
    )
    require_land_use_lists(ok)  # should not raise

    bad_type = gpd.GeoDataFrame(
        {"land_uses": ["retail"]},
        geometry=[Point(0, 0)],
        crs=CRS,  # scalar, not list
    )
    with pytest.raises(SchemaError, match="list-like"):
        require_land_use_lists(bad_type)

    mismatched = gpd.GeoDataFrame(
        {"land_uses": [["retail", "office"]], "land_uses_overlap": [[1.0]]},
        geometry=[Point(0, 0)],
        crs=CRS,
    )
    with pytest.raises(SchemaError, match="matching list lengths"):
        require_land_use_lists(mismatched)


def test_validate_reports_flag_missing_columns():
    empty = gpd.GeoDataFrame({"x": [1]}, geometry=[Point(0, 0)], crs=CRS)
    assert validate_nodes_gdf(empty).ok is False
    assert validate_edges_gdf(empty).ok is False

    buildings = gpd.GeoDataFrame({"buildingID": [1]}, geometry=[Point(0, 0)], crs=CRS)
    assert validate_buildings_gdf(buildings).ok is True
    assert validate_buildings_gdf(buildings, require_height=True).ok is False  # no height column
