"""Tests for the hard utilities split."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon

import cityImage as ci


def test_scaling_helper_remains_available_from_top_level():
    series = pd.Series([10.0, 20.0, 30.0], index=["a", "b", "c"])

    assert ci.scaling_columnDF(series).tolist() == [0.0, 0.5, 1.0]
    assert ci.scaling_columnDF(series, inverse=True).tolist() == [1.0, 0.5, 0.0]


def test_low_level_data_helpers_are_not_public_api():
    assert not hasattr(ci, "rescale_ranges")
    assert not hasattr(ci, "resolve_lists_columns")


def test_geometry_helpers_remain_available_from_top_level():
    line_a = LineString([(0, 0), (10, 0)])
    line_b = LineString([(0, 2), (10, 2)])

    centre = ci.center_line([line_a, line_b])
    assert list(centre.coords) == [(0.0, 1.0), (10.0, 1.0)]

    segments = ci.split_line_at_MultiPoint(
        LineString([(0, 0), (10, 0)]),
        MultiPoint([Point(5, 0)]),
        z=None,
    )
    assert len(segments) == 2


def test_multipart_lines_are_merged_or_exploded_without_dataframe_append():
    gdf = gpd.GeoDataFrame(
        {"id": [1]},
        geometry=[MultiLineString([[(0, 0), (1, 0)], [(1, 0), (2, 0)]])],
        crs="EPSG:3857",
    )

    fixed = ci.fix_multiparts_LineString_gdf(gdf)

    assert not fixed.empty
    assert set(fixed.geometry.type).issubset({"LineString"})


def test_multipolygon_to_polygon_preserves_polygon_output_and_area():
    gdf = gpd.GeoDataFrame(
        {"buildingID": [10]},
        geometry=[MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])])],
        crs="EPSG:3857",
    )

    out = ci.gdf_multipolygon_to_polygon(gdf)

    assert out["buildingID"].tolist() == [0]
    assert out.geometry.iloc[0].geom_type == "Polygon"
    assert out["area"].tolist() == pytest.approx([1.0])


def test_downloader_is_no_longer_owned_by_cityimage_top_level():
    assert not hasattr(ci, "downloader")
