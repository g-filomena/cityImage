"""Offline tests for land-use assignment (``cityImage.landuse.assign``).

These exercise the spatial point/polygon matching, overlap aggregation, default
filling and error handling without any live OSM or optional dependencies.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

import cityImage as ci

CRS = "EPSG:3857"


def _square(x0, y0, side=10.0):
    return Polygon([(x0, y0), (x0 + side, y0), (x0 + side, y0 + side), (x0, y0 + side)])


def _buildings():
    # Three disjoint 10x10 squares; the third never receives a match.
    return gpd.GeoDataFrame(
        {"buildingID": [1, 2, 3]},
        geometry=[_square(0, 0), _square(20, 0), _square(40, 0)],
        crs=CRS,
    )


def test_land_use_from_points_collects_and_deduplicates_preserving_order():
    points = gpd.GeoDataFrame(
        {"lu": ["retail", "retail", "education", "office"]},
        geometry=[Point(5, 5), Point(6, 6), Point(7, 7), Point(25, 5)],
        crs=CRS,
    )

    out = ci.land_use_from_points(
        _buildings(), points, new_land_use_column="land_uses", other_land_use_column="lu"
    )

    labels = out.set_index("buildingID")["land_uses"]
    assert labels[1] == ["retail", "education"]  # duplicate dropped, order preserved
    assert labels[2] == ["office"]
    assert labels[3] == []  # no matching point
    assert "overlap" not in out.columns  # point mode never creates an overlap column


def test_land_use_from_polygons_filters_by_threshold_and_normalises_overlaps():
    # Within building 1: retail covers 50%, green covers 20% (kept, == threshold),
    # tiny covers 10% (dropped, < threshold).
    others = gpd.GeoDataFrame(
        {"lu": ["retail", "green", "tiny"]},
        geometry=[
            Polygon([(0, 0), (5, 0), (5, 10), (0, 10)]),
            Polygon([(5, 0), (7, 0), (7, 10), (5, 10)]),
            Polygon([(7, 0), (8, 0), (8, 10), (7, 10)]),
        ],
        crs=CRS,
    )

    out = ci.land_use_from_polygons(
        _buildings(),
        others,
        new_land_use_column="land_uses",
        other_land_use_column="lu",
        overlap_column_name="overlap",
    )

    row = out.set_index("buildingID").loc[1]
    labels, overlaps = row["land_uses"], row["overlap"]
    assert "tiny" not in labels  # below the 0.20 threshold
    assert set(labels) == {"retail", "green"}
    assert labels[0] == "retail"  # ordered by (normalised) overlap, descending
    assert len(overlaps) == len(labels) == 2
    assert overlaps == sorted(overlaps, reverse=True)
    assert abs(sum(overlaps) - 1.0) < 1e-6  # renormalised to sum to 1.0


def test_land_use_from_polygons_identical_geometry_splits_multi_label_evenly():
    identical = gpd.GeoDataFrame(
        {"lu": [["a", "b"]]},  # a single polygon carrying two labels
        geometry=[_square(20, 0)],  # exactly equal to building 2
        crs=CRS,
    )

    out = ci.land_use_from_polygons(
        _buildings(),
        identical,
        new_land_use_column="land_uses",
        other_land_use_column="lu",
        overlap_column_name="overlap",
    )

    row = out.set_index("buildingID").loc[2]
    assert set(row["land_uses"]) == {"a", "b"}
    assert row["overlap"] == [0.5, 0.5]


def test_land_use_from_other_gdf_point_mode_fills_default_for_unmatched():
    points = gpd.GeoDataFrame({"lu": ["retail"]}, geometry=[Point(5, 5)], crs=CRS)

    out = ci.land_use_from_other_gdf(
        _buildings(),
        points,
        new_land_use_column="land_uses",
        other_land_use_column="lu",
        default_land_use="residential",
    )

    labels = out.set_index("buildingID")["land_uses"]
    assert labels[1] == ["retail"]
    assert labels[2] == ["residential"] and labels[3] == ["residential"]  # defaulted


def test_land_use_from_other_gdf_no_default_leaves_unmatched_empty():
    points = gpd.GeoDataFrame({"lu": ["retail"]}, geometry=[Point(5, 5)], crs=CRS)

    out = ci.land_use_from_other_gdf(
        _buildings(),
        points,
        new_land_use_column="land_uses",
        other_land_use_column="lu",
        fill_default_when_no_match=False,
    )

    labels = out.set_index("buildingID")["land_uses"]
    assert labels[2] == [] and labels[3] == []


def test_land_use_from_other_gdf_polygon_mode_assigns_default_overlap():
    others = gpd.GeoDataFrame(
        {"lu": ["retail"]},
        geometry=[Polygon([(0, 0), (5, 0), (5, 10), (0, 10)])],  # covers building 1
        crs=CRS,
    )

    out = ci.land_use_from_other_gdf(
        _buildings(),
        others,
        new_land_use_column="land_uses",
        other_land_use_column="lu",
        overlap_column_name="overlap",
        default_land_use="residential",
    )

    by_id = out.set_index("buildingID")
    assert by_id.loc[1, "land_uses"] == ["retail"]
    # Unmatched buildings get the default label and a full (1.0) default overlap.
    assert by_id.loc[3, "land_uses"] == ["residential"]
    assert by_id.loc[3, "overlap"] == [1.0]


def test_land_use_from_other_gdf_raises_on_crs_mismatch():
    others = gpd.GeoDataFrame({"lu": ["retail"]}, geometry=[Point(5, 5)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="CRS mismatch"):
        ci.land_use_from_other_gdf(
            _buildings(), others, new_land_use_column="land_uses", other_land_use_column="lu"
        )


def test_land_use_from_other_gdf_raises_on_empty_other_gdf():
    others = gpd.GeoDataFrame({"lu": []}, geometry=[], crs=CRS)
    with pytest.raises(ValueError, match="empty"):
        ci.land_use_from_other_gdf(
            _buildings(), others, new_land_use_column="land_uses", other_land_use_column="lu"
        )


def test_land_use_from_other_gdf_raises_on_mixed_geometry_types():
    mixed = gpd.GeoDataFrame(
        {"lu": ["retail", "office"]},
        geometry=[Point(5, 5), _square(20, 0)],
        crs=CRS,
    )
    with pytest.raises(ValueError, match="Unsupported/mixed geometry types"):
        ci.land_use_from_other_gdf(
            _buildings(), mixed, new_land_use_column="land_uses", other_land_use_column="lu"
        )
