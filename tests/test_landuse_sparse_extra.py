"""Additional offline tests for ``cityImage.landuse.sparse``.

Complements ``test_landuse_sparse.py`` by covering the point/line join branch,
mapping application, and the validation/early-return paths.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon

import cityImage as ci

CRS = "EPSG:3857"


def _buildings():
    return gpd.GeoDataFrame(
        {"buildingID": [1, 2]},
        geometry=[
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            Polygon([(20, 0), (30, 0), (30, 10), (20, 10)]),
        ],
        crs=CRS,
    )


def test_attach_sparse_from_points_uses_count_weights():
    points = gpd.GeoDataFrame(
        {"use": ["shop", "shop"]},
        geometry=[Point(5, 5), Point(6, 6)],  # both inside building 1
        crs=CRS,
    )

    out = ci.attach_sparse_land_uses(
        _buildings(), points, source_column="use", mapping={"shop": "retail"}
    )

    by_id = out.set_index("buildingID")
    assert by_id.loc[1, "land_uses"] == ["retail"]
    assert by_id.loc[1, "land_uses_overlap"] == [1.0]
    assert by_id.loc[2, "land_uses"] == ["unknown"]  # untouched default


def test_classify_sparse_land_uses_requires_source_column():
    with pytest.raises(ValueError, match="must contain"):
        ci.classify_sparse_land_uses(_buildings(), source_column="missing")


def test_attach_sparse_requires_source_column():
    with pytest.raises(ValueError, match="must contain"):
        ci.attach_sparse_land_uses(
            _buildings(),
            gpd.GeoDataFrame({"x": [1]}, geometry=[Point(5, 5)], crs=CRS),
            source_column="use",
        )


def test_attach_sparse_rejects_crs_mismatch():
    other = gpd.GeoDataFrame({"use": ["shop"]}, geometry=[Point(5, 5)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="CRS mismatch"):
        ci.attach_sparse_land_uses(_buildings(), other, source_column="use")


def test_attach_sparse_empty_land_uses_returns_defaults():
    empty = gpd.GeoDataFrame({"use": []}, geometry=[], crs=CRS)
    out = ci.attach_sparse_land_uses(_buildings(), empty, source_column="use")
    assert out["land_uses"].tolist() == [["unknown"], ["unknown"]]
    assert "_ci_sparse_row" not in out.columns


def test_attach_sparse_points_with_no_intersection_keep_defaults():
    far_points = gpd.GeoDataFrame(
        {"use": ["shop"]},
        geometry=[Point(999, 999)],
        crs=CRS,  # outside every building
    )
    out = ci.attach_sparse_land_uses(_buildings(), far_points, source_column="use")
    assert out["land_uses"].tolist() == [["unknown"], ["unknown"]]


def test_attach_sparse_polygons_with_no_overlap_keep_defaults():
    far_poly = gpd.GeoDataFrame(
        {"use": ["shop"]},
        geometry=[Polygon([(900, 900), (910, 900), (910, 910), (900, 910)])],
        crs=CRS,
    )
    out = ci.attach_sparse_land_uses(_buildings(), far_poly, source_column="use")
    assert out["land_uses"].tolist() == [["unknown"], ["unknown"]]


def test_attach_sparse_from_polygons_uses_area_weights():
    # A polygon covering building 1 and mapped to "retail" -> that label with full overlap.
    poly = gpd.GeoDataFrame(
        {"use": ["shop"]},
        geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])],
        crs=CRS,
    )
    out = ci.attach_sparse_land_uses(
        _buildings(), poly, source_column="use", mapping={"shop": "retail"}
    )
    by_id = out.set_index("buildingID")
    assert by_id.loc[1, "land_uses"] == ["retail"]
    assert abs(sum(by_id.loc[1, "land_uses_overlap"]) - 1.0) < 1e-9


def test_classify_sparse_unmapped_falls_back_to_default_when_not_preserved():
    buildings = _buildings()
    buildings["use"] = ["mystery", "shop"]
    out = ci.classify_sparse_land_uses(
        buildings,
        source_column="use",
        mapping={"shop": "retail"},
        preserve_unmapped=False,
        default_land_use="unknown",
    )
    by_id = out.set_index("buildingID")["land_uses"]
    assert by_id[1] == ["unknown"]  # 'mystery' unmapped -> default
    assert by_id[2] == ["retail"]
