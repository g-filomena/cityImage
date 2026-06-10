"""Sparse/multi-label land-use tests.

Merged from:
- test_sparse_land_uses.py
- test_regression_land_use_semantics.py
- sparse top-level lazy API assertion from test_landuse_package.py
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

import cityImage as ci
from tests.fixtures.cityimage_minimal import (
    minimal_buildings,
    raw_sparse_buildings,
    sparse_land_use_polygons,
)

LAND_USE_MAPPING = {
    "shop": "retail",
    "school": "education",
}


def test_cityimage_top_level_sparse_landuse_api_resolves_via_lazy_loading():
    buildings = gpd.GeoDataFrame(
        {"lu_eng": ["church", "bank", None]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            Polygon([(4, 0), (5, 0), (5, 1), (4, 1)]),
        ],
        crs="EPSG:3857",
    )

    out = ci.classify_sparse_land_uses(
        buildings,
        source_column="lu_eng",
        mapping={
            "church": "religious",
            "bank": "financial",
        },
        default_land_use="unknown",
    )

    assert out["land_uses"].tolist() == [["religious"], ["financial"], ["unknown"]]


def test_classify_sparse_land_uses_from_non_osm_column():
    buildings = gpd.GeoDataFrame(
        {"lu_eng": ["church", "bank", None]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            Polygon([(4, 0), (5, 0), (5, 1), (4, 1)]),
        ],
        crs="EPSG:3857",
    )

    out = ci.classify_sparse_land_uses(
        buildings,
        source_column="lu_eng",
        mapping={
            "church": "religious",
            "bank": "financial",
        },
        default_land_use="unknown",
    )

    assert out["land_uses"].tolist() == [["religious"], ["financial"], ["unknown"]]


def test_attach_sparse_land_uses_from_external_polygons():
    buildings = gpd.GeoDataFrame(
        {"buildingID": [1, 2]},
        geometry=[
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(3, 0), (5, 0), (5, 2), (3, 2)]),
        ],
        crs="EPSG:3857",
    )

    sparse_land_uses = gpd.GeoDataFrame(
        {"lu_eng": ["shop", "office"]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 2), (0, 2)]),
            Polygon([(1, 0), (2, 0), (2, 2), (1, 2)]),
        ],
        crs="EPSG:3857",
    )

    out = ci.attach_sparse_land_uses(
        buildings,
        sparse_land_uses,
        source_column="lu_eng",
        mapping={
            "shop": "retail",
            "office": "office",
        },
        default_land_use="unknown",
    )

    assert out.loc[0, "land_uses"] == ["retail", "office"]
    assert out.loc[0, "land_uses_overlap"] == [0.5, 0.5]
    assert out.loc[1, "land_uses"] == ["unknown"]
    assert out.loc[1, "land_uses_overlap"] == [1.0]


def test_sparse_classification_preserves_order_deduplicates_and_default():
    classified = ci.classify_sparse_land_uses(
        raw_sparse_buildings(),
        source_column="raw",
        mapping=LAND_USE_MAPPING,
        default_land_use="residential",
        preserve_unmapped=False,
    )

    assert classified["land_uses"].tolist() == [
        ["retail", "education"],
        ["residential"],
        ["residential"],
    ]


def test_sparse_polygon_attachment_preserves_labels_and_normalised_overlap():
    attached = ci.attach_sparse_land_uses(
        minimal_buildings(),
        sparse_land_use_polygons(),
        source_column="raw_use",
        mapping=LAND_USE_MAPPING,
        default_land_use="residential",
    )

    rows = attached.sort_values("buildingID")[
        [
            "buildingID",
            "land_uses",
            "land_uses_overlap",
        ]
    ].to_dict("records")

    assert rows[0]["buildingID"] == 1
    assert rows[0]["land_uses"] == ["retail", "education"]
    assert rows[0]["land_uses_overlap"] == pytest.approx([0.5, 0.5])

    assert rows[1]["buildingID"] == 2
    assert rows[1]["land_uses"] == ["retail"]
    assert rows[1]["land_uses_overlap"] == pytest.approx([1.0])

    assert rows[2]["buildingID"] == 3
    assert rows[2]["land_uses"] == ["residential"]
    assert rows[2]["land_uses_overlap"] == pytest.approx([1.0])
