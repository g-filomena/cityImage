"""Land-use taxonomy and scalar classification tests.

Merged from:
- test_landuse_package.py
- test_land_use_compat.py

Sparse/multi-label land-use behaviour intentionally lives in
``test_landuse_sparse.py``.
"""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Polygon

import cityImage as ci
from cityImage import landuse


def _scalar_landuse_gdf():
    return gpd.GeoDataFrame(
        {"lu_eng": ["church", "bank", "unknown"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])] * 3,
        crs="EPSG:3857",
    )


def test_landuse_package_exports_previous_public_classification_api():
    assert landuse.UNCLASSIFIED == ci.UNCLASSIFIED

    out = landuse.classify_land_use(
        _scalar_landuse_gdf(),
        raw_land_use_column="lu_eng",
        new_land_use_column="land_use",
        categories=[["church"], ["bank"]],
        strings=["religious", "business_services"],
    )

    assert out["land_use"].tolist() == ["religious", "business_services", "unknown"]


def test_classify_land_use_sparse_attribute_wrapper_scalar_values():
    """The wrapper maps non-OSM scalar attribute values."""
    out = ci.classify_land_use(
        _scalar_landuse_gdf(),
        raw_land_use_column="lu_eng",
        new_land_use_column="land_use",
        categories=[["church"], ["bank"]],
        strings=["religious", "business_services"],
    )

    assert out["land_use"].tolist() == ["religious", "business_services", "unknown"]


def test_landuse_constants_and_utils_are_available_from_new_package_namespace():
    assert "amenity" in landuse.OSM_DOMAINS
    assert any("place_of_worship" in values for values in landuse.AMENITY_GROUPS.values())

    buildings = gpd.GeoDataFrame(
        {"land_uses": [["amenity:church:religious"], ["shop:bakery:retail"], []]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            Polygon([(4, 0), (5, 0), (5, 1), (4, 1)]),
        ],
        crs="EPSG:3857",
    )

    matches = landuse.find_land_use_values_matching(
        buildings,
        land_uses_column="land_uses",
        pattern="church",
        return_counts=False,
    )

    assert matches == ["amenity:church:religious"]
