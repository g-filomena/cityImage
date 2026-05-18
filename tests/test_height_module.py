"""Tests for the hard height split.

The vector height-transfer helper should remain available without importing
raster dependencies. Raster-backed functions live in the same module but import
rasterio/rasterstats lazily only when called.
"""

from __future__ import annotations

import sys

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

import cityImage as ci


def _building_fixture():
    crs = "EPSG:3857"
    buildings = gpd.GeoDataFrame(
        {"buildingID": [1, 2]},
        geometry=[
            Polygon([(0, 0), (4, 0), (4, 4), (0, 4)]),
            Polygon([(10, 0), (14, 0), (14, 4), (10, 4)]),
        ],
        crs=crs,
    )
    detailed = gpd.GeoDataFrame(
        {"base": [2.0, 5.0], "height": [12.0, 20.0]},
        geometry=[
            Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]),
            Polygon([(10, 0), (14, 0), (14, 4), (10, 4)]),
        ],
        crs=crs,
    )
    return buildings, detailed, crs


def test_height_module_import_does_not_require_raster_extras():
    sys.modules.pop("rasterio", None)
    sys.modules.pop("rasterstats", None)

    import cityImage.height as height

    assert hasattr(height, "assign_building_heights_from_other_gdf")
    assert "rasterio" not in sys.modules
    assert "rasterstats" not in sys.modules


def test_assign_building_heights_from_other_gdf_preserves_vector_height_transfer():
    buildings, detailed, crs = _building_fixture()

    out = ci.assign_building_heights_from_other_gdf(buildings, detailed, crs)
    out = out.sort_values("buildingID")

    assert out["base"].tolist() == pytest.approx([2.0, 5.0])
    assert out["height"].tolist() == pytest.approx([12.0, 20.0])


def test_assign_building_heights_from_other_gdf_rejects_crs_mismatch():
    buildings, detailed, _ = _building_fixture()
    detailed = detailed.to_crs("EPSG:4326")

    with pytest.raises(ValueError, match="CRS mismatch"):
        ci.assign_building_heights_from_other_gdf(buildings, detailed, buildings.crs)
