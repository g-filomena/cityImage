"""Tests for hard visibility2d / visibility3d split."""

from __future__ import annotations

import sys

import geopandas as gpd
from shapely.geometry import Polygon

import cityImage as ci
from cityImage import visibility2d


def test_visibility_polygon2d_lives_in_lightweight_visibility2d_module():
    assert ci.visibility_polygon2d is visibility2d.visibility_polygon2d


def test_visibility_polygon2d_does_not_import_heavy_3d_stack():
    for module_name in ["pyvista", "dask", "psutil"]:
        sys.modules.pop(module_name, None)

    building = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
    obstructions = gpd.GeoDataFrame(
        geometry=[building, Polygon([(4, -1), (5, -1), (5, 3), (4, 3)])],
        crs="EPSG:3857",
    )

    area = ci.visibility_polygon2d(building, obstructions, obstructions.sindex, 20)

    assert area > 0
    assert "pyvista" not in sys.modules
    assert "dask" not in sys.modules
    assert "psutil" not in sys.modules
