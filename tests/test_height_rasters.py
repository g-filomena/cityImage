"""Offline tests for the raster-backed helpers in ``cityImage.height``.

Small synthetic GeoTIFFs (a flat terrain DTM and a raised surface DSM) are
written to a temp dir so the rasterio/rasterstats paths run without real data.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import Point, Polygon

import cityImage as ci

rasterio = pytest.importorskip("rasterio")
pytest.importorskip("rasterstats")
from rasterio.transform import from_origin  # noqa: E402

CRS = "EPSG:3857"


def _write_raster(path, value, *, crs=CRS, size=20):
    transform = from_origin(0, size, 1.0, 1.0)  # covers x[0,size], y[0,size]
    data = np.full((size, size), value, dtype="float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)
    return str(path)


@pytest.fixture
def rasters(tmp_path):
    dtm = _write_raster(tmp_path / "dtm.tif", 10.0)  # bare-earth terrain
    dsm = _write_raster(tmp_path / "dsm.tif", 25.0)  # surface incl. rooftops
    return dtm, dsm


def _buildings(crs=CRS):
    return gpd.GeoDataFrame(
        {"buildingID": [1, 2]},
        geometry=[
            Polygon([(1, 1), (4, 1), (4, 4), (1, 4)]),
            Polygon([(10, 10), (14, 10), (14, 14), (10, 14)]),
        ],
        crs=crs,
    )


def _nodes():
    return gpd.GeoDataFrame({"nodeID": [1, 2]}, geometry=[Point(2, 2), Point(12, 12)], crs=CRS)


def test_buildings_height_from_dem_dtm(rasters):
    dtm, dsm = rasters
    out = ci.buildings_height_from_dem_dtm(_buildings(), dsm, dtm).sort_values("buildingID")
    assert out["base"].tolist() == pytest.approx([10.0, 10.0])
    assert out["height"].tolist() == pytest.approx([15.0, 15.0])  # 25 surface - 10 terrain


def test_assign_height_from_dtm_samples_terrain(rasters):
    dtm, _ = rasters
    out = ci.assign_height_from_dtm(_nodes(), dtm)
    assert out["z"].tolist() == pytest.approx([10.0, 10.0])


def test_buildings_base_from_dtm(rasters):
    dtm, _ = rasters
    out = ci.buildings_base_from_dtm(_buildings(), dtm).sort_values("buildingID")
    assert out["base"].tolist() == pytest.approx([10.0, 10.0])


def test_assign_elevations_from_rasters_with_and_without_surface(rasters):
    dtm, dsm = rasters
    nodes_out, buildings_out = ci.assign_elevations_from_rasters(_nodes(), _buildings(), dtm, dsm)
    assert nodes_out["z"].tolist() == pytest.approx([10.0, 10.0])
    buildings_out = buildings_out.sort_values("buildingID")
    assert buildings_out["base"].tolist() == pytest.approx([10.0, 10.0])
    assert buildings_out["height"].tolist() == pytest.approx([15.0, 15.0])

    # Without a surface model only base is derivable (no height).
    _, base_only = ci.assign_elevations_from_rasters(None, _buildings(), dtm, surface_path=None)
    assert base_only["base"].tolist() == pytest.approx([10.0, 10.0])


def test_assign_height_from_dtm_rejects_non_point_geometries(rasters):
    dtm, _ = rasters
    with pytest.raises(ValueError, match="must be Points"):
        ci.assign_height_from_dtm(_buildings(), dtm)


def test_buildings_height_requires_matching_raster_crs(rasters, tmp_path):
    dtm, _ = rasters
    dsm_4326 = _write_raster(tmp_path / "dsm_4326.tif", 25.0, crs="EPSG:4326")
    with pytest.raises(ValueError, match="different CRS"):
        ci.buildings_height_from_dem_dtm(_buildings(), dsm_4326, dtm)


def test_buildings_height_requires_intersection_and_crs(rasters):
    dtm, dsm = rasters
    far = gpd.GeoDataFrame(
        {"buildingID": [1]},
        geometry=[Polygon([(1000, 1000), (1004, 1000), (1004, 1004), (1000, 1004)])],
        crs=CRS,
    )
    with pytest.raises(ValueError, match="intersect"):
        ci.buildings_height_from_dem_dtm(far, dsm, dtm)

    no_crs = _buildings(crs=None)
    with pytest.raises(ValueError, match="no CRS"):
        ci.buildings_height_from_dem_dtm(no_crs, dsm, dtm)
