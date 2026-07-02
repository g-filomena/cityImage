"""Building and node elevation/height helpers.

This module keeps height-related capability outside the lightweight core import
path. Raster-backed functions import ``rasterio`` and ``rasterstats`` lazily,
inside the functions that need them, so ``import cityImage`` and
``import cityImage.height`` do not require raster extras.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import box

from .geometry import gdf_multipolygon_to_polygon


def _require_raster_deps():
    """Import the optional raster stack, with a helpful error if it is missing."""
    try:
        import rasterio
        from rasterstats import zonal_stats
    except ImportError as exc:
        raise ImportError(
            "This height operation requires the optional 'height' extra "
            '(rasterio, rasterstats). Install with: python -m pip install -e ".[height]"'
        ) from exc
    return rasterio, zonal_stats


def assign_building_heights_from_other_gdf(
    buildings_gdf,
    detailed_buildings_gdf,
    crs,
    base_field="base",
    height_field="height",
    min_overlap=0.4,
):
    """Assign base and height attributes from a detailed building layer.

    This is a vector-only helper: it does not require raster dependencies. It is
    kept because it preserves a useful cityImage data-preparation capability
    while avoiding OSM/raster ownership.
    """
    if (buildings_gdf.crs != crs) or (detailed_buildings_gdf.crs != crs):
        raise ValueError(
            "CRS mismatch: buildings_gdf "
            f"({buildings_gdf.crs}) and detailed_buildings_gdf "
            f"({detailed_buildings_gdf.crs}) must have the same CRS."
        )

    buildings_gdf = buildings_gdf.copy()
    detailed_buildings_gdf = gdf_multipolygon_to_polygon(detailed_buildings_gdf)

    detailed_buildings_gdf["base"] = detailed_buildings_gdf[base_field]
    detailed_buildings_gdf["height"] = detailed_buildings_gdf[height_field]

    buildings_gdf["base"] = 9999.0
    buildings_gdf["height"] = -9999.0

    # 1. Main buildings containing detailed buildings.
    containment = gpd.sjoin(buildings_gdf, detailed_buildings_gdf, predicate="contains", how="left")
    contained_bases = containment.groupby(containment.index)["base_right"].min()
    contained_height = containment.groupby(containment.index)["height_right"].max()

    buildings_gdf["base"] = buildings_gdf["base"].combine(contained_bases, min)
    buildings_gdf["height"] = buildings_gdf["height"].combine(contained_height, max)
    buildings_gdf.index = buildings_gdf.index.astype(int)

    # 2. Detailed buildings containing main buildings.
    reverse_containment = gpd.sjoin(
        detailed_buildings_gdf,
        buildings_gdf,
        predicate="contains",
        how="left",
    )
    container_bases = reverse_containment.groupby("index_right")["base_left"].min()
    container_height = reverse_containment.groupby("index_right")["height_left"].max()

    buildings_gdf["base"] = buildings_gdf["base"].combine(container_bases, min)
    buildings_gdf["height"] = buildings_gdf["height"].combine(container_height, max)
    buildings_gdf.index = buildings_gdf.index.astype(int)

    # 3. Detailed buildings partially overlapping main buildings.
    buildings_gdf["geo_check"] = buildings_gdf.geometry
    buildings_gdf["ix"] = buildings_gdf.index

    intersections = gpd.sjoin(
        detailed_buildings_gdf,
        buildings_gdf,
        predicate="intersects",
        how="left",
    )
    intersections = intersections[intersections["geo_check"].notna()]

    intersections["area_intersection"] = intersections.apply(
        lambda row: row["geometry"].intersection(row["geo_check"]).area,
        axis=1,
    )
    intersections["overlap_ratio"] = (
        intersections["area_intersection"] / intersections["geometry"].area
    )

    valid_matches = intersections[intersections["overlap_ratio"] >= min_overlap]
    best_matches = valid_matches.loc[
        valid_matches.groupby(valid_matches.index)["overlap_ratio"].idxmax()
    ]
    best_matches = best_matches.set_index("ix")
    best_matches.index = best_matches.index.astype(int)

    intersection_bases = best_matches.groupby(best_matches.index)["base_left"].min()
    intersection_height = best_matches.groupby(best_matches.index)["height_left"].max()

    buildings_gdf["base"] = buildings_gdf["base"].combine(intersection_bases, min)
    buildings_gdf["height"] = buildings_gdf["height"].combine(intersection_height, max)
    buildings_gdf = buildings_gdf.drop(["geo_check", "ix"], axis=1)
    buildings_gdf.index = buildings_gdf.index.astype(int)

    # 4. Reverse intersections: main buildings borrow from overlapping detailed buildings.
    detailed_buildings_gdf["geo_check"] = detailed_buildings_gdf.geometry
    buildings_gdf["ix"] = buildings_gdf.index

    intersections = gpd.sjoin(
        buildings_gdf,
        detailed_buildings_gdf,
        predicate="intersects",
        how="left",
    )
    intersections = intersections[intersections.geo_check.notnull()]

    intersections["area_intersection"] = intersections.apply(
        lambda row: row["geometry"].intersection(row["geo_check"]).area,
        axis=1,
    )
    intersections["overlap_ratio"] = (
        intersections["area_intersection"] / intersections["geometry"].area
    )

    valid_matches = intersections[intersections["overlap_ratio"] >= min_overlap]
    best_matches = valid_matches.loc[
        valid_matches.groupby(valid_matches.index)["overlap_ratio"].idxmax()
    ]

    intersection_bases = best_matches.groupby(best_matches.index)["base_right"].min()
    intersection_height = best_matches.groupby(best_matches.index)["height_right"].max()

    buildings_gdf["base"] = buildings_gdf["base"].combine(intersection_bases, min)
    buildings_gdf["height"] = buildings_gdf["height"].combine(intersection_height, max)
    buildings_gdf.index = buildings_gdf.index.astype(int)

    buildings_gdf["base"] = buildings_gdf["base"].replace(9999.0, np.nan)
    buildings_gdf["height"] = buildings_gdf["height"].replace(-9999.0, np.nan)
    buildings_gdf = buildings_gdf.drop(["geo_check", "ix"], axis=1, errors="ignore")

    return buildings_gdf


def buildings_height_from_dem_dtm(
    buildings_gdf,
    dem_path,
    dtm_path,
    base_stat="mean",
    top_stat="max",
    all_touched=False,
    min_valid_elev=-50,
):
    """Compute per-building base elevation and height from DEM and DTM rasters.

    Requires the optional ``height`` extra: ``rasterio`` and ``rasterstats``.
    """
    rasterio, zonal_stats = _require_raster_deps()

    original_crs = buildings_gdf.crs
    buildings_with_data = buildings_gdf.copy()

    with rasterio.open(dem_path) as dem_src, rasterio.open(dtm_path) as dtm_src:
        dem_crs = dem_src.crs
        dtm_crs = dtm_src.crs

        if dem_crs != dtm_crs:
            raise ValueError("DEM and DTM have different CRS; reproject one of them beforehand.")

        if buildings_with_data.crs is None:
            raise ValueError("buildings_gdf has no CRS set.")
        if buildings_with_data.crs != dem_crs:
            buildings_with_data = buildings_with_data.to_crs(dem_crs)

        raster_geom = box(*dem_src.bounds)
        buildings_with_data = buildings_with_data[buildings_with_data.intersects(raster_geom)]

        if buildings_with_data.empty:
            raise ValueError("No buildings intersect the DEM/DTM extent.")

        dem_data = np.ma.masked_less(dem_src.read(1, masked=True), min_valid_elev)
        dtm_data = np.ma.masked_less(dtm_src.read(1, masked=True), min_valid_elev)

        dem_transform = dem_src.transform
        dtm_transform = dtm_src.transform

    buildings_with_data["geometry"] = buildings_with_data.geometry.buffer(0)

    dem_stats = zonal_stats(
        buildings_with_data,
        dem_data,
        affine=dem_transform,
        stats=[top_stat],
        all_touched=all_touched,
        geojson_out=False,
        nodata=None,
    )
    dem_df = pd.DataFrame(dem_stats)
    dem_col = f"dem_{top_stat}"
    dem_df.columns = [dem_col]

    dtm_stats = zonal_stats(
        buildings_with_data,
        dtm_data,
        affine=dtm_transform,
        stats=[base_stat],
        all_touched=all_touched,
        geojson_out=False,
        nodata=None,
    )
    dtm_df = pd.DataFrame(dtm_stats)
    dtm_col = f"dtm_{base_stat}"
    dtm_df.columns = [dtm_col]

    buildings_with_data = buildings_with_data.reset_index(drop=True)
    buildings_with_data = pd.concat([buildings_with_data, dem_df, dtm_df], axis=1)

    buildings_with_data["base"] = buildings_with_data[dtm_col]
    buildings_with_data["height"] = buildings_with_data[dem_col] - buildings_with_data["base"]
    buildings_with_data = buildings_with_data.drop(columns=[dem_col, dtm_col])

    if buildings_with_data.crs != original_crs:
        buildings_with_data = buildings_with_data.to_crs(original_crs)

    if "buildingID" in buildings_with_data.columns:
        buildings_with_data.index = buildings_with_data.buildingID
        buildings_with_data.index.name = None

    return buildings_with_data


def assign_height_from_dtm(
    nodes_gdf: gpd.GeoDataFrame,
    dtm_path: str,
    z_col: str = "z",
    min_valid_elev: float = -50.0,
):
    """Sample a DTM raster to assign elevation to point nodes.

    Requires the optional ``height`` extra: ``rasterio``.
    """
    rasterio, _ = _require_raster_deps()

    nodes_gdf_with_data = nodes_gdf.copy()
    original_crs = nodes_gdf.crs

    with rasterio.open(dtm_path) as src:
        dtm_crs = src.crs
        nodata = src.nodata

        if dtm_crs is None:
            raise ValueError("DTM has no CRS set.")
        if nodes_gdf_with_data.crs != dtm_crs:
            nodes_gdf_with_data = nodes_gdf_with_data.to_crs(dtm_crs)

        if not all(geom.geom_type == "Point" for geom in nodes_gdf_with_data.geometry):
            raise ValueError("All geometries in nodes_gdf must be Points.")

        coords = [(geom.x, geom.y) for geom in nodes_gdf_with_data.geometry]
        sampled = list(src.sample(coords))
        elev = np.array([vals[0] if len(vals) > 0 else np.nan for vals in sampled], dtype=float)

        if nodata is not None:
            elev[elev == nodata] = np.nan
        elev[elev < min_valid_elev] = np.nan

        nodes_gdf_with_data[z_col] = elev

    if nodes_gdf_with_data.crs != original_crs:
        nodes_gdf_with_data = nodes_gdf_with_data.to_crs(original_crs)

    return nodes_gdf_with_data
