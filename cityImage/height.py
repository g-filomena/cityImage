"""Building and node elevation/height helpers.

This module keeps height-related capability outside the lightweight core import
path. Raster-backed functions import ``rasterio`` and ``rasterstats`` lazily,
inside the functions that need them, so ``import cityImage`` and
``import cityImage.height`` do not require raster extras.

Terminology (this module follows common GIS usage, with one caveat):

- **DTM** (Digital Terrain Model): bare-earth *terrain* elevation — buildings and
  vegetation stripped. This is what gives observer nodes their ``z`` and building
  footprints their ground ``base``.
- **DSM** (Digital Surface Model): first-return *surface* elevation — including
  rooftops and canopy. Comparing it against the DTM yields above-ground building
  heights (``height = top-of-surface − ground``).
- **DEM** (Digital Elevation Model): strictly the umbrella term for both. In
  :func:`buildings_height_from_dem_dtm` the ``dem_path`` argument plays the **DSM
  role** (the surface with the roofs on it); the name is kept for backwards
  compatibility.

What you need for what:

===========================  =========================================
You have                     You can derive
===========================  =========================================
DTM only                     node ``z``, building ``base`` (no heights)
DTM + DSM/DEM                node ``z``, building ``base`` **and** ``height``
detailed building layer      building ``height`` (and ``base`` if it carries one)
===========================  =========================================

:func:`assign_elevations_from_rasters` is the one-stop entry point covering the
raster rows of that table — useful for cities without any building-height data.
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

    ``dem_path`` must point at a **surface** model (DSM: first-return elevations
    including rooftops); ``dtm_path`` at the bare-earth **terrain** model. Per
    building, ``base`` is the ``base_stat`` of the terrain under the footprint and
    ``height`` is the ``top_stat`` of the surface minus that base — i.e. an
    above-ground height, not an absolute elevation.

    .. warning::
       Buildings whose footprint does not intersect the raster extent are **dropped
       from the returned frame** (and an error is raised when none intersect). Use
       :func:`assign_elevations_from_rasters` for a non-destructive variant that
       merges results back onto the full input frame.

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

    The raster must be a bare-earth **terrain** model (DTM): the sampled value is
    written to ``z_col`` and represents the ground under the node — e.g. the
    observer elevation used by the 3D sight lines. Values below ``min_valid_elev``
    (nodata seas, voids) become NaN for the caller to fill.

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


def buildings_base_from_dtm(
    buildings_gdf,
    dtm_path,
    base_stat="mean",
    all_touched=False,
    min_valid_elev=-50.0,
):
    """Assign per-building ground elevation (``base``) from a bare-earth DTM.

    The DTM-only counterpart of :func:`buildings_height_from_dem_dtm`: it gives every
    footprint the ``base_stat`` of the terrain under it, without touching ``height``.
    Use it when a city has building heights from another source (detailed layer, OSM
    tags) but the terrain relief should still feed the 3D sight lines.

    Non-destructive: buildings outside the raster extent keep their row, with
    ``base`` left as NaN.

    Requires the optional ``height`` extra: ``rasterio`` and ``rasterstats``.
    """
    rasterio, zonal_stats = _require_raster_deps()

    buildings_gdf = buildings_gdf.copy()
    original_crs = buildings_gdf.crs
    if original_crs is None:
        raise ValueError("buildings_gdf has no CRS set.")

    with rasterio.open(dtm_path) as dtm_src:
        working = buildings_gdf
        if working.crs != dtm_src.crs:
            working = working.to_crs(dtm_src.crs)

        raster_geom = box(*dtm_src.bounds)
        inside = working[working.intersects(raster_geom)]
        if inside.empty:
            raise ValueError("No buildings intersect the DTM extent.")

        dtm_data = np.ma.masked_less(dtm_src.read(1, masked=True), min_valid_elev)
        dtm_transform = dtm_src.transform

    stats = zonal_stats(
        inside.geometry.buffer(0),
        dtm_data,
        affine=dtm_transform,
        stats=[base_stat],
        all_touched=all_touched,
        geojson_out=False,
        nodata=None,
    )
    base_values = pd.Series(
        [record[base_stat] for record in stats], index=inside.index, dtype=float
    )

    buildings_gdf["base"] = base_values.reindex(buildings_gdf.index)
    return buildings_gdf


def assign_elevations_from_rasters(
    nodes_gdf,
    buildings_gdf,
    dtm_path,
    surface_path=None,
    z_col="z",
    base_stat="mean",
    top_stat="max",
    all_touched=False,
    min_valid_elev=-50.0,
):
    """One-stop elevation assignment from rasters, for cities without height data.

    Given a bare-earth **DTM** (terrain) and optionally a **surface** model
    (DSM — first-return elevations including rooftops; the raster often shipped
    under the generic name "DEM"), this assigns:

    - node ``z`` (``z_col``): terrain elevation sampled at each point — the observer
      elevation used by the 3D sight lines;
    - building ``base``: terrain elevation under each footprint;
    - building ``height`` (only when ``surface_path`` is given): above-ground height
      as top-of-surface minus base.

    With a DTM alone, heights are *not* derivable — only ``z`` and ``base`` are set;
    supply heights from a detailed layer or OSM tags instead.

    Non-destructive: results are merged back onto the full input frames, so features
    outside the raster extent keep their rows (with NaN elevations). Either
    ``nodes_gdf`` or ``buildings_gdf`` may be ``None`` to skip that side.

    Returns
    -------
    tuple(GeoDataFrame or None, GeoDataFrame or None)
        ``(nodes_gdf, buildings_gdf)`` with elevations assigned.

    Requires the optional ``height`` extra: ``rasterio`` and ``rasterstats``.
    """
    nodes_out = nodes_gdf
    if nodes_gdf is not None:
        nodes_out = assign_height_from_dtm(
            nodes_gdf, dtm_path, z_col=z_col, min_valid_elev=min_valid_elev
        )

    buildings_out = buildings_gdf
    if buildings_gdf is not None:
        if surface_path is not None:
            scored = buildings_height_from_dem_dtm(
                buildings_gdf,
                surface_path,
                dtm_path,
                base_stat=base_stat,
                top_stat=top_stat,
                all_touched=all_touched,
                min_valid_elev=min_valid_elev,
            )
            # Merge back onto the full frame: buildings_height_from_dem_dtm drops
            # footprints outside the raster extent and re-keys its output by
            # buildingID, so the merge must go through buildingID, not the index.
            if "buildingID" in buildings_gdf.columns and "buildingID" in scored.columns:
                buildings_out = buildings_gdf.copy()
                for column in ("base", "height"):
                    mapping = pd.Series(scored[column].values, index=scored["buildingID"].values)
                    buildings_out[column] = buildings_out["buildingID"].map(mapping)
            else:
                # No stable key to merge back on: return the (filtered) scored frame.
                buildings_out = scored
        else:
            buildings_out = buildings_base_from_dtm(
                buildings_gdf,
                dtm_path,
                base_stat=base_stat,
                all_touched=all_touched,
                min_valid_elev=min_valid_elev,
            )

    return nodes_out, buildings_out
