"""Building GeoDataFrame helpers for cityImage.

Live building acquisition is delegated to OSMnx and file IO is delegated to
GeoPandas. This module keeps only small schema/selection helpers that preserve
cityImage's downstream semantics.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd


def _geometry_union(geometry: gpd.GeoSeries) -> Any:
    """Return a geometry union compatible with older/newer GeoPandas versions."""
    try:
        return geometry.union_all()
    except AttributeError:
        return geometry.unary_union


def select_buildings_by_study_area(
    larger_buildings_gdf: gpd.GeoDataFrame,
    *,
    method: str = "polygon",
    polygon: Any = None,
    distance: float = 1000,
) -> gpd.GeoDataFrame:
    """Select buildings within a polygon or centroid-distance study area.

    Use GeoPandas/OSMnx to acquire buildings, ``standardize_buildings_gdf`` to
    normalise schema, then this helper if a cityImage-style study-area subset is
    needed.
    """
    if larger_buildings_gdf.empty:
        return gpd.GeoDataFrame(
            columns=larger_buildings_gdf.columns,
            geometry=larger_buildings_gdf.geometry.name
            if hasattr(larger_buildings_gdf, "geometry")
            else None,
            crs=getattr(larger_buildings_gdf, "crs", None),
        )

    if method == "distance":
        study_area = _geometry_union(larger_buildings_gdf.geometry).centroid.buffer(distance)
    elif method == "polygon":
        study_area = polygon
    else:
        raise ValueError("method must be either 'polygon' or 'distance'")

    if study_area is None:
        return gpd.GeoDataFrame(
            columns=larger_buildings_gdf.columns,
            geometry=larger_buildings_gdf.geometry.name,
            crs=larger_buildings_gdf.crs,
        )

    return larger_buildings_gdf[larger_buildings_gdf.geometry.within(study_area)].copy()


def filter_buildings_by_height(
    buildings_gdf: gpd.GeoDataFrame,
    min_height: float = 5,
    height_column: str = "height",
) -> gpd.GeoDataFrame:
    """Drop buildings below ``min_height`` when the layer carries real heights.

    When the layer's mean known height is above ``min_height``, buildings with a height below it
    are dropped, and so are buildings whose height is missing or zero: a layer mixing known and
    unknown heights would otherwise give the unknown ones a NaN landmark score. A layer without
    heights (no column, all missing, or values that look like floor counts, with a mean at or
    below ``min_height``) is returned unchanged, and the visual component is then left out of the
    scores for every building.

    Parameters
    ----------
    buildings_gdf : geopandas.GeoDataFrame
        Buildings table.
    min_height : float, default 5
        Minimum height, in metres, of a building kept when the layer has heights.
    height_column : str, default "height"
        Column holding the heights; values are read as numbers, unparseable ones as missing.

    Returns
    -------
    geopandas.GeoDataFrame
        The kept buildings, a copy of the input rows.
    """
    buildings = buildings_gdf.copy()
    if height_column not in buildings.columns:
        return buildings
    heights = pd.to_numeric(buildings[height_column], errors="coerce")
    mean_height = heights.mean(skipna=True)
    if pd.isna(mean_height) or mean_height <= min_height:
        return buildings
    return buildings[heights.notna() & (heights >= min_height)].copy()
