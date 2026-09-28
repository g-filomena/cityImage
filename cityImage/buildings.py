"""Building GeoDataFrame helpers for cityImage.

Live building acquisition is delegated to OSMnx and file IO is delegated to
GeoPandas. This module keeps only small schema/selection helpers that preserve
cityImage's downstream semantics.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import geopandas as gpd
import pandas as pd

LOGGER = logging.getLogger(__name__)


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


def parse_height(value: Any) -> float | None:
    """Read a height in metres: the first number in the value, or None when it holds none.

    Accepts numbers, strings such as ``"12 m"`` or ``"12,5"`` (as OSM tags carry them), and
    list-like values, of which the first item is read.
    """
    if isinstance(value, (list, tuple, set)):
        value = next(iter(value), None)
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"-?\d+(\.\d+)?", str(value).replace(",", "."))
    return float(match.group()) if match else None


def known_heights(values: pd.Series) -> pd.Series:
    """Heights in metres (see ``parse_height``), NaN where unknown: missing, unreadable or not
    above zero. Every height the package uses is read through this: the building schema, the
    loaders, the landmark scores and the 3D sight lines.
    """
    heights = values.apply(parse_height).astype(float)
    return heights.where(heights > 0.0)


def _drop_buildings_below_height(
    buildings_gdf: gpd.GeoDataFrame, min_height: float
) -> gpd.GeoDataFrame:
    """Drop the buildings whose known height is lower than ``min_height``.

    A building without a height is kept, with a NaN height (see ``known_heights``).
    """
    heights = known_heights(buildings_gdf["height"])
    keep = ~(heights < min_height)
    dropped = int((~keep).sum())
    if dropped:
        LOGGER.info("Dropped %d building(s) lower than %s m", dropped, min_height)
    kept = buildings_gdf[keep].copy()
    kept["height"] = heights[keep]
    return kept
