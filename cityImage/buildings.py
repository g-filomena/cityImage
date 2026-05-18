"""Building GeoDataFrame helpers for cityImage.

Live building acquisition is delegated to OSMnx and file IO is delegated to
GeoPandas. This module keeps only small schema/selection helpers that preserve
cityImage's downstream semantics.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd


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
