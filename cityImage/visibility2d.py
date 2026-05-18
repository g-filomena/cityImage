"""Lightweight 2D visibility helpers for cityImage.

This module contains the core 2D visibility primitive used by landmark
structural scoring. It deliberately avoids heavy 3D/runtime dependencies such
as PyVista, Dask, psutil, and tqdm.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import unary_union

from .angles import get_coord_angle


def visibility_polygon2d(
    building_geometry,
    obstructions_gdf,
    obstructions_sindex=None,
    max_expansion_distance=600,
):
    """Return the area of the 2D advance-visibility polygon around a building.

    Parameters
    ----------
    building_geometry
        Building footprint geometry.
    obstructions_gdf
        GeoDataFrame containing obstruction polygons.
    obstructions_sindex
        Accepted for backward compatibility with the previous signature. The
        current implementation uses vectorised GeoPandas/Shapely operations.
    max_expansion_distance
        Maximum radial distance for visibility rays.

    Returns
    -------
    float
        Area of the visible polygon after subtracting the building footprint.
    """
    del obstructions_sindex  # kept only to preserve the old call signature

    distance_along = 10
    origin = building_geometry.centroid
    building_geometry = (
        building_geometry.convex_hull
        if building_geometry.geom_type == "MultiPolygon"
        else building_geometry
    )
    max_expansion_distance += origin.distance(building_geometry.envelope.exterior)

    angles = np.arange(0, 360, distance_along)
    coords = np.array(
        [
            get_coord_angle([origin.x, origin.y], distance=max_expansion_distance, angle=i)
            for i in angles
        ]
    )
    lines = [LineString([origin, Point(x)]) for x in coords]

    obstacles = obstructions_gdf[obstructions_gdf.crosses(unary_union(lines))]
    obstacles = obstacles[obstacles.geometry != building_geometry]
    obstacles = obstacles[~obstacles.geometry.within(building_geometry.convex_hull)]

    if len(obstacles) > 0:
        obstruction_union = unary_union(obstacles.geometry)
        intersections = [line.intersection(obstruction_union) for line in lines]
        clipped_lines = [
            LineString([origin, Point(intersection.geoms[0].coords[0])])
            if isinstance(intersection, MultiLineString) and not intersection.is_empty
            else LineString([origin, Point(intersection.coords[0])])
            if isinstance(intersection, LineString) and not intersection.is_empty
            else LineString([origin, Point(intersection.coords[0])])
            if isinstance(intersection, Point) and not intersection.is_empty
            else line
            for intersection, line in zip(intersections, lines, strict=False)
        ]
    else:
        clipped_lines = lines

    polygon = Polygon(
        [[p.x, p.y] for p in [origin] + [Point(line.coords[1]) for line in clipped_lines] + [origin]]
    )
    visible_polygon = polygon.difference(building_geometry)
    if visible_polygon.is_empty:
        visible_polygon = polygon.buffer(0).difference(building_geometry)

    return visible_polygon.area
