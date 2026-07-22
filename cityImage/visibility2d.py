"""Lightweight 2D visibility helpers for cityImage.

This module contains the core 2D visibility primitive used by landmark
structural scoring. It deliberately avoids heavy 3D/runtime dependencies such
as Dask and psutil.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from .angles import get_coord_angle


def visibility_polygon2d(
    building_geometry,
    obstructions_gdf,
    obstructions_sindex=None,
    max_expansion_distance=600,
):
    """Return the area of the 2D advance-visibility polygon around a building.

    A ring of rays is cast outward from the building centroid, one every
    ``angular_step`` degrees (36 rays at the default 10 deg). Each ray reaches out
    to ``max_expansion_distance`` and is then shortened to the nearest obstruction
    it hits; connecting the (shortened) ray tips traces an isovist-like polygon
    whose area, minus the footprint, measures how much open space the building can
    be seen from.

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
    angular_step = 10  # degrees between rays -> 360 / 10 = 36 rays
    origin = building_geometry.centroid
    building_geometry = (
        building_geometry.convex_hull
        if building_geometry.geom_type == "MultiPolygon"
        else building_geometry
    )
    max_expansion_distance += origin.distance(building_geometry.envelope.exterior)

    angles = np.arange(0, 360, angular_step)
    coords = np.array(
        [
            get_coord_angle([origin.x, origin.y], distance=max_expansion_distance, angle=i)
            for i in angles
        ]
    )
    lines = [LineString([origin, Point(x)]) for x in coords]
    rays = unary_union(lines)

    # Spatial pre-filter: only obstructions whose bounding box meets the ray fan can be
    # crossed by it, so query the index and run the exact (comparatively costly) `crosses`
    # test on just those few candidates instead of the whole obstruction set. This is
    # identical to crossing every obstruction (the crossers are a subset of the bbox
    # candidates) but avoids the per-building full-table scan that dominated runtime. The
    # frame's own cached index is used, so `obstructions_sindex` stays a no-op accepted for
    # backward compatibility.
    del obstructions_sindex
    candidate_positions = obstructions_gdf.sindex.query(rays)
    candidates = obstructions_gdf.iloc[candidate_positions]

    obstacles = candidates[candidates.crosses(rays)]
    obstacles = obstacles[obstacles.geometry != building_geometry]
    obstacles = obstacles[~obstacles.geometry.within(building_geometry.convex_hull)]

    def _clip_to_nearest_obstruction(ray, obstruction_union):
        """Shorten one ray to the obstruction vertex nearest the origin.

        Works for every geometry type a line/polygon intersection can yield
        (LineString, MultiLineString, Point, MultiPoint, GeometryCollection) and
        returns the full ray when there is no real hit. Picking the nearest vertex
        explicitly avoids relying on the order in which GEOS returns the pieces (and,
        unlike the previous code, does not silently ignore point/collection hits).
        """
        intersection = ray.intersection(obstruction_union)
        if intersection.is_empty:
            return ray
        inter_coords = shapely.get_coordinates(intersection)
        if len(inter_coords) == 0:
            return ray
        squared_distance = (inter_coords[:, 0] - origin.x) ** 2 + (
            inter_coords[:, 1] - origin.y
        ) ** 2
        nearest = inter_coords[int(np.argmin(squared_distance))]
        return LineString([origin, Point(nearest)])

    if len(obstacles) > 0:
        obstruction_union = unary_union(obstacles.geometry)
        clipped_lines = [_clip_to_nearest_obstruction(line, obstruction_union) for line in lines]
    else:
        clipped_lines = lines

    polygon = Polygon(
        [
            [p.x, p.y]
            for p in [origin] + [Point(line.coords[1]) for line in clipped_lines] + [origin]
        ]
    )
    visible_polygon = polygon.difference(building_geometry)
    if visible_polygon.is_empty:
        visible_polygon = polygon.buffer(0).difference(building_geometry)

    return visible_polygon.area
