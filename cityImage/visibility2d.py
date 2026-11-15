"""Lightweight 2D visibility helpers for cityImage.

This module contains the core 2D visibility primitive used by landmark
structural scoring. It deliberately avoids heavy 3D/runtime dependencies such
as Dask and psutil.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry import LineString, Point, Polygon

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
    # The rays meet only at the origin, so their union is the plain multi-line.
    rays = shapely.multilinestrings(lines)

    # Spatial pre-filter: only obstructions whose bounding box meets the ray fan can be
    # crossed by it, so query the index and run the exact (comparatively costly) `crosses`
    # test on just those few candidates instead of the whole obstruction set. This is
    # identical to crossing every obstruction (the crossers are a subset of the bbox
    # candidates) but avoids the per-building full-table scan that dominated runtime. The
    # frame's own cached index is used, so `obstructions_sindex` stays a no-op accepted for
    # backward compatibility.
    # Only the geometries are read: taking rows of the whole frame would copy every attribute
    # column for each building, which cost more than the geometry work itself.
    del obstructions_sindex
    candidate_positions = obstructions_gdf.sindex.query(rays)
    candidates = obstructions_gdf.geometry.array[candidate_positions].to_numpy()

    obstacles = candidates[shapely.crosses(candidates, rays)]
    obstacles = obstacles[np.array([geometry != building_geometry for geometry in obstacles], bool)]
    obstacles = obstacles[~shapely.within(obstacles, building_geometry.convex_hull)]

    # Each ray ends at the obstruction point nearest the origin: the nearest coordinate of its
    # intersections with every obstacle, which is that of its intersection with their union,
    # without computing the union. Point, line and collection hits all count.
    tips = coords.copy()
    if len(obstacles) > 0:
        ray_array = np.asarray(lines, dtype=object)
        hits = shapely.intersection(ray_array[:, None], obstacles[None, :])
        hit_coords, hit_index = shapely.get_coordinates(hits.ravel(), return_index=True)
        if len(hit_coords):
            ray_index = hit_index // len(obstacles)
            squared = (hit_coords[:, 0] - origin.x) ** 2 + (hit_coords[:, 1] - origin.y) ** 2
            order = np.lexsort((squared, ray_index))
            first = order[np.r_[True, ray_index[order][1:] != ray_index[order][:-1]]]
            tips[ray_index[first]] = hit_coords[first]

    # The ring runs through the ray tips only: closing it through the origin would cut it along
    # the first and last rays and leave out the slice between them.
    polygon = Polygon(tips)
    visible_polygon = polygon.difference(building_geometry)
    if visible_polygon.is_empty:
        visible_polygon = polygon.buffer(0).difference(building_geometry)

    return visible_polygon.area
