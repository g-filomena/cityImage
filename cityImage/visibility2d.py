"""2D visibility: the visibility polygon (isovist) used by the structural landmark score, and
its measures.

It needs only NumPy and Shapely; the 3D sight lines, which need Dask and psutil, are in
``cityImage.visibility3d``.
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


## Isovists ###########################
#
# The isovist from a point is the area visible from it (Benedikt 1979). Its size is how open the
# place is, and the corners along its edge, where the view stops at one building and continues
# past it, are where someone could stand unseen: the prospect and refuge of Fisher and Nasar
# (1992). ``isovist_measures`` measures one isovist from a fan of rays.


def isovist_measures(radii, hits, min_jump=2.0, jump_ratio=0.5, walls=None):
    """Area, compactness and occlusion of one isovist, from its ray lengths.

    Parameters
    ----------
    radii
        Length of each ray, in order around the circle, equally spaced.
    hits
        For each ray, an identifier of the obstruction it stopped at, or -1 if it met none.
    min_jump, jump_ratio
        Two neighbouring rays mark an occluding edge when they stop at different obstructions (or
        one at none) and their lengths differ by more than ``min_jump`` and by more than
        ``jump_ratio`` times the shorter one: the view passes a corner rather than running along a
        facade split into several footprints.
    walls
        Optionally, for each ray, the wall segment it stopped at, as ``(x1, y1, x2, y2)`` relative
        to the observer (NaN for a ray that met none). With it, a jump is an occluding edge only
        if the longer ray crosses the line of the shorter ray's wall within its own length - past
        the end of the wall. A longer ray that never crosses that line, or crosses it beyond its
        reach, is looking along the wall at a grazing angle, not past a corner.

    Returns
    -------
    tuple
        ``(area, compactness, occlusions, occlusivity)``: the area of the polygon through the ray
        tips, its isoperimetric quotient ``4 pi area / perimeter^2`` (1 for a circle), the number
        of occluding edges, and their summed length - a discrete form of Benedikt's occlusivity.
    """
    radii = np.asarray(radii, dtype=float)
    hits = np.asarray(hits)
    k = len(radii)
    step = 2.0 * np.pi / k
    following = np.roll(radii, -1)
    area = 0.5 * np.sin(step) * float(np.sum(radii * following))
    angles = np.arange(k) * step
    x, y = radii * np.cos(angles), radii * np.sin(angles)
    perimeter = float(np.sum(np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)))
    compactness = 4.0 * np.pi * area / perimeter**2 if perimeter > 0 else 0.0
    jump = np.abs(following - radii)
    shorter = np.minimum(radii, following)
    occluding = (hits != np.roll(hits, -1)) & (jump > min_jump) & (jump > jump_ratio * shorter)
    if walls is not None:
        for i in np.flatnonzero(occluding):
            j = (i + 1) % k
            near, far = (i, j) if radii[i] < radii[j] else (j, i)
            occluding[i] = _passes_wall(angles[far], radii[far], walls[near])
    return area, compactness, int(occluding.sum()), float(jump[occluding].sum())


def _passes_wall(angle, reach, wall):
    """Whether a ray from the origin at ``angle`` crosses the line through ``wall`` within
    ``reach``."""
    x1, y1, x2, y2 = wall
    if np.isnan(x1):
        return True
    dx, dy = np.cos(angle), np.sin(angle)
    ex, ey = x2 - x1, y2 - y1
    den = dx * ey - dy * ex
    if abs(den) < 1e-12:
        return False
    t = (x1 * ey - y1 * ex) / den
    return 0.0 < t <= reach
