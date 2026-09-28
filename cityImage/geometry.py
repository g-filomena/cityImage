"""Small geometry helpers used by cityImage-owned workflows.

The module contains only geometry operations that preserve cityImage-specific
network, topology, visibility, or building-schema semantics. Generic envelope,
convex hull, distance-nearest, and GeoDataFrame-construction helpers were
removed during the hard API cleanup; use GeoPandas/Shapely directly for those.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, MultiPoint, MultiPolygon, Point
from shapely.ops import linemerge, split


def _union_all(geometries: Any) -> Any:
    """Return a unary union with GeoPandas/Shapely version compatibility."""
    if hasattr(geometries, "union_all"):
        return geometries.union_all()
    return geometries.unary_union


def center_line(line_geometries: list[LineString]) -> LineString:
    """Compute a centre line from consistently oriented LineStrings.

    clean_same_vertexes_edges uses it when a street mapped an even number of times between the
    same two nodes is collapsed to the centre line of its two middle copies.

    Parameters
    ----------
    line_geometries : list[shapely.geometry.LineString]
        LineStrings representing comparable edge geometries.

    Returns
    -------
    shapely.geometry.LineString
        Average line geometry: the lines are sampled at the same fractions of their length (every
        vertex of every line), so lines with different vertex counts are averaged whole. It runs in
        the first line's direction, and is 3D when every line is.
    """
    if len(line_geometries) < 2:
        raise ValueError("At least two LineStrings are required to compute a center line.")

    all_coords = [list(line.coords) for line in line_geometries]
    reference_start = all_coords[0][0]
    reference_end = all_coords[0][-1]

    def _squared_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    # Orient every line against the first line. Reverse only when the reversed
    # endpoints match the reference direction better. This preserves parallel
    # same-direction lines and fixes opposite-direction duplicates.
    for i in range(1, len(all_coords)):
        coords = all_coords[i]
        same_direction_distance = _squared_distance(coords[0], reference_start) + _squared_distance(
            coords[-1], reference_end
        )
        reversed_direction_distance = _squared_distance(
            coords[-1], reference_start
        ) + _squared_distance(coords[0], reference_end)
        if reversed_direction_distance < same_direction_distance:
            all_coords[i] = coords[::-1]

    # Average points taken at the same fractions of each line's length, whatever its vertex count:
    # every vertex of every line, as a fraction of its own line, so no line's corners are lost.
    lines = [LineString(coords) for coords in all_coords]
    fractions = {0.0, 1.0}
    for coords, line in zip(all_coords, lines, strict=True):
        if line.length == 0:
            continue
        travelled = 0.0
        for a, b in zip(coords[:-1], coords[1:], strict=True):
            travelled += _squared_distance(a, b) ** 0.5
            fractions.add(round(min(travelled / line.length, 1.0), 12))

    has_z = all(line.has_z for line in lines)
    center_line_coords = []
    for fraction in sorted(fractions):
        points = [line.interpolate(fraction, normalized=True) for line in lines]
        coords = (
            sum(point.x for point in points) / len(points),
            sum(point.y for point in points) / len(points),
        )
        if has_z:
            coords += (sum(point.z for point in points) / len(points),)
        center_line_coords.append(coords)

    return LineString(center_line_coords)


def split_line_at_MultiPoint(
    line_geometry: LineString,
    intersections: MultiPoint | list[Point],
    z: float | None = 0.0,
) -> list[LineString]:
    """Split a LineString at one or more point intersections.

    Parameters
    ----------
    line_geometry : shapely.geometry.LineString
        LineString to split.
    intersections : shapely.geometry.MultiPoint or list[Point]
        Points where the line should be split.
    z : float or None, default 0.0
        Optional z value to assign to split segments. Pass ``None`` to preserve
        2D geometries.

    Returns
    -------
    list[shapely.geometry.LineString]
        Split line segments.
    """
    line_geometry_tmp = line_geometry
    points = list(intersections.geoms) if hasattr(intersections, "geoms") else list(intersections)

    for point in points:
        new_line_coords = list(line_geometry_tmp.coords)
        for n, coord in enumerate(new_line_coords):
            if n == 0:
                continue

            line_section = LineString([Point(new_line_coords[n - 1]), Point(coord)])
            if point.intersects(line_section) or line_section.distance(point) < 1e-8:
                new_line_coords.insert(n, point.coords[0])
                break

        line_geometry_tmp = LineString(new_line_coords)

    split_points = MultiPoint(points)
    lines = split(line_geometry_tmp, split_points)
    lines_list = [line for line in lines.geoms]

    if z is not None:
        lines_list = [
            LineString([(coords[0], coords[1], z) for coords in line.coords]) for line in lines_list
        ]

    return lines_list


def fix_multiparts_LineString_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Merge or explode MultiLineString geometries into LineString rows."""
    gdf = gdf.copy()

    if "MultiLineString" not in gdf.geometry.type.unique():
        return gdf

    condition = gdf.geometry.type == "MultiLineString"
    gdf.loc[condition, "geometry"] = gpd.GeoSeries(
        [linemerge(geometry) for geometry in gdf.loc[condition, "geometry"]],
        index=gdf.loc[condition].index,
        crs=gdf.crs,
    )

    if "MultiLineString" not in gdf.geometry.type.unique():
        return gdf

    multiline_gdf = gdf[gdf.geometry.type == "MultiLineString"].copy()
    line_gdf = gdf[gdf.geometry.type == "LineString"].copy()
    multiline_gdf = multiline_gdf.explode(ignore_index=True)

    return gpd.GeoDataFrame(
        pd.concat([line_gdf, multiline_gdf], ignore_index=True),
        geometry="geometry",
        crs=gdf.crs,
    )


def gdf_multipolygon_to_polygon(
    gdf: gpd.GeoDataFrame,
    columnID: str = "buildingID",
) -> gpd.GeoDataFrame:
    """Convert one-part MultiPolygons to Polygons and explode remaining MultiPolygons.

    ``columnID`` keeps its values unless a MultiPolygon is split into several rows; only then is
    it renumbered from 0, since the parts would otherwise share one ID.
    """

    def convert_multipolygon_to_polygon(geometry: Any) -> Any:
        if isinstance(geometry, MultiPolygon) and len(geometry.geoms) == 1:
            return geometry.geoms[0]
        return geometry

    out = gdf.copy()
    out["geometry"] = out["geometry"].apply(convert_multipolygon_to_polygon)

    split = out["geometry"].apply(lambda geom: isinstance(geom, MultiPolygon)).any()
    if split:
        out = out.explode(index_parts=False, ignore_index=True)

    out = out.reset_index(drop=True)
    # Renumbered only when rows were split, as the parts would share one ID; otherwise the IDs a
    # file or caller supplied are kept.
    if split and columnID in out.columns:
        out[columnID] = out.index

    out["area"] = out.geometry.area
    return out
