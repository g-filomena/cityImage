"""Lynchian barrier extraction and assignment.

This module preserves cityImage's barrier semantics while delegating raw OSM
feature retrieval to external libraries such as OSMnx.

Preferred workflow:

```python
roads = ox.features_from_place(place, tags={"highway": True})
waterways = ox.features_from_place(place, tags={"waterway": True})
water = ox.features_from_place(place, tags={"natural": "water"})
coastline = ox.features_from_place(place, tags={"natural": "coastline"})
railways = ox.features_from_place(place, tags={"railway": True})
parks = ox.features_from_place(place, tags={"leisure": True})

barriers = ci.barriers_from_osm_features(
    roads_gdf=roads,
    waterways_gdf=waterways,
    water_gdf=water,
    coastline_gdf=coastline,
    railways_gdf=railways,
    parks_gdf=parks,
    crs=target_crs,
)
```

The old live OSM-loading functions were intentionally removed from the core API.
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Polygon,
)
from shapely.ops import nearest_points, polygonize_full, unary_union

pd.set_option("display.precision", 3)


ROAD_BARRIER_HIGHWAYS = {"trunk", "motorway"}
PRIMARY_ROAD_BARRIER_HIGHWAYS = {"primary"}
SECONDARY_ROAD_BARRIER_HIGHWAYS = {"secondary"}
WATERWAY_BARRIER_VALUES = {"river", "canal"}
EXCLUDED_WATER_VALUES = {
    "river",
    "stream",
    "canal",
    "riverbank",
    "reflecting_pool",
    "reservoir",
    "bay",
}
RAILWAY_BARRIER_VALUES = {"rail"}
LIGHT_RAILWAY_BARRIER_VALUES = {"light_rail", "tram"}
PARK_LEISURE_VALUES = {"park"}


def barrier_osm_feature_tags() -> dict[str, dict[str, Any]]:
    """Return OSM tag queries needed to build the full barrier layer externally."""
    return {
        "roads_gdf": {"highway": True},
        "waterways_gdf": {"waterway": True},
        "water_gdf": {"natural": "water"},
        "coastline_gdf": {"natural": "coastline"},
        "railways_gdf": {"railway": True},
        "parks_gdf": {"leisure": True},
    }


def _empty_barriers(crs: Any = None, barrier_type: str | None = None) -> gpd.GeoDataFrame:
    """Return an empty barrier GeoDataFrame."""
    data = {"barrier_type": []}
    if barrier_type is not None:
        data["barrier_type"] = pd.Series(dtype="object")
    return gpd.GeoDataFrame(data, geometry=gpd.GeoSeries([], crs=crs), crs=crs)


def _as_projected(gdf: gpd.GeoDataFrame | None, crs: Any = None) -> gpd.GeoDataFrame:
    """Return a defensive projected copy, or an empty GeoDataFrame."""
    if gdf is None:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError("barrier feature inputs must be GeoDataFrames or None")

    out = gdf.copy()
    if "geometry" not in out:
        out = gpd.GeoDataFrame(out, geometry=[], crs=crs)

    out = out[out.geometry.notna() & ~out.geometry.is_empty].copy()

    if crs is not None:
        if out.crs is None:
            out = out.set_crs("EPSG:4326")
        out = out.to_crs(crs)

    return out


def _ensure_column(gdf: gpd.GeoDataFrame, column: str, default: Any = pd.NA) -> gpd.GeoDataFrame:
    """Ensure a GeoDataFrame has a column."""
    if column not in gdf.columns:
        gdf = gdf.copy()
        gdf[column] = default
    return gdf


def _union_all(geometries: Any) -> Any:
    """Union geometries with GeoPandas/Shapely compatibility."""
    if hasattr(geometries, "union_all"):
        return geometries.union_all()
    return unary_union(list(geometries))


def _features_gdf(geometries: list[Any], crs: Any, barrier_type: str) -> gpd.GeoDataFrame:
    """Create a normalised barrier GeoDataFrame from geometries."""
    geometries = [geometry for geometry in geometries if geometry is not None and not geometry.is_empty]
    if not geometries:
        return _empty_barriers(crs=crs, barrier_type=barrier_type)

    return gpd.GeoDataFrame(
        {"barrier_type": [barrier_type] * len(geometries)},
        geometry=geometries,
        crs=crs,
    )


def _geometries_to_lines(gdf: gpd.GeoDataFrame, barrier_type: str, crs: Any) -> gpd.GeoDataFrame:
    """Union and simplify a set of geometries into barrier line features."""
    if gdf.empty:
        return _empty_barriers(crs=crs, barrier_type=barrier_type)

    geometries = _simplify_barrier(_union_all(gdf.geometry))
    return _features_gdf(geometries, crs=crs, barrier_type=barrier_type)


def road_barriers_from_osm_features(
    roads_gdf: gpd.GeoDataFrame | None,
    crs: Any = None,
    *,
    include_primary: bool = False,
    include_secondary: bool = False,
) -> gpd.GeoDataFrame:
    """Build road barrier features from already-downloaded OSM highway features."""
    roads = _as_projected(roads_gdf, crs)
    if roads.empty:
        return _empty_barriers(crs=crs, barrier_type="road")

    roads = _ensure_column(roads, "highway")
    to_keep = set(ROAD_BARRIER_HIGHWAYS)
    if include_primary:
        to_keep |= PRIMARY_ROAD_BARRIER_HIGHWAYS
    if include_secondary:
        to_keep |= SECONDARY_ROAD_BARRIER_HIGHWAYS

    roads = roads[roads["highway"].isin(to_keep)].copy()

    if "tunnel" in roads.columns:
        roads["tunnel"] = roads["tunnel"].fillna(0)
        roads = roads[roads["tunnel"] == 0].copy()

    return _geometries_to_lines(roads, barrier_type="road", crs=crs)


def water_barriers_from_osm_features(
    *,
    waterways_gdf: gpd.GeoDataFrame | None = None,
    water_gdf: gpd.GeoDataFrame | None = None,
    coastline_gdf: gpd.GeoDataFrame | None = None,
    crs: Any = None,
    lakes_area: float = 1000,
    min_lake_boundary_length: float = 500,
) -> gpd.GeoDataFrame:
    """Build water barrier features from already-downloaded OSM water features."""
    parts: list[gpd.GeoDataFrame] = []

    waterways = _as_projected(waterways_gdf, crs)
    if not waterways.empty:
        waterways = _ensure_column(waterways, "waterway")
        waterways = waterways[waterways["waterway"].isin(WATERWAY_BARRIER_VALUES)].copy()
        parts.append(_geometries_to_lines(waterways, barrier_type="water", crs=crs))

    water = _as_projected(water_gdf, crs)
    if not water.empty:
        if "water" in water.columns:
            water = water[~water["water"].isin(EXCLUDED_WATER_VALUES)].copy()
        water["area"] = water.geometry.area
        water = water[water["area"] > lakes_area].copy()
        lakes = _geometries_to_lines(water, barrier_type="water", crs=crs)
        if not lakes.empty:
            lakes["length"] = lakes.geometry.length
            lakes = lakes[lakes["length"] >= min_lake_boundary_length].drop(columns=["length"])
        parts.append(lakes)

    coastline = _as_projected(coastline_gdf, crs)
    if not coastline.empty:
        parts.append(_geometries_to_lines(coastline, barrier_type="water", crs=crs))

    parts = [part for part in parts if not part.empty]
    if not parts:
        return _empty_barriers(crs=crs, barrier_type="water")

    water_all = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=crs)
    return _geometries_to_lines(water_all, barrier_type="water", crs=crs)


def railway_barriers_from_osm_features(
    railways_gdf: gpd.GeoDataFrame | None,
    crs: Any = None,
    *,
    keep_light_rail: bool = False,
) -> gpd.GeoDataFrame:
    """Build railway barrier features from already-downloaded OSM railway features."""
    railways = _as_projected(railways_gdf, crs)
    if railways.empty:
        return _empty_barriers(crs=crs, barrier_type="railway")

    railways = _ensure_column(railways, "railway")
    to_keep = set(RAILWAY_BARRIER_VALUES)
    if keep_light_rail:
        to_keep |= LIGHT_RAILWAY_BARRIER_VALUES

    railways = railways[railways["railway"].isin(to_keep)].copy()

    if "tunnel" in railways.columns:
        railways["tunnel"] = railways["tunnel"].fillna(0)
        railways = railways[railways["tunnel"] == 0].copy()

    if railways.empty:
        return _empty_barriers(crs=crs, barrier_type="railway")

    railway_union = _union_all(railways.geometry)
    polygons = polygonize_full(railway_union)
    railway_boundary = unary_union(polygons).buffer(10).boundary
    return _features_gdf(_simplify_barrier(railway_boundary), crs=crs, barrier_type="railway")


def park_barriers_from_osm_features(
    parks_gdf: gpd.GeoDataFrame | None,
    crs: Any = None,
    *,
    min_area: float = 100000,
) -> gpd.GeoDataFrame:
    """Build park barrier features from already-downloaded OSM leisure features."""
    parks = _as_projected(parks_gdf, crs)
    if parks.empty:
        return _empty_barriers(crs=crs, barrier_type="park")

    parks = _ensure_column(parks, "leisure")
    parks = parks[parks["leisure"].isin(PARK_LEISURE_VALUES)].copy()
    parks["area"] = parks.geometry.area
    parks = parks[parks["area"] >= min_area].copy()

    if parks.empty:
        return _empty_barriers(crs=crs, barrier_type="park")

    park_union = _union_all(parks.geometry)
    polygons = polygonize_full(park_union)
    park_boundary = unary_union(polygons).buffer(10).boundary
    return _features_gdf(_simplify_barrier(park_boundary), crs=crs, barrier_type="park")


def barriers_from_osm_features(
    *,
    roads_gdf: gpd.GeoDataFrame | None = None,
    waterways_gdf: gpd.GeoDataFrame | None = None,
    water_gdf: gpd.GeoDataFrame | None = None,
    coastline_gdf: gpd.GeoDataFrame | None = None,
    railways_gdf: gpd.GeoDataFrame | None = None,
    parks_gdf: gpd.GeoDataFrame | None = None,
    crs: Any = None,
    include_primary: bool = True,
    include_secondary: bool = False,
    lakes_area: float = 1000,
    parks_min_area: float = 100000,
    keep_light_rail: bool = False,
) -> gpd.GeoDataFrame:
    """Build a combined cityImage barrier layer from already-downloaded OSM features."""
    parts = [
        road_barriers_from_osm_features(
            roads_gdf,
            crs=crs,
            include_primary=include_primary,
            include_secondary=include_secondary,
        ),
        water_barriers_from_osm_features(
            waterways_gdf=waterways_gdf,
            water_gdf=water_gdf,
            coastline_gdf=coastline_gdf,
            crs=crs,
            lakes_area=lakes_area,
        ),
        railway_barriers_from_osm_features(
            railways_gdf,
            crs=crs,
            keep_light_rail=keep_light_rail,
        ),
        park_barriers_from_osm_features(
            parks_gdf,
            crs=crs,
            min_area=parks_min_area,
        ),
    ]
    parts = [part for part in parts if not part.empty]

    if not parts:
        return gpd.GeoDataFrame(
            {"barrierID": pd.Series(dtype="int64"), "barrier_type": pd.Series(dtype="object")},
            geometry=gpd.GeoSeries([], crs=crs),
            crs=crs,
        )

    barriers = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), geometry="geometry", crs=crs)
    barriers = barriers.reset_index(drop=True)
    barriers["barrierID"] = barriers.index.astype(int)
    return barriers


def along_water(edges_gdf: gpd.GeoDataFrame, barriers_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assign water barriers lying along/crossing each street segment."""
    sindex = edges_gdf.sindex
    tmp = barriers_gdf[barriers_gdf["barrier_type"].isin(["water"])]
    edges_gdf["ac_rivers"] = edges_gdf.apply(
        lambda row: barriers_along(row["edgeID"], edges_gdf, tmp, sindex, offset=200),
        axis=1,
    )
    edges_gdf["c_rivers"] = edges_gdf.apply(
        lambda row: _crossing_barriers(row["geometry"], tmp),
        axis=1,
    )
    edges_gdf["bridge"] = edges_gdf.apply(lambda row: len(row["c_rivers"]) > 0, axis=1)
    edges_gdf["a_rivers"] = edges_gdf.apply(
        lambda row: list(set(row["ac_rivers"]) - set(row["c_rivers"])),
        axis=1,
    )
    edges_gdf["a_rivers"] = edges_gdf.apply(
        lambda row: row["ac_rivers"] if not row["bridge"] else [],
        axis=1,
    )
    edges_gdf.drop(["ac_rivers", "c_rivers"], axis=1, inplace=True)

    return edges_gdf


def along_within_parks(edges_gdf: gpd.GeoDataFrame, barriers_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assign park barrier IDs lying along or containing each street segment."""
    park_polygons = barriers_gdf[barriers_gdf["barrier_type"] == "park"].copy()
    if park_polygons.empty:
        edges_gdf["w_parks"] = [[] for _ in range(len(edges_gdf))]
        return edges_gdf

    park_polygons["geometry"] = park_polygons.apply(
        lambda row: polygonize_full(row["geometry"])[0],
        axis=1,
    )
    park_polygons = gpd.GeoDataFrame(
        park_polygons["barrierID"],
        geometry=park_polygons["geometry"],
        crs=edges_gdf.crs,
    )

    edges_gdf["w_parks"] = edges_gdf.apply(
        lambda row: _within_parks(row["geometry"], park_polygons),
        axis=1,
    )

    return edges_gdf


def barriers_along(
    ix_line: int,
    edges_gdf: gpd.GeoDataFrame,
    barriers_gdf: gpd.GeoDataFrame,
    edges_gdf_sindex: Any,
    offset: float = 100,
) -> list[int]:
    """Return barrier IDs along a given edge, excluding touching/crossing barriers."""
    if barriers_gdf.empty:
        return []

    buffer = edges_gdf.loc[ix_line].geometry.buffer(offset)
    intersecting_barriers = barriers_gdf[
        barriers_gdf.geometry.intersects(buffer)
        & ~barriers_gdf.geometry.touches(edges_gdf.loc[ix_line].geometry)
    ]
    if intersecting_barriers.empty:
        return []

    possible_matches = edges_gdf.iloc[list(edges_gdf_sindex.intersection(buffer.bounds))].drop(
        ix_line
    )
    along = []
    for _, barrier in intersecting_barriers.iterrows():
        midpoint = edges_gdf.loc[ix_line].geometry.interpolate(0.5, normalized=True)
        line = LineString([midpoint, nearest_points(midpoint, barrier["geometry"])[1]])
        if not possible_matches[possible_matches.geometry.intersects(line)].empty:
            continue
        along.append(barrier["barrierID"])

    return along


def _within_parks(line_geometry: LineString, park_polygons: gpd.GeoDataFrame) -> list[int]:
    """Return IDs of parks intersecting a line, excluding boundary-touching parks."""
    if park_polygons.empty:
        return []

    park_sindex = park_polygons.sindex
    possible_matches_index = list(park_sindex.intersection(line_geometry.bounds))
    possible_matches = park_polygons.iloc[possible_matches_index]
    intersecting_parks = possible_matches[possible_matches.geometry.intersects(line_geometry)]
    touching_parks = possible_matches[possible_matches.geometry.touches(line_geometry)]

    if len(intersecting_parks) == 0:
        return []

    intersecting_parks = intersecting_parks[
        ~intersecting_parks.barrierID.isin(list(touching_parks.barrierID))
    ]
    return list(intersecting_parks.barrierID)


def assign_structuring_barriers(
    edges_gdf: gpd.GeoDataFrame,
    barriers_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Add boolean ``sep_barr`` indicating crossing with structuring barriers."""
    barriers_gdf = barriers_gdf.copy()
    edges_gdf = edges_gdf.copy()
    exclude = ["secondary_road", "park"]
    tmp = barriers_gdf[~barriers_gdf["barrier_type"].isin(exclude)].copy()

    edges_gdf["c_barr"] = edges_gdf.apply(
        lambda row: _crossing_barriers(row["geometry"], tmp),
        axis=1,
    )
    edges_gdf["sep_barr"] = edges_gdf.apply(lambda row: len(row["c_barr"]) > 0, axis=1)
    edges_gdf.drop("c_barr", axis=1, inplace=True)

    return edges_gdf


def _crossing_barriers(line_geometry: LineString, barriers_gdf: gpd.GeoDataFrame) -> list[int]:
    """Return IDs of barriers crossing a line, excluding boundary touches."""
    adjacent_barriers = []
    if barriers_gdf.empty:
        return adjacent_barriers

    intersecting_barriers = barriers_gdf[barriers_gdf.geometry.intersects(line_geometry)]
    touching_barriers = barriers_gdf[barriers_gdf.geometry.touches(line_geometry)]

    if len(intersecting_barriers) == 0:
        return adjacent_barriers

    intersecting_barriers = intersecting_barriers[
        ~intersecting_barriers.barrierID.isin(list(touching_barriers.barrierID))
    ]
    return list(intersecting_barriers.barrierID)


def _simplify_barrier(geometries: Any) -> list[Any]:
    """Return line-like barrier geometries from a Shapely geometry collection."""
    if geometries is None or geometries.is_empty:
        return []

    if isinstance(geometries, Polygon):
        return [geometries.boundary]

    if isinstance(geometries, LineString):
        return [geometries]

    if isinstance(geometries, MultiLineString):
        return list(geometries.geoms)

    if isinstance(geometries, MultiPolygon):
        return list(geometries.boundary.geoms)

    if isinstance(geometries, GeometryCollection):
        features = list(geometries.geoms)
        for index, feature in enumerate(features):
            if isinstance(feature, Polygon):
                features[index] = feature.boundary
        return features

    return []
