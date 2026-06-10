"""Pedestrian network filtering and construction.

This module preserves the cityImage-specific pedestrian filtering semantics
previously embedded in the old OSM loading route, while delegating raw OSM
feature retrieval to OSMnx and generic line-to-network construction to
``cityImage.network``.

The core entry point is ``pedestrian_network_from_osm_features``: pass a
GeoDataFrame of OSM highway features that you downloaded elsewhere, and this
module filters it into a cityImage-style pedestrian network.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import geopandas as gpd
import pandas as pd

from .network import network_from_lines

EXCLUDED_HIGHWAY_VALUES = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "busway",
    "construction",
    "proposed",
    "raceway",
}

EXCLUDED_ACCESS_VALUES = {
    "private",
    "no",
}

PEDESTRIAN_OK_FOOT_VALUES = {
    "yes",
    "designated",
    "permissive",
    "destination",
}

EXCLUDED_FOOTWAY_VALUES = {
    "sidewalk",
    "traffic_island",
}

SIDEWALK_NO_VALUES = {
    "no",
    "none",
    "separate",
}

RESIDENTIAL_SIDEWALK_EXCEPTION = {
    "residential",
    "living_street",
}


def _as_tokens(value: Any) -> list[str]:
    """Return a normalised list of lowercase OSM tag tokens."""
    if isinstance(value, str):
        return [value.strip().lower()] if value.strip() else []

    if isinstance(value, Iterable) and not isinstance(value, (bytes, str)):
        tokens: list[str] = []
        for item in value:
            tokens.extend(_as_tokens(item))
        return tokens

    if pd.isna(value):
        return []

    token = str(value).strip().lower()
    return [token] if token else []


def _first_token(value: Any) -> str | None:
    """Return the first normalised OSM tag token, if available."""
    tokens = _as_tokens(value)
    return tokens[0] if tokens else None


def _truthy_osm_yes(value: Any) -> bool:
    """Return True for OSM-style yes/true/1 values."""
    return _first_token(value) in {"yes", "true", "1"}


def _ensure_columns(gdf: gpd.GeoDataFrame, columns: Iterable[str]) -> gpd.GeoDataFrame:
    """Ensure all requested columns exist, filled with NA values when absent."""
    gdf = gdf.copy()
    for column in columns:
        if column not in gdf.columns:
            gdf[column] = pd.NA
    return gdf


def _line_geometries_only(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep only LineString and MultiLineString geometries."""
    return gdf[gdf.geometry.geom_type.isin({"LineString", "MultiLineString"})].copy()


def _is_pedestrian_row(row: pd.Series) -> bool:
    """Return True when an OSM highway row should remain in the pedestrian network."""
    highway_tokens = set(_as_tokens(row.get("highway")))
    highway = _first_token(row.get("highway"))

    if not highway_tokens:
        return False

    if highway_tokens & EXCLUDED_HIGHWAY_VALUES:
        return False

    if _truthy_osm_yes(row.get("area")):
        return False

    if set(_as_tokens(row.get("access"))) & EXCLUDED_ACCESS_VALUES:
        return False

    if set(_as_tokens(row.get("foot"))) & {"no"}:
        return False

    if highway == "cycleway" and not (set(_as_tokens(row.get("foot"))) & PEDESTRIAN_OK_FOOT_VALUES):
        return False

    if set(_as_tokens(row.get("footway"))) & EXCLUDED_FOOTWAY_VALUES:
        return False

    sidewalk_tokens = set()
    for column, value in row.items():
        if str(column).startswith("sidewalk"):
            sidewalk_tokens.update(_as_tokens(value))

    return not (
        sidewalk_tokens & SIDEWALK_NO_VALUES and highway not in RESIDENTIAL_SIDEWALK_EXCEPTION
    )


def filter_pedestrian_osm_features(highways_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter OSM highway features using cityImage pedestrian-network semantics.

    Parameters
    ----------
    highways_gdf
        GeoDataFrame of OSM features, usually downloaded with an OSMnx
        ``features_from_*``/``geometries_from_*`` call using
        ``tags={"highway": True}``.

    Returns
    -------
    geopandas.GeoDataFrame
        Line/MultiLine features that remain after pedestrian filtering.
    """
    if not isinstance(highways_gdf, gpd.GeoDataFrame):
        raise TypeError("highways_gdf must be a GeoDataFrame")

    if highways_gdf.empty:
        return highways_gdf.copy()

    required_filter_columns = [
        "highway",
        "area",
        "foot",
        "footway",
        "cycleway",
        "access",
    ]

    gdf = _ensure_columns(highways_gdf, required_filter_columns)
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = _line_geometries_only(gdf)

    if gdf.empty:
        return gdf

    mask = gdf.apply(_is_pedestrian_row, axis=1)
    return gdf[mask].copy()


def pedestrian_network_from_osm_features(
    highways_gdf: gpd.GeoDataFrame,
    crs: Any,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Build a cityImage pedestrian network from already-downloaded OSM features.

    Raw OSM download is deliberately not owned by this function. Use OSMnx or
    another data source to retrieve highway features, then pass them here.
    """
    filtered = filter_pedestrian_osm_features(highways_gdf)

    if filtered.empty:
        empty_edges = gpd.GeoDataFrame(columns=["edgeID", "u", "v", "length"], geometry=[], crs=crs)
        empty_nodes = gpd.GeoDataFrame(columns=["nodeID", "x", "y"], geometry=[], crs=crs)
        return empty_nodes, empty_edges

    metadata_columns = [
        "name",
        "highway",
        "lit",
        "foot",
        "footway",
        "cycleway",
        "access",
        "surface",
        "width",
    ]
    metadata_columns.extend(
        column for column in filtered.columns if str(column).startswith("sidewalk")
    )

    other_columns = []
    seen = set()
    for column in metadata_columns:
        if column in filtered.columns and column not in seen:
            other_columns.append(column)
            seen.add(column)

    return network_from_lines(filtered, crs, other_columns=other_columns)


def _call_osmnx_features(ox: Any, method_name: str, *args: Any, **kwargs: Any) -> gpd.GeoDataFrame:
    """Call modern OSMnx feature functions with fallback for older versions."""
    modern = getattr(ox, f"features_from_{method_name}", None)
    if modern is not None:
        return modern(*args, **kwargs)

    legacy = getattr(ox, f"geometries_from_{method_name}", None)
    if legacy is not None:
        return legacy(*args, **kwargs)

    raise ImportError(
        "Installed OSMnx version does not expose features_from_* or geometries_from_* APIs"
    )


def pedestrian_network_from_osm(
    query: Any = None,
    *,
    crs: Any,
    download_method: str = "OSMplace",
    distance: float = 500,
    address: str | None = None,
    point: tuple[float, float] | None = None,
    polygon: Any = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Download OSM highway features with OSMnx and build a pedestrian network.

    This is a thin optional convenience wrapper. The cityImage-specific part is
    the pedestrian filtering; OSM acquisition remains delegated to OSMnx.

    Parameters
    ----------
    query
        Place name for ``download_method="OSMplace"``. For address/point/polygon
        methods, prefer the explicit keyword arguments.
    crs
        Target projected CRS for the output network.
    download_method
        One of ``"OSMplace"``, ``"distance_from_address"``,
        ``"distance_from_point"``, or ``"polygon"``.
    distance
        Distance in metres for address/point downloads.
    address, point, polygon
        Explicit spatial inputs for the corresponding download methods.
    """
    try:
        import osmnx as ox
    except ImportError as exc:
        raise ImportError(
            "pedestrian_network_from_osm requires the optional 'osm' dependency. "
            'Install with: python -m pip install -e ".[osm]"'
        ) from exc

    tags = {"highway": True}

    if download_method == "OSMplace":
        if query is None:
            raise ValueError("query must be provided when download_method='OSMplace'")
        features = _call_osmnx_features(ox, "place", query, tags=tags)
    elif download_method == "distance_from_address":
        address_query = address or query
        if address_query is None:
            raise ValueError(
                "address or query must be provided when download_method='distance_from_address'"
            )
        features = _call_osmnx_features(ox, "address", address_query, tags=tags, dist=distance)
    elif download_method == "distance_from_point":
        point_query = point or query
        if point_query is None:
            raise ValueError(
                "point or query must be provided when download_method='distance_from_point'"
            )
        features = _call_osmnx_features(ox, "point", point_query, tags=tags, dist=distance)
    elif download_method == "polygon":
        polygon_query = polygon or query
        if polygon_query is None:
            raise ValueError("polygon or query must be provided when download_method='polygon'")
        features = _call_osmnx_features(ox, "polygon", polygon_query, tags=tags)
    else:
        raise ValueError(
            "download_method must be one of: OSMplace, distance_from_address, "
            "distance_from_point, polygon"
        )

    return pedestrian_network_from_osm_features(features, crs)
