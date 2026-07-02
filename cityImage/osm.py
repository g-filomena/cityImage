"""OSM bridge helpers for cityImage.

OSMnx owns raw OSM acquisition. cityImage owns conversion from OSMnx outputs
into cityImage schemas and downstream semantics.

This module is lazily exposed from ``cityImage.__init__``. Importing
``cityImage`` does not import OSMnx.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import Point

from .adapters import standardize_buildings_gdf
from .barriers import barrier_osm_feature_tags, barriers_from_osm_features
from .geometry import gdf_multipolygon_to_polygon
from .landuse import classify_land_uses_raws_into_OSMgroups, derive_land_uses_raw_fromOSM
from .network import _resolve_list_edges_gdf, reset_index_graph_gdfs
from .pedestrian import pedestrian_network_from_osm

OSM_DOWNLOAD_METHODS = {"OSMplace", "distance_from_address", "distance_from_point", "polygon"}
DEFAULT_NETWORK_METADATA_COLUMNS = (
    "name",
    "highway",
    "oneway",
    "lanes",
    "maxspeed",
    "bridge",
    "tunnel",
    "lit",
    "surface",
)


def _empty_osm_features(tags, crs=None):
    """Return an empty OSM-like GeoDataFrame for missing Overpass results."""
    columns = list(tags.keys()) if isinstance(tags, dict) else []
    return gpd.GeoDataFrame({column: [] for column in columns}, geometry=[], crs=crs or "EPSG:4326")


def _empty_network(crs=None) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return an empty cityImage-style network."""
    output_crs = crs or "EPSG:4326"
    empty_edges = gpd.GeoDataFrame(
        columns=["edgeID", "u", "v", "key", "length"],
        geometry=[],
        crs=output_crs,
    )
    empty_nodes = gpd.GeoDataFrame(
        columns=["nodeID", "x", "y", "z"],
        geometry=[],
        crs=output_crs,
    )
    return empty_nodes, empty_edges


def _distance_arg(distance: float | None, download_method: str) -> float | None:
    """Return distance unchanged, but do not silently invent one."""
    if download_method in {"distance_from_address", "distance_from_point"} and distance is None:
        raise ValueError(
            "distance is required when download_method is "
            f"{download_method!r}; pass an explicit distance in metres"
        )
    return distance


def _unit_overlaps_for_land_uses(value):
    """Return equal overlap weights matching a land-use list."""
    if isinstance(value, tuple | set | list):
        values = [item for item in value if item is not None]
    elif value is None:
        values = []
    else:
        values = [value]

    if not values:
        return [1.0]

    weight = 1.0 / len(values)
    return [weight] * len(values)


def _normalise_crs(crs: Any) -> Any:
    """Accept integer EPSG codes as a convenience."""
    return f"EPSG:{crs}" if isinstance(crs, int) else crs


def _validate_download_method(download_method: str) -> None:
    if download_method not in OSM_DOWNLOAD_METHODS:
        raise ValueError(
            f"download_method must be one of: {', '.join(sorted(OSM_DOWNLOAD_METHODS))}"
        )


def features_from_osm(
    query: Any,
    tags: Mapping[str, Any],
    *,
    download_method: str = "OSMplace",
    distance: float | None = None,
    crs: Any = None,
) -> gpd.GeoDataFrame:
    """Download raw OSM features through OSMnx and optionally project them.

    OSMnx owns acquisition. cityImage keeps this wrapper to centralise the
    download-method dispatch used by the OSM bridge helpers.
    """
    _validate_download_method(download_method)

    try:
        if download_method == "OSMplace":
            features = ox.features_from_place(query, tags=tags)
        elif download_method == "distance_from_address":
            features = ox.features_from_address(
                query,
                tags=tags,
                dist=_distance_arg(distance, download_method),
            )
        elif download_method == "distance_from_point":
            features = ox.features_from_point(
                query,
                tags=tags,
                dist=_distance_arg(distance, download_method),
            )
        elif download_method == "polygon":
            features = ox.features_from_polygon(query, tags=tags)
        else:
            raise ValueError(f"Unknown download_method: {download_method}")
    except Exception as exc:
        if exc.__class__.__name__ == "InsufficientResponseError":
            features = _empty_osm_features(tags, crs="EPSG:4326")
        else:
            raise

    crs = _normalise_crs(crs)
    if crs is not None and not features.empty:
        features = features.to_crs(crs)
    elif crs is not None:
        features = features.set_crs("EPSG:4326", allow_override=True).to_crs(crs)

    return features


def _network_from_osmnx_graph(
    graph: Any,
    ox: Any,
    crs: Any,
    metadata_columns: Sequence[str],
    dict_columns: Mapping[str, str | None] | None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Convert an OSMnx graph while preserving legacy cityImage semantics."""
    graph = ox.project_graph(graph, to_crs=crs) if crs is not None else ox.project_graph(graph)
    graph_crs = crs or graph.graph.get("crs")

    nodes_gdf = ox.graph_to_gdfs(
        graph,
        nodes=True,
        edges=False,
        node_geometry=True,
        fill_edge_geometry=False,
    )
    edges_gdf = ox.graph_to_gdfs(
        graph,
        nodes=False,
        edges=True,
        node_geometry=False,
        fill_edge_geometry=True,
    )

    if nodes_gdf.empty or edges_gdf.empty:
        return _empty_network(graph_crs)

    nodes_gdf = nodes_gdf.drop(["highway", "ref"], axis=1, errors="ignore")
    edges_gdf = edges_gdf.reset_index()
    nodes_gdf["nodeID"] = nodes_gdf.index
    nodes_gdf, edges_gdf = reset_index_graph_gdfs(nodes_gdf, edges_gdf, nodeID="nodeID")

    if "geometry" not in nodes_gdf.columns:
        nodes_gdf = gpd.GeoDataFrame(
            nodes_gdf,
            geometry=[Point(xy) for xy in zip(nodes_gdf["x"], nodes_gdf["y"], strict=False)],
            crs=graph_crs,
        )

    new_columns: list[str] = []
    for key, value in (dict_columns or {}).items():
        if value is not None:
            if value not in edges_gdf.columns:
                raise ValueError(f"dict_columns maps {key!r} to missing column {value!r}")
            edges_gdf[key] = edges_gdf[value]
            new_columns.append(key)

    other_columns = [column for column in metadata_columns if column in edges_gdf.columns]
    to_keep = ["edgeID", "u", "v", "key", "geometry", "length", *new_columns, *other_columns]
    # Preserve order while avoiding duplicate columns when dict mappings target metadata names.
    to_keep = list(dict.fromkeys(to_keep))
    edges_gdf = edges_gdf[to_keep].copy()
    edges_gdf = _resolve_list_edges_gdf(edges_gdf)

    nodes_gdf = nodes_gdf[["nodeID", "x", "y", "geometry"]].copy()
    nodes_gdf["x"] = nodes_gdf.geometry.apply(lambda geom: geom.coords[0][0])
    nodes_gdf["y"] = nodes_gdf.geometry.apply(lambda geom: geom.coords[0][1])
    if len(nodes_gdf.geometry.iloc[0].coords[0]) > 2:
        nodes_gdf["z"] = nodes_gdf.geometry.apply(lambda geom: geom.coords[0][2])
    else:
        nodes_gdf["z"] = 2.0

    nodes_gdf = nodes_gdf[nodes_gdf.nodeID.isin(pd.unique(edges_gdf[["u", "v"]].values.ravel()))]
    return nodes_gdf, edges_gdf


def network_from_osm(
    query: Any,
    *,
    download_method: str = "OSMplace",
    network_type: str = "all",
    crs: Any = None,
    distance: float | None = None,
    metadata_columns: Sequence[str] = DEFAULT_NETWORK_METADATA_COLUMNS,
    dict_columns: Mapping[str, str | None] | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Download an OSM network and convert it to cityImage network schema.

    OSMnx owns graph acquisition and projection. cityImage owns pedestrian
    filtering, node/edge schema construction, and selected metadata retention.
    """
    crs = _normalise_crs(crs)

    if network_type in {"walk", "foot", "pedestrian"}:
        return pedestrian_network_from_osm(
            query,
            crs=crs,
            download_method=download_method,
            distance=distance,
        )

    _validate_download_method(download_method)

    try:
        if download_method == "OSMplace":
            graph = ox.graph_from_place(
                query,
                network_type=network_type,
                retain_all=True,
                simplify=True,
            )
        elif download_method == "distance_from_address":
            graph = ox.graph_from_address(
                query,
                dist=_distance_arg(distance, download_method),
                network_type=network_type,
                retain_all=True,
                simplify=True,
            )
        elif download_method == "distance_from_point":
            graph = ox.graph_from_point(
                query,
                dist=_distance_arg(distance, download_method),
                network_type=network_type,
                retain_all=True,
                simplify=True,
            )
        else:
            graph = ox.graph_from_polygon(
                query,
                network_type=network_type,
                retain_all=True,
                simplify=True,
            )
    except Exception as exc:
        if exc.__class__.__name__ == "InsufficientResponseError":
            return _empty_network(crs)
        raise

    return _network_from_osmnx_graph(
        graph,
        ox,
        crs,
        metadata_columns=metadata_columns,
        dict_columns=dict_columns,
    )


def _polygonal_buildings(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep valid polygonal building geometries."""
    return buildings[
        buildings.geometry.notna()
        & ~buildings.geometry.is_empty
        & buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()


def buildings_from_osm(
    query: Any,
    *,
    download_method: str = "OSMplace",
    crs: Any = None,
    distance: float | None = 1000,
    min_area: float | None = 200,
    default_land_use: str = "residential",
) -> gpd.GeoDataFrame:
    """Download OSM buildings and convert them to cityImage building schema.

    The function derives raw OSM land-use candidates, classifies them into
    cityImage land-use groups, removes very small polygons, and standardises
    building identifiers and area.
    """
    crs = _normalise_crs(crs)
    buildings = features_from_osm(
        query,
        {"building": True},
        download_method=download_method,
        distance=distance,
        crs=crs,
    )
    buildings = _polygonal_buildings(buildings)

    if crs is None:
        buildings = ox.projection.project_gdf(buildings)

    buildings = gdf_multipolygon_to_polygon(buildings)
    buildings = derive_land_uses_raw_fromOSM(buildings, default=default_land_use)
    buildings = classify_land_uses_raws_into_OSMgroups(
        buildings,
        land_uses_raw_column="land_uses_raw",
        new_group_column="land_uses",
    )

    buildings["area"] = buildings.geometry.area
    if min_area is not None:
        buildings = buildings[buildings["area"] >= min_area].copy()

    buildings = buildings.reset_index(drop=True)
    buildings["buildingID"] = buildings.index.astype(int)

    buildings["land_uses_overlap"] = buildings["land_uses"].apply(_unit_overlaps_for_land_uses)

    return standardize_buildings_gdf(
        buildings,
        building_id_column="buildingID",
        land_uses_raw_column="land_uses_raw",
    )


def barriers_from_osm(
    query: Any,
    *,
    download_method: str = "OSMplace",
    crs: Any = None,
    distance: float | None = None,
    include_primary: bool = True,
    include_secondary: bool = False,
    parks_min_area: float = 100000,
) -> gpd.GeoDataFrame:
    """Download OSM features and build the combined cityImage barrier layer.

    OSMnx downloads roads, railways, water, coastline, and parks. cityImage then
    applies its barrier extraction semantics and returns a single barrier
    GeoDataFrame.
    """
    crs = _normalise_crs(crs)
    feature_inputs = {
        name: features_from_osm(
            query,
            tags,
            download_method=download_method,
            distance=distance,
            crs=crs,
        )
        for name, tags in barrier_osm_feature_tags().items()
    }

    return barriers_from_osm_features(
        **feature_inputs,
        crs=crs,
        include_primary=include_primary,
        include_secondary=include_secondary,
        parks_min_area=parks_min_area,
    )
