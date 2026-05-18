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

from .adapters import standardize_buildings_gdf
from .barriers import barrier_osm_feature_tags, barriers_from_osm_features
from .geometry import gdf_multipolygon_to_polygon
from .landuse import classify_land_uses_raws_into_OSMgroups, derive_land_uses_raw_fromOSM
from .network import network_from_lines
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


def _require_osmnx():
    """Import OSMnx only when an OSM acquisition helper is called."""
    try:
        import osmnx as ox
    except ImportError as exc:
        raise ImportError(
            "OSM acquisition helpers require the optional 'osm' extra. "
            "Install with: pip install cityImage[osm]"
        ) from exc

    return ox


def _normalise_crs(crs: Any) -> Any:
    """Accept integer EPSG codes as a convenience."""
    return f"EPSG:{crs}" if isinstance(crs, int) else crs


def _validate_download_method(download_method: str) -> None:
    if download_method not in OSM_DOWNLOAD_METHODS:
        raise ValueError(
            "download_method must be one of: "
            f"{', '.join(sorted(OSM_DOWNLOAD_METHODS))}"
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
    ox = _require_osmnx()

    if download_method == "OSMplace":
        features = ox.features_from_place(query, tags=tags)
    elif download_method == "distance_from_address":
        features = ox.features_from_address(query, tags=tags, dist=distance or 500)
    elif download_method == "distance_from_point":
        features = ox.features_from_point(query, tags=tags, dist=distance or 500)
    else:
        features = ox.features_from_polygon(query, tags=tags)

    crs = _normalise_crs(crs)
    if crs is not None:
        features = features.to_crs(crs)

    return features


def network_from_osm(
    query: Any,
    *,
    download_method: str = "OSMplace",
    network_type: str = "walk",
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
        if crs is None:
            raise ValueError("crs is required for pedestrian network acquisition")
        return pedestrian_network_from_osm(
            query,
            crs=crs,
            download_method=download_method,
            distance=distance or 500,
        )

    _validate_download_method(download_method)
    ox = _require_osmnx()

    if download_method == "OSMplace":
        graph = ox.graph_from_place(query, network_type=network_type, simplify=True)
    elif download_method == "distance_from_address":
        graph = ox.graph_from_address(
            query,
            dist=distance or 500,
            network_type=network_type,
            simplify=True,
        )
    elif download_method == "distance_from_point":
        graph = ox.graph_from_point(
            query,
            dist=distance or 500,
            network_type=network_type,
            simplify=True,
        )
    else:
        graph = ox.graph_from_polygon(query, network_type=network_type, simplify=True)

    graph = ox.project_graph(graph, to_crs=crs) if crs is not None else ox.project_graph(graph)
    crs = crs or graph.graph.get("crs")

    _, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    other_columns = [column for column in metadata_columns if column in edges.columns]

    return network_from_lines(
        edges.reset_index(drop=True),
        crs,
        dict_columns=dict_columns,
        other_columns=other_columns,
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
    min_area: float = 200,
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
        ox = _require_osmnx()
        buildings = ox.projection.project_gdf(buildings)

    buildings = gdf_multipolygon_to_polygon(buildings)
    buildings = derive_land_uses_raw_fromOSM(buildings, default=default_land_use)
    buildings = classify_land_uses_raws_into_OSMgroups(
        buildings,
        land_uses_raw_column="land_uses_raw",
        new_group_column="land_uses",
    )

    buildings["area"] = buildings.geometry.area
    buildings = buildings[buildings["area"] >= min_area].copy()
    buildings = buildings.reset_index(drop=True)
    buildings["buildingID"] = buildings.index.astype(int)

    return standardize_buildings_gdf(
        buildings,
        building_id_column="buildingID",
        land_uses_column="land_uses",
        land_uses_raw_column="land_uses_raw",
        default_land_use=default_land_use,
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

