"""Thin adapters from external GeoDataFrames to cityImage schemas.

These helpers are deliberately small. They do not download data, clean an
entire street network, classify land use, or compute scores. Their job is to
normalise data produced by OSMnx, GeoPandas, momepy, cityseer, or custom
workflows into the minimal cityImage column contracts defined in
``cityImage.schema``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import geopandas as gpd
import pandas as pd

from .landuse.utils import _to_list
from .schema import (
    AREA,
    BUILDING_ID,
    EDGE_ID,
    LAND_USES,
    LAND_USES_OVERLAP,
    LAND_USES_RAW,
    NODE_ID,
    ensure_building_schema_defaults,
    require_columns,
    require_geometry,
    validate_buildings_gdf,
    validate_edges_gdf,
    validate_nodes_gdf,
)


def _copy_or_view(gdf: gpd.GeoDataFrame, copy: bool) -> gpd.GeoDataFrame:
    """Return a copy when requested, otherwise return the original object."""
    return gdf.copy() if copy else gdf


def _require_geodataframe(gdf: Any, *, frame_name: str) -> gpd.GeoDataFrame:
    """Raise a clear error if an adapter input is not a GeoDataFrame."""
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise TypeError(f"{frame_name} must be a GeoDataFrame")
    require_geometry(gdf, frame_name=frame_name)
    return gdf


def _index_level_as_series(gdf: gpd.GeoDataFrame, level: str | int) -> pd.Series:
    """Return one index level as a Series aligned to ``gdf.index``."""
    values = gdf.index.get_level_values(level)
    return pd.Series(values.to_numpy(), index=gdf.index)


def _column_or_index_level(
    gdf: gpd.GeoDataFrame,
    column: str,
    *,
    fallback_level: int | None = None,
) -> pd.Series | None:
    """Get a value column from either columns or index levels."""
    if column in gdf.columns:
        return gdf[column]

    if column in gdf.index.names:
        return _index_level_as_series(gdf, column)

    if fallback_level is not None and gdf.index.nlevels > fallback_level:
        return _index_level_as_series(gdf, fallback_level)

    return None


def _as_list_column(series: pd.Series, *, default: str = "unknown") -> pd.Series:
    """Convert a scalar/list-like Series to list-valued cells."""

    def _normalise_cell(value: Any) -> list[Any]:
        values = _to_list(value)
        return values if values else [default]

    return series.apply(_normalise_cell).astype("object")


def standardize_nodes_gdf(
    nodes_gdf: gpd.GeoDataFrame,
    *,
    node_id_column: str | None = None,
    copy: bool = True,
    validate: bool = True,
) -> gpd.GeoDataFrame:
    """Return a nodes GeoDataFrame matching the minimal cityImage schema.

    The output always contains ``nodeID`` and an active geometry column. If
    ``nodeID`` is absent, the current index is used. Point geometries are also
    used to populate missing ``x``/``y`` columns.
    """
    _require_geodataframe(nodes_gdf, frame_name="nodes_gdf")
    gdf = _copy_or_view(nodes_gdf, copy)

    if node_id_column is not None:
        require_columns(gdf, [node_id_column], frame_name="nodes_gdf")
        gdf[NODE_ID] = gdf[node_id_column].to_numpy()
    elif NODE_ID not in gdf.columns:
        gdf[NODE_ID] = gdf.index.to_numpy()

    if "x" not in gdf.columns and all(gdf.geometry.geom_type == "Point"):
        gdf["x"] = gdf.geometry.x
    if "y" not in gdf.columns and all(gdf.geometry.geom_type == "Point"):
        gdf["y"] = gdf.geometry.y

    if validate:
        report = validate_nodes_gdf(gdf)
        if not report.ok:
            missing = ", ".join(report.missing_columns)
            raise ValueError(f"nodes_gdf does not match cityImage schema: {missing}")

    return gdf


def standardize_edges_gdf(
    edges_gdf: gpd.GeoDataFrame,
    *,
    edge_id_column: str | None = None,
    u_column: str = "u",
    v_column: str = "v",
    copy: bool = True,
    validate: bool = True,
) -> gpd.GeoDataFrame:
    """Return an edges GeoDataFrame matching the minimal cityImage schema.

    This is compatible with OSMnx edge GeoDataFrames using a MultiIndex
    ``(u, v, key)`` as well as custom edge tables that already expose ``u`` and
    ``v`` columns.
    """
    _require_geodataframe(edges_gdf, frame_name="edges_gdf")
    gdf = _copy_or_view(edges_gdf, copy)

    if edge_id_column is not None:
        require_columns(gdf, [edge_id_column], frame_name="edges_gdf")
        gdf[EDGE_ID] = gdf[edge_id_column].to_numpy()
    elif EDGE_ID not in gdf.columns:
        gdf[EDGE_ID] = range(len(gdf))

    u_values = _column_or_index_level(gdf, u_column, fallback_level=0)
    v_values = _column_or_index_level(gdf, v_column, fallback_level=1)

    if u_values is None or v_values is None:
        raise ValueError("edges_gdf must contain u/v columns or a compatible edge MultiIndex")

    gdf["u"] = u_values.to_numpy()
    gdf["v"] = v_values.to_numpy()

    if "length" not in gdf.columns:
        gdf["length"] = gdf.geometry.length

    if validate:
        report = validate_edges_gdf(gdf)
        if not report.ok:
            missing = ", ".join(report.missing_columns)
            raise ValueError(f"edges_gdf does not match cityImage schema: {missing}")

    return gdf


def standardize_buildings_gdf(
    buildings_gdf: gpd.GeoDataFrame,
    *,
    building_id_column: str | None = None,
    land_uses_column: str | None = None,
    land_uses_raw_column: str | None = None,
    land_uses_overlap_column: str | None = None,
    default_land_use: str = "unknown",
    add_area: bool = True,
    copy: bool = True,
    validate: bool = True,
) -> gpd.GeoDataFrame:
    """Return a buildings GeoDataFrame matching the cityImage schema.

    The function only normalises column names and safe defaults. It does not
    classify land-use labels. Use ``classify_sparse_land_uses`` or the OSM
    land-use route when semantic classification is needed.
    """
    _require_geodataframe(buildings_gdf, frame_name="buildings_gdf")
    gdf = _copy_or_view(buildings_gdf, copy)

    if building_id_column is not None:
        require_columns(gdf, [building_id_column], frame_name="buildings_gdf")
        gdf[BUILDING_ID] = gdf[building_id_column].to_numpy()
    elif BUILDING_ID not in gdf.columns:
        gdf[BUILDING_ID] = range(len(gdf))

    if land_uses_column is not None:
        require_columns(gdf, [land_uses_column], frame_name="buildings_gdf")
        gdf[LAND_USES] = _as_list_column(gdf[land_uses_column], default=default_land_use)

    if land_uses_raw_column is not None:
        require_columns(gdf, [land_uses_raw_column], frame_name="buildings_gdf")
        gdf[LAND_USES_RAW] = gdf[land_uses_raw_column]

    if land_uses_overlap_column is not None:
        require_columns(gdf, [land_uses_overlap_column], frame_name="buildings_gdf")
        gdf[LAND_USES_OVERLAP] = gdf[land_uses_overlap_column].apply(_to_list).astype("object")

    gdf = ensure_building_schema_defaults(
        gdf,
        default_land_use=default_land_use,
        add_area=add_area,
    )

    if add_area and AREA in gdf.columns:
        gdf[AREA] = gdf[AREA].fillna(gdf.geometry.area)

    if validate:
        report = validate_buildings_gdf(gdf, require_land_uses=True)
        if not report.ok:
            missing = ", ".join(report.missing_columns)
            raise ValueError(f"buildings_gdf does not match cityImage schema: {missing}")

    return gdf


def standardize_cityimage_inputs(
    *,
    nodes_gdf: gpd.GeoDataFrame | None = None,
    edges_gdf: gpd.GeoDataFrame | None = None,
    buildings_gdf: gpd.GeoDataFrame | None = None,
    node_kwargs: Mapping[str, Any] | None = None,
    edge_kwargs: Mapping[str, Any] | None = None,
    building_kwargs: Mapping[str, Any] | None = None,
) -> dict[str, gpd.GeoDataFrame]:
    """Standardize any provided cityImage input tables.

    Returns a dictionary containing only the inputs that were provided. This is
    useful in notebooks and adapters where an upstream workflow may provide a
    network, buildings, or both.
    """
    outputs: dict[str, gpd.GeoDataFrame] = {}

    if nodes_gdf is not None:
        outputs["nodes_gdf"] = standardize_nodes_gdf(nodes_gdf, **dict(node_kwargs or {}))
    if edges_gdf is not None:
        outputs["edges_gdf"] = standardize_edges_gdf(edges_gdf, **dict(edge_kwargs or {}))
    if buildings_gdf is not None:
        outputs["buildings_gdf"] = standardize_buildings_gdf(
            buildings_gdf,
            **dict(building_kwargs or {}),
        )

    return outputs
