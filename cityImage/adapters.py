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


def _as_raw_land_use_column(series: pd.Series) -> pd.Series:
    """Convert source/provenance land-use values to list-valued cells."""
    return series.apply(_to_list).astype("object")


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
    """Return a buildings GeoDataFrame matching the cityImage building schema.

    This adapter normalises identifiers, geometry-derived attributes, numeric
    defaults, and source/provenance land-use values. It does not classify land
    use and does not create ``land_uses`` or ``land_uses_overlap``.

    Use ``land_uses_raw_column`` to map an unclassified source field such as
    ``land_use`` into ``land_uses_raw`` as list-like values. The older
    ``land_uses_column`` argument is retained as a compatibility alias for a
    raw/source land-use field only; it no longer creates semantic
    ``land_uses``. Semantic ``land_uses`` and matching overlap weights should
    be produced by an explicit classifier/assignment step before or after this
    adapter.
    """
    del default_land_use  # retained for backward-compatible call signatures

    _require_geodataframe(buildings_gdf, frame_name="buildings_gdf")
    gdf = _copy_or_view(buildings_gdf, copy)

    if building_id_column is not None:
        require_columns(gdf, [building_id_column], frame_name="buildings_gdf")
        gdf[BUILDING_ID] = gdf[building_id_column].to_numpy()
    elif BUILDING_ID not in gdf.columns:
        gdf[BUILDING_ID] = range(len(gdf))

    if land_uses_overlap_column is not None:
        raise ValueError(
            "standardize_buildings_gdf does not create land_uses_overlap. "
            "Create semantic land_uses and overlap weights in an explicit "
            "classifier/assignment step."
        )

    raw_source_column = land_uses_raw_column
    if raw_source_column is None:
        raw_source_column = land_uses_column

    if raw_source_column is not None:
        require_columns(gdf, [raw_source_column], frame_name="buildings_gdf")
        gdf[LAND_USES_RAW] = _as_raw_land_use_column(gdf[raw_source_column])
    elif LAND_USES_RAW in gdf.columns:
        gdf[LAND_USES_RAW] = _as_raw_land_use_column(gdf[LAND_USES_RAW])

    if LAND_USES_OVERLAP in gdf.columns and LAND_USES not in gdf.columns:
        raise ValueError("land_uses_overlap requires land_uses.")

    gdf = ensure_building_schema_defaults(
        gdf,
        add_area=add_area,
    )

    if add_area and AREA in gdf.columns:
        gdf[AREA] = gdf[AREA].fillna(gdf.geometry.area)

    if validate:
        report = validate_buildings_gdf(
            gdf,
            require_land_uses=LAND_USES in gdf.columns,
        )
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
