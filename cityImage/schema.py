"""Schema helpers for cityImage core data tables.

This module defines the minimum column contracts used by the scoring layer.
It is intentionally lightweight: it validates data produced by OSMnx,
GeoPandas, momepy, cityseer, or any other upstream workflow without forcing
cityImage to own the whole loading/cleaning pipeline.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

NODE_ID = "nodeID"
EDGE_ID = "edgeID"
BUILDING_ID = "buildingID"
GEOMETRY = "geometry"

LAND_USES = "land_uses"
LAND_USES_OVERLAP = "land_uses_overlap"
LAND_USES_RAW = "land_uses_raw"

AREA = "area"
HEIGHT = "height"
BASE = "base"

REQUIRED_NODES_COLUMNS = (NODE_ID, GEOMETRY)
REQUIRED_EDGES_COLUMNS = (EDGE_ID, "u", "v", GEOMETRY)
REQUIRED_BUILDINGS_COLUMNS = (BUILDING_ID, GEOMETRY)

OPTIONAL_BUILDINGS_COLUMNS = (
    AREA,
    HEIGHT,
    BASE,
    LAND_USES,
    LAND_USES_OVERLAP,
    LAND_USES_RAW,
)


class SchemaError(ValueError):
    """Raised when a GeoDataFrame does not match a cityImage schema contract."""


@dataclass(frozen=True)
class SchemaReport:
    """Simple validation report returned by schema-checking helpers."""

    name: str
    ok: bool
    missing_columns: tuple[str, ...] = ()
    message: str = ""


def missing_columns(gdf: pd.DataFrame, required_columns: Iterable[str]) -> tuple[str, ...]:
    """Return required columns that are absent from a DataFrame."""
    return tuple(column for column in required_columns if column not in gdf.columns)


def require_columns(
    gdf: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    frame_name: str = "DataFrame",
) -> None:
    """Raise a SchemaError if required columns are missing."""
    missing = missing_columns(gdf, required_columns)
    if missing:
        missing_str = ", ".join(missing)
        raise SchemaError(f"{frame_name} is missing required columns: {missing_str}")


def require_geometry(
    gdf: gpd.GeoDataFrame,
    *,
    frame_name: str = "GeoDataFrame",
    allow_empty: bool = False,
) -> None:
    """Validate that a GeoDataFrame has a geometry column with valid entries."""
    if not isinstance(gdf, gpd.GeoDataFrame):
        raise SchemaError(f"{frame_name} must be a GeoDataFrame")

    if gdf.geometry.name not in gdf.columns:
        raise SchemaError(f"{frame_name} must have an active geometry column")

    if gdf.geometry.isna().any():
        raise SchemaError(f"{frame_name} contains missing geometries")

    if not allow_empty and gdf.geometry.is_empty.any():
        raise SchemaError(f"{frame_name} contains empty geometries")


def _is_list_like_cell(value: Any) -> bool:
    """Return True when a cell contains a non-string iterable."""
    return isinstance(value, (list, tuple, set))


def require_land_use_lists(
    buildings_gdf: pd.DataFrame,
    *,
    land_uses_column: str = LAND_USES,
    overlaps_column: str = LAND_USES_OVERLAP,
    frame_name: str = "buildings_gdf",
) -> None:
    """Validate list-like land-use columns used by pragmatic scoring."""
    require_columns(buildings_gdf, [land_uses_column], frame_name=frame_name)

    bad_land_uses = ~buildings_gdf[land_uses_column].apply(_is_list_like_cell)
    if bad_land_uses.any():
        raise SchemaError(f"{frame_name}.{land_uses_column} must contain list-like values")

    if overlaps_column not in buildings_gdf.columns:
        return

    bad_overlaps = ~buildings_gdf[overlaps_column].apply(_is_list_like_cell)
    if bad_overlaps.any():
        raise SchemaError(f"{frame_name}.{overlaps_column} must contain list-like values")

    lengths_match = buildings_gdf.apply(
        lambda row: len(row[land_uses_column]) == len(row[overlaps_column]),
        axis=1,
    )
    if not lengths_match.all():
        raise SchemaError(
            f"{frame_name}.{land_uses_column} and {frame_name}.{overlaps_column} "
            "must have matching list lengths"
        )


def validate_nodes_gdf(nodes_gdf: gpd.GeoDataFrame) -> SchemaReport:
    """Validate the minimal cityImage node table contract."""
    missing = missing_columns(nodes_gdf, REQUIRED_NODES_COLUMNS)
    if missing:
        return SchemaReport("nodes_gdf", False, missing, "missing required columns")

    require_geometry(nodes_gdf, frame_name="nodes_gdf")
    return SchemaReport("nodes_gdf", True)


def validate_edges_gdf(edges_gdf: gpd.GeoDataFrame) -> SchemaReport:
    """Validate the minimal cityImage edge table contract."""
    missing = missing_columns(edges_gdf, REQUIRED_EDGES_COLUMNS)
    if missing:
        return SchemaReport("edges_gdf", False, missing, "missing required columns")

    require_geometry(edges_gdf, frame_name="edges_gdf")
    return SchemaReport("edges_gdf", True)


def validate_buildings_gdf(
    buildings_gdf: gpd.GeoDataFrame,
    *,
    require_land_uses: bool = False,
    require_height: bool = False,
) -> SchemaReport:
    """Validate the minimal cityImage building table contract."""
    required = list(REQUIRED_BUILDINGS_COLUMNS)
    if require_height:
        required.append(HEIGHT)
    if require_land_uses:
        required.append(LAND_USES)

    missing = missing_columns(buildings_gdf, required)
    if missing:
        return SchemaReport("buildings_gdf", False, missing, "missing required columns")

    require_geometry(buildings_gdf, frame_name="buildings_gdf")

    if require_land_uses:
        require_land_use_lists(buildings_gdf)

    return SchemaReport("buildings_gdf", True)


def ensure_building_schema_defaults(
    buildings_gdf: gpd.GeoDataFrame,
    *,
    add_area: bool = True,
) -> gpd.GeoDataFrame:
    """Return buildings with common non-semantic cityImage defaults populated.

    This function fills geometry-derived and numeric defaults only. It does not
    classify land use and does not fabricate ``land_uses`` or
    ``land_uses_overlap``.
    """
    gdf = buildings_gdf.copy()
    require_columns(gdf, REQUIRED_BUILDINGS_COLUMNS, frame_name="buildings_gdf")
    require_geometry(gdf, frame_name="buildings_gdf")

    if add_area and AREA not in gdf.columns:
        gdf[AREA] = gdf.geometry.area

    if HEIGHT not in gdf.columns:
        gdf[HEIGHT] = np.nan

    if BASE not in gdf.columns:
        gdf[BASE] = 0.0

    return gdf
