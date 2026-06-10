"""File IO bridge helpers for cityImage.

GeoPandas owns file reading. cityImage owns conversion from loaded raw
GeoDataFrames into cityImage schemas and downstream semantics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import geopandas as gpd

from .adapters import standardize_buildings_gdf
from .buildings import select_buildings_by_study_area
from .geometry import gdf_multipolygon_to_polygon
from .network import network_from_lines
from .schema import LAND_USES_RAW


def _normalise_crs(crs: Any) -> Any:
    """Accept integer EPSG codes as a convenience."""
    return f"EPSG:{crs}" if isinstance(crs, int) else crs


def _polygonal_buildings(buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep valid polygonal building geometries."""
    return buildings[
        buildings.geometry.notna()
        & ~buildings.geometry.is_empty
        & buildings.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()


def network_from_file(
    input_path: str,
    crs: Any,
    *,
    dict_columns: Mapping[str, str | None] | None = None,
    other_columns: Sequence[str] | None = None,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load line geometries from file and convert them to cityImage network schema.

    Parameters
    ----------
    input_path : str
        Path to a vector file readable by GeoPandas.
    crs : Any
        Target CRS for output nodes and edges.
    dict_columns : Mapping[str, str | None], optional
        Mapping from cityImage edge columns to source columns.
    other_columns : Sequence[str], optional
        Additional source columns to preserve on the output edges.

    Returns
    -------
    tuple[geopandas.GeoDataFrame, geopandas.GeoDataFrame]
        Nodes and edges in cityImage schema.
    """
    crs = _normalise_crs(crs)
    edges_raw = gpd.read_file(input_path)
    return network_from_lines(
        edges_raw,
        crs,
        dict_columns=dict_columns,
        other_columns=other_columns or [],
    )


def buildings_from_file(
    input_path: str,
    crs: Any,
    *,
    case_study_area: Any = None,
    distance_from_center: float | None = None,
    height_field: str | None = None,
    base_field: str | None = None,
    land_uses_raw_field: str | None = None,
    min_area: float = 200,
    min_height: float = 5,
) -> gpd.GeoDataFrame:
    """Load building polygons from file and convert them to cityImage schema.

    GeoPandas handles file reading and CRS conversion. cityImage standardises
    identifiers, area, height/base defaults, and source/provenance land-use
    columns.
    """
    crs = _normalise_crs(crs)
    buildings = gpd.read_file(input_path).to_crs(crs).copy()
    buildings = _polygonal_buildings(buildings)

    buildings["area"] = buildings.geometry.area
    buildings = buildings[buildings["area"] >= min_area].copy()

    if height_field is not None and height_field not in buildings.columns:
        raise ValueError(f"height_field {height_field!r} not found in input file")
    if base_field is not None and base_field not in buildings.columns:
        raise ValueError(f"base_field {base_field!r} not found in input file")
    if land_uses_raw_field is not None and land_uses_raw_field not in buildings.columns:
        raise ValueError(f"land_uses_raw_field {land_uses_raw_field!r} not found in input file")

    if height_field is not None:
        buildings["height"] = buildings[height_field]
    elif "height" not in buildings.columns:
        buildings["height"] = min_height

    if base_field is not None:
        buildings["base"] = buildings[base_field]
    elif "base" not in buildings.columns:
        buildings["base"] = 0.0

    land_uses_raw_column = land_uses_raw_field
    if land_uses_raw_column is None and LAND_USES_RAW in buildings.columns:
        land_uses_raw_column = LAND_USES_RAW

    if "buildingID" not in buildings.columns:
        buildings = buildings.reset_index(drop=True)
        buildings["buildingID"] = buildings.index.astype(int)

    buildings = gdf_multipolygon_to_polygon(buildings, columnID="buildingID")
    buildings = standardize_buildings_gdf(
        buildings,
        building_id_column="buildingID",
        land_uses_raw_column=land_uses_raw_column,
        validate=False,
    )

    if case_study_area is not None:
        buildings = select_buildings_by_study_area(
            buildings,
            method="polygon",
            polygon=case_study_area,
        )
    elif distance_from_center not in (None, 0):
        buildings = select_buildings_by_study_area(
            buildings,
            method="distance",
            distance=float(distance_from_center),
        )

    return buildings.reset_index(drop=True)
