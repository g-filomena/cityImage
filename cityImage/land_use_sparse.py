"""Sparse/non-OSM land-use helpers.

This module is intentionally separate from the OSM land-use pipeline.

Use this route when land-use evidence comes from:
- a sparse attribute already attached to building footprints;
- a municipal / survey / classified land-use layer;
- any non-OSM source with labels that need to be mapped to the same
  macro-groups consumed downstream by landmark/pragmatic scoring.

The expected downstream schema is:

    land_uses          list[str]
    land_uses_overlap  list[float]

`land_uses_overlap` is optional for some consumers. When it is present, it
should align 1:1 with `land_uses` and sum to 1 per row.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from .land_use_utils import _to_list


NULL_STRINGS = {"", "none", "nan", "null", "na", "n/a"}


def _clean_label(value: Any) -> str | None:
    """Normalize one sparse land-use label to a comparable string."""
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    label = str(value).strip()
    if label.lower() in NULL_STRINGS:
        return None

    return label


def _mapping_lookup(mapping: dict[Any, Any] | None) -> dict[str, Any]:
    """Build a case-insensitive mapping lookup."""
    if mapping is None:
        return {}

    lookup: dict[str, Any] = {}
    for raw, target in mapping.items():
        cleaned = _clean_label(raw)
        if cleaned is not None:
            lookup[cleaned] = target
            lookup[cleaned.lower()] = target
    return lookup


def _mapped_groups(
    value: Any,
    lookup: dict[str, Any],
    *,
    default_land_use: str,
    preserve_unmapped: bool,
) -> list[str]:
    """Map one raw value to one or more target macro-groups."""
    cleaned = _clean_label(value)
    if cleaned is None:
        return []

    mapped = lookup.get(cleaned, lookup.get(cleaned.lower()))

    if mapped is None:
        mapped = cleaned if preserve_unmapped else default_land_use

    groups: list[str] = []
    seen: set[str] = set()

    for item in _to_list(mapped):
        label = _clean_label(item)
        if label is None or label in seen:
            continue
        seen.add(label)
        groups.append(label)

    return groups


def classify_sparse_land_uses(
    buildings_gdf: gpd.GeoDataFrame,
    source_column: str,
    target_column: str = "land_uses",
    mapping: dict[Any, Any] | None = None,
    default_land_use: str = "unknown",
    preserve_unmapped: bool = True,
) -> gpd.GeoDataFrame:
    """Classify sparse/non-OSM land-use attributes attached to buildings.

    The output column is always a list[str] column so it can feed directly into
    `pragmatic_score`.
    """
    if source_column not in buildings_gdf.columns:
        raise ValueError(f"GeoDataFrame must contain '{source_column}'")

    gdf = buildings_gdf.copy()
    lookup = _mapping_lookup(mapping)

    def _classify_cell(cell: Any) -> list[str]:
        groups: list[str] = []
        seen: set[str] = set()

        for raw_value in _to_list(cell):
            for group in _mapped_groups(
                raw_value,
                lookup,
                default_land_use=default_land_use,
                preserve_unmapped=preserve_unmapped,
            ):
                if group not in seen:
                    seen.add(group)
                    groups.append(group)

        return groups or [default_land_use]

    gdf[target_column] = gdf[source_column].apply(_classify_cell)
    return gdf


def _normalise_weight_dict(
    weights: dict[str, float],
    default_land_use: str,
) -> tuple[list[str], list[float]]:
    """Return stable labels and normalised weights from a group -> mass dict."""
    clean_weights = {
        group: float(weight)
        for group, weight in weights.items()
        if group and np.isfinite(float(weight)) and float(weight) > 0
    }

    if not clean_weights:
        return [default_land_use], [1.0]

    total = sum(clean_weights.values())
    labels = list(clean_weights.keys())
    overlaps = [clean_weights[label] / total for label in labels]
    return labels, overlaps


def attach_sparse_land_uses(
    buildings_gdf: gpd.GeoDataFrame,
    land_uses_gdf: gpd.GeoDataFrame,
    source_column: str,
    target_column: str = "land_uses",
    overlaps_column: str = "land_uses_overlap",
    mapping: dict[Any, Any] | None = None,
    default_land_use: str = "unknown",
    preserve_unmapped: bool = True,
    min_overlap_area: float = 0.0,
) -> gpd.GeoDataFrame:
    """Spatially attach sparse/non-OSM land-use labels to buildings.

    For polygon land-use layers, overlap weights are based on intersection area.
    For point/line layers, weights are based on intersecting feature counts.
    """
    if source_column not in land_uses_gdf.columns:
        raise ValueError(f"land_uses_gdf must contain '{source_column}'")

    if buildings_gdf.crs is not None and land_uses_gdf.crs is not None:
        if buildings_gdf.crs != land_uses_gdf.crs:
            raise ValueError("CRS mismatch: buildings_gdf and land_uses_gdf must have the same CRS")

    buildings = buildings_gdf.copy()
    buildings["_ci_sparse_row"] = range(len(buildings))

    # Force object dtype so nested lists are stored as scalar cell values.
    buildings[target_column] = pd.Series(
        [[default_land_use] for _ in range(len(buildings))],
        index=buildings.index,
        dtype="object",
    )
    buildings[overlaps_column] = pd.Series(
        [[1.0] for _ in range(len(buildings))],
        index=buildings.index,
        dtype="object",
    )

    if buildings.empty or land_uses_gdf.empty:
        return buildings.drop(columns=["_ci_sparse_row"], errors="ignore")

    land_uses = land_uses_gdf[[source_column, "geometry"]].copy()
    land_uses = land_uses[land_uses.geometry.notna()].copy()

    if land_uses.empty:
        return buildings.drop(columns=["_ci_sparse_row"], errors="ignore")

    lookup = _mapping_lookup(mapping)

    geom_types = set(land_uses.geometry.geom_type.dropna().unique())
    area_like = bool(geom_types) and all("Polygon" in geom_type for geom_type in geom_types)

    if area_like:
        intersections = gpd.overlay(
            buildings[["_ci_sparse_row", "geometry"]],
            land_uses[[source_column, "geometry"]],
            how="intersection",
            keep_geom_type=False,
        )
        if intersections.empty:
            return buildings.drop(columns=["_ci_sparse_row"], errors="ignore")

        intersections["_ci_weight"] = intersections.geometry.area.astype(float)
        intersections = intersections[intersections["_ci_weight"] > min_overlap_area]
    else:
        try:
            intersections = gpd.sjoin(
                buildings[["_ci_sparse_row", "geometry"]],
                land_uses[[source_column, "geometry"]],
                how="inner",
                predicate="intersects",
            )
        except TypeError:
            intersections = gpd.sjoin(
                buildings[["_ci_sparse_row", "geometry"]],
                land_uses[[source_column, "geometry"]],
                how="inner",
                op="intersects",
            )

        if intersections.empty:
            return buildings.drop(columns=["_ci_sparse_row"], errors="ignore")

        intersections["_ci_weight"] = 1.0

    grouped_weights: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for _, row in intersections.iterrows():
        row_id = int(row["_ci_sparse_row"])
        raw_weight = float(row["_ci_weight"])

        groups = _mapped_groups(
            row[source_column],
            lookup,
            default_land_use=default_land_use,
            preserve_unmapped=preserve_unmapped,
        )

        if not groups:
            continue

        split_weight = raw_weight / len(groups)
        for group in groups:
            grouped_weights[row_id][group] += split_weight

    for row_id, weights in grouped_weights.items():
        labels, overlaps = _normalise_weight_dict(weights, default_land_use)

        matched_index = buildings.index[buildings["_ci_sparse_row"] == row_id]
        if len(matched_index) == 0:
            continue

        # Use scalar .at assignment. Do not use:
        # buildings.loc[mask, column] = [labels]
        # because pandas may treat the nested list as a broadcastable ndarray.
        idx = matched_index[0]
        buildings.at[idx, target_column] = labels
        buildings.at[idx, overlaps_column] = overlaps

    return buildings.drop(columns=["_ci_sparse_row"], errors="ignore")
