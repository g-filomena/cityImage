"""High-level building landmark scoring facade.

This module is the thin semantic layer that should become the main public
entry point for cityImage scoring. It consumes already-prepared GeoDataFrames
and delegates component computations to ``cityImage.landmarks`` only when a
score is actually requested.

The purpose is to keep the scientific cityImage API separate from data loading,
network cleaning, plotting, and optional heavyweight dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any

import geopandas as gpd

from .schema import (
    LAND_USES,
    LAND_USES_OVERLAP,
    ensure_building_schema_defaults,
    require_land_use_lists,
    validate_buildings_gdf,
    validate_edges_gdf,
)

DEFAULT_GLOBAL_INDEX_WEIGHTS: dict[str, float] = {
    "fac": 0.30,
    "height": 0.20,
    "3dvis": 0.50,
    "area": 0.40,
    "2dvis": 0.00,
    "neigh": 0.30,
    "road": 0.30,
}

DEFAULT_GLOBAL_COMPONENT_WEIGHTS: dict[str, float] = {
    "vScore": 0.25,
    "sScore": 0.35,
    "cScore": 0.10,
    "pScore": 0.30,
}

DEFAULT_LOCAL_INDEX_WEIGHTS: dict[str, float] = DEFAULT_GLOBAL_INDEX_WEIGHTS.copy()
DEFAULT_LOCAL_COMPONENT_WEIGHTS: dict[str, float] = DEFAULT_GLOBAL_COMPONENT_WEIGHTS.copy()


@dataclass(frozen=True)
class LandmarkScoringConfig:
    """Configuration for high-level cityImage building scoring."""

    global_index_weights: Mapping[str, float] = field(
        default_factory=lambda: DEFAULT_GLOBAL_INDEX_WEIGHTS.copy()
    )
    global_component_weights: Mapping[str, float] = field(
        default_factory=lambda: DEFAULT_GLOBAL_COMPONENT_WEIGHTS.copy()
    )
    local_index_weights: Mapping[str, float] = field(
        default_factory=lambda: DEFAULT_LOCAL_INDEX_WEIGHTS.copy()
    )
    local_component_weights: Mapping[str, float] = field(
        default_factory=lambda: DEFAULT_LOCAL_COMPONENT_WEIGHTS.copy()
    )
    pragmatic_search_radius: float = 1500.0
    local_rescaling_radius: float = 1500.0
    visibility_method: str = "longest"


def _landmark_module() -> Any:
    """Import the implementation module only when scoring is requested."""
    return import_module(".landmarks", __package__)


def validate_score_weights(
    weights: Mapping[str, float],
    *,
    name: str = "weights",
    expected_sum: float | None = 1.0,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    """Return numeric weights after validating their values and optional sum."""
    if not weights:
        raise ValueError(f"{name} must not be empty")

    clean_weights: dict[str, float] = {}
    for key, value in weights.items():
        clean_value = float(value)
        if clean_value < 0:
            raise ValueError(f"{name}.{key} must be non-negative")
        clean_weights[str(key)] = clean_value

    if expected_sum is not None:
        actual_sum = sum(clean_weights.values())
        if abs(actual_sum - expected_sum) > tolerance:
            raise ValueError(
                f"{name} must sum to {expected_sum}; got {actual_sum:.12g}"
            )

    return clean_weights


def _validate_buildings_for_scoring(
    buildings_gdf: gpd.GeoDataFrame,
    *,
    require_land_uses: bool = False,
) -> None:
    """Validate the building schema and raise a concise error if invalid."""
    report = validate_buildings_gdf(buildings_gdf, require_land_uses=require_land_uses)
    if not report.ok:
        missing = ", ".join(report.missing_columns)
        raise ValueError(f"buildings_gdf does not match cityImage schema: {missing}")


def score_building_components(
    buildings_gdf: gpd.GeoDataFrame,
    *,
    edges_gdf: gpd.GeoDataFrame | None = None,
    obstructions_gdf: gpd.GeoDataFrame | None = None,
    sight_lines: gpd.GeoDataFrame | None = None,
    historic_elements_gdf: gpd.GeoDataFrame | None = None,
    cultural_score_column: str | None = None,
    cultural_from_osm: bool = False,
    compute_structural: bool = False,
    compute_visibility: bool = True,
    compute_cultural: bool = True,
    compute_pragmatic: bool = True,
    visibility_method: str = "longest",
    pragmatic_search_radius: float = 1500.0,
    land_uses_column: str = LAND_USES,
    land_uses_overlap_column: str = LAND_USES_OVERLAP,
    structural_kwargs: Mapping[str, Any] | None = None,
    validate_schema: bool = True,
) -> gpd.GeoDataFrame:
    """Compute requested building-level cityImage component scores.

    This function intentionally assumes that data has already been loaded and
    standardised. Use ``standardize_buildings_gdf`` and related adapters before
    calling it when inputs come from external workflows.
    """
    buildings = ensure_building_schema_defaults(buildings_gdf)

    if validate_schema:
        _validate_buildings_for_scoring(buildings, require_land_uses=compute_pragmatic)

    impl = _landmark_module()

    if compute_structural:
        if edges_gdf is None:
            raise ValueError("compute_structural=True requires edges_gdf")
        if validate_schema:
            report = validate_edges_gdf(edges_gdf)
            if not report.ok:
                missing = ", ".join(report.missing_columns)
                raise ValueError(f"edges_gdf does not match cityImage schema: {missing}")
        buildings = impl.structural_score(
            buildings,
            obstructions_gdf,
            edges_gdf,
            **dict(structural_kwargs or {}),
        )

    if compute_visibility:
        buildings = impl.visibility_score(
            buildings,
            sight_lines=sight_lines,
            method=visibility_method,
        )

    if compute_cultural:
        buildings = impl.cultural_score(
            buildings,
            historic_elements_gdf=historic_elements_gdf,
            score_column=cultural_score_column,
            from_OSM=cultural_from_osm,
        )

    if compute_pragmatic:
        require_land_use_lists(
            buildings,
            land_uses_column=land_uses_column,
            overlaps_column=land_uses_overlap_column,
        )
        buildings = impl.pragmatic_score(
            buildings,
            search_radius=pragmatic_search_radius,
            land_uses_column=land_uses_column,
            overlaps_column=land_uses_overlap_column,
        )

    return buildings


def score_buildings_global(
    buildings_gdf: gpd.GeoDataFrame,
    *,
    index_weights: Mapping[str, float] | None = None,
    component_weights: Mapping[str, float] | None = None,
) -> gpd.GeoDataFrame:
    """Compute global cityImage landmarkness scores from component columns."""
    index_weights = dict(index_weights or DEFAULT_GLOBAL_INDEX_WEIGHTS)
    component_weights = validate_score_weights(
        component_weights or DEFAULT_GLOBAL_COMPONENT_WEIGHTS,
        name="component_weights",
    )
    return _landmark_module().compute_global_scores(
        buildings_gdf,
        index_weights,
        component_weights,
    )


def score_buildings_local(
    buildings_gdf: gpd.GeoDataFrame,
    *,
    index_weights: Mapping[str, float] | None = None,
    component_weights: Mapping[str, float] | None = None,
    rescaling_radius: float = 1500.0,
) -> gpd.GeoDataFrame:
    """Compute local cityImage landmarkness scores from component columns."""
    index_weights = dict(index_weights or DEFAULT_LOCAL_INDEX_WEIGHTS)
    component_weights = validate_score_weights(
        component_weights or DEFAULT_LOCAL_COMPONENT_WEIGHTS,
        name="component_weights",
    )
    return _landmark_module().compute_local_scores(
        buildings_gdf,
        index_weights,
        component_weights,
        rescaling_radius=rescaling_radius,
    )


def score_cityimage_buildings(
    buildings_gdf: gpd.GeoDataFrame,
    *,
    config: LandmarkScoringConfig | None = None,
    compute_global: bool = True,
    compute_local: bool = False,
    **component_kwargs: Any,
) -> gpd.GeoDataFrame:
    """Run the high-level building component/global/local scoring pipeline."""
    config = config or LandmarkScoringConfig()

    buildings = score_building_components(
        buildings_gdf,
        visibility_method=config.visibility_method,
        pragmatic_search_radius=config.pragmatic_search_radius,
        **component_kwargs,
    )

    if compute_global:
        buildings = score_buildings_global(
            buildings,
            index_weights=config.global_index_weights,
            component_weights=config.global_component_weights,
        )

    if compute_local:
        buildings = score_buildings_local(
            buildings,
            index_weights=config.local_index_weights,
            component_weights=config.local_component_weights,
            rescaling_radius=config.local_rescaling_radius,
        )

    return buildings
