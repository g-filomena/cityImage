"""Landmark and imageability scoring tests.

Merged from:
- test_landmarks_module.py
- test_landmarks_unit.py
- test_regression_landmark_scoring.py
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

import cityImage as ci
from cityImage import landmarks
from tests.fixtures.cityimage_minimal import (
    historic_elements,
    minimal_buildings_with_land_use,
    sight_lines,
)


def _buildings():
    return gpd.GeoDataFrame(
        {
            "buildingID": [0, 1],
            "height": [10.0, 5.0],
            "historic": ["yes", None],
            "land_uses": [["religious"], ["residential"]],
        },
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
        ],
        crs="EPSG:3857",
    )


# ---------------------------------------------------------------------------
# Module/public API routing
# ---------------------------------------------------------------------------


def test_core_landmark_symbols_resolve_from_landmarks_module():
    assert ci.structural_score is landmarks.structural_score
    assert ci.visibility_score is landmarks.visibility_score
    assert ci.cultural_score is landmarks.cultural_score
    assert ci.pragmatic_score is landmarks.pragmatic_score
    assert ci.compute_global_scores is landmarks.compute_global_scores
    assert ci.compute_local_scores is landmarks.compute_local_scores


def test_historic_osm_loader_is_not_part_of_core_public_api():
    assert not hasattr(ci, "get_historic_buildings_fromOSM")


# ---------------------------------------------------------------------------
# Unit semantics
# ---------------------------------------------------------------------------


def test_visibility_score_empty_sight_lines_returns_gdf_not_tuple():
    out = ci.visibility_score(
        _buildings(),
        sight_lines=gpd.GeoDataFrame(geometry=[], crs="EPSG:3857"),
    )

    assert hasattr(out, "geometry")
    assert "3dvis" in out.columns
    assert out["3dvis"].eq(0.0).all()


def test_cultural_score_from_osm_uses_historic_helper():
    out = ci.cultural_score(_buildings(), from_OSM=True)

    assert out["cult"].tolist() == [1.0, 0.0]


def test_pragmatic_score_accepts_normalised_land_uses_without_overlap_column():
    out = ci.pragmatic_score(_buildings(), search_radius=10)

    assert "prag" in out.columns
    assert out["prag"].notna().all()


# ---------------------------------------------------------------------------
# Regression semantics
# ---------------------------------------------------------------------------


def test_component_scoring_preserves_visibility_cultural_and_pragmatic_outputs():
    scored = ci.score_building_components(
        minimal_buildings_with_land_use(),
        sight_lines=sight_lines(),
        historic_elements_gdf=historic_elements(),
        cultural_score_column="importance",
        compute_structural=False,
        compute_visibility=True,
        compute_cultural=True,
        compute_pragmatic=True,
        pragmatic_search_radius=10,
    ).sort_values("buildingID")

    rows = scored[["buildingID", "fac", "3dvis", "cult", "prag"]].to_dict("records")

    assert rows == [
        {"buildingID": 1, "fac": 20.0, "3dvis": 0.0, "cult": 2.0, "prag": 0.5},
        {"buildingID": 2, "fac": 10.0, "3dvis": 0.0, "cult": 0.0, "prag": 0.5},
        {"buildingID": 3, "fac": 30.0, "3dvis": 1.0, "cult": 0.0, "prag": 1.0},
    ]


def test_global_scoring_preserves_landmarkness_composition():
    components = ci.score_building_components(
        minimal_buildings_with_land_use(),
        sight_lines=sight_lines(),
        historic_elements_gdf=historic_elements(),
        cultural_score_column="importance",
        compute_structural=False,
        compute_visibility=True,
        compute_cultural=True,
        compute_pragmatic=True,
        pragmatic_search_radius=10,
    )
    scored = ci.score_buildings_global(components).sort_values("buildingID")

    assert scored["sScore"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert scored["cScore"].tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert scored["pScore"].tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert scored["gScore"].tolist() == pytest.approx([0.1625, 0.0, 0.55])
    assert scored["gScore_sc"].tolist() == pytest.approx([0.2954545454545454, 0.0, 1.0])
