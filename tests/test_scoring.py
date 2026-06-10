from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Polygon

import cityImage as ci
import cityImage.scoring as scoring


def _buildings_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "buildingID": [10, 11],
            "land_uses": [["retail"], ["residential"]],
            "land_uses_overlap": [[1.0], [1.0]],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(2, 0), (3, 0), (3, 1), (2, 1)]),
            ],
        },
        geometry="geometry",
        crs="EPSG:3857",
    )


class _FakeLandmarkModule:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def visibility_score(self, buildings_gdf, sight_lines=None, method="longest"):
        self.calls.append(f"visibility:{method}")
        out = buildings_gdf.copy()
        out["3dvis"] = 0.0
        out["fac"] = 0.0
        return out

    def cultural_score(
        self,
        buildings_gdf,
        historic_elements_gdf=None,
        score_column=None,
        from_OSM=False,
    ):
        self.calls.append("cultural")
        out = buildings_gdf.copy()
        out["cult"] = 0.0
        return out

    def pragmatic_score(
        self,
        buildings_gdf,
        search_radius=1500,
        land_uses_column="land_uses",
        overlaps_column="land_uses_overlap",
    ):
        self.calls.append(f"pragmatic:{search_radius}")
        out = buildings_gdf.copy()
        out["prag"] = 0.0
        return out

    def compute_global_scores(
        self,
        buildings_gdf,
        global_indexes_weights,
        global_components_weights,
    ):
        self.calls.append("global")
        out = buildings_gdf.copy()
        out["gScore"] = 0.0
        return out

    def compute_local_scores(
        self,
        buildings_gdf,
        local_indexes_weights,
        local_components_weights,
        rescaling_radius=1500,
    ):
        self.calls.append(f"local:{rescaling_radius}")
        out = buildings_gdf.copy()
        out["lScore"] = 0.0
        return out


def test_validate_score_weights_accepts_valid_weights():
    weights = scoring.validate_score_weights({"a": 0.25, "b": 0.75}, name="test")
    assert weights == {"a": 0.25, "b": 0.75}


def test_validate_score_weights_rejects_invalid_sum():
    with pytest.raises(ValueError, match="must sum"):
        scoring.validate_score_weights({"a": 0.25, "b": 0.25}, name="test")


def test_score_building_components_uses_component_sequence(monkeypatch):
    fake = _FakeLandmarkModule()
    monkeypatch.setattr(scoring, "_landmark_module", lambda: fake)

    scored = ci.score_building_components(
        _buildings_gdf(),
        visibility_method="combined",
        pragmatic_search_radius=250,
    )

    assert fake.calls == ["visibility:combined", "cultural", "pragmatic:250"]
    assert {"3dvis", "fac", "cult", "prag"}.issubset(scored.columns)


def test_score_cityimage_buildings_can_run_pipeline_facade(monkeypatch):
    fake = _FakeLandmarkModule()
    monkeypatch.setattr(scoring, "_landmark_module", lambda: fake)

    config = ci.LandmarkScoringConfig(local_rescaling_radius=300)
    scored = ci.score_cityimage_buildings(
        _buildings_gdf(),
        config=config,
        compute_global=True,
        compute_local=True,
    )

    assert fake.calls == ["visibility:longest", "cultural", "pragmatic:1500.0", "global", "local:300"]
    assert {"gScore", "lScore"}.issubset(scored.columns)
