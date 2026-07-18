"""Offline tests for landmark scoring (``cityImage.landmarks``).

Exercises the structural / visibility / cultural / pragmatic sub-scores and the
global + local aggregators on small synthetic building fixtures.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

import cityImage as ci

CRS = "EPSG:3857"

GLOBAL_INDEXES = {
    "3dvis": 0.5,
    "fac": 0.3,
    "height": 0.2,
    "area": 0.3,
    "2dvis": 0.3,
    "neigh": 0.2,
    "road": 0.2,
}
GLOBAL_COMPONENTS = {"vScore": 0.5, "sScore": 0.3, "cScore": 0.1, "pScore": 0.1}
LOCAL_COMPONENTS = {"vScore": 0.25, "sScore": 0.35, "cScore": 0.1, "pScore": 0.3}


def _square(x0, side=10.0):
    return Polygon([(x0, 0), (x0 + side, 0), (x0 + side, side), (x0, side)])


def _buildings(n=4):
    gdf = gpd.GeoDataFrame(
        {"buildingID": list(range(n))},
        geometry=[_square(i * 15.0) for i in range(n)],
        crs=CRS,
    )
    gdf["area"] = gdf.geometry.area
    return gdf


def _scored_buildings(n=4):
    gdf = _buildings(n)
    rng = np.arange(n, dtype=float)
    gdf["height"] = 10.0 + rng * 5.0
    gdf["3dvis"] = rng
    gdf["fac"] = 100.0 + rng * 10.0
    gdf["2dvis"] = rng * 2.0
    gdf["cult"] = rng
    gdf["prag"] = rng / (n - 1)
    gdf["neigh"] = rng
    gdf["road"] = 5.0 + rng
    return gdf


def test_structural_score_empty_buildings_returns_typed_columns():
    empty = _buildings(0)
    out = ci.structural_score(empty, obstructions_gdf=None, edges_gdf=_buildings(1))
    assert {"road", "2dvis", "neigh"}.issubset(out.columns)
    assert len(out) == 0


def test_structural_score_computes_distance_visibility_and_neighbours():
    buildings = _buildings(3)
    edges = gpd.GeoDataFrame({"edgeID": [1]}, geometry=[LineString([(-5, -5), (100, -5)])], crs=CRS)
    out = ci.structural_score(buildings, obstructions_gdf=None, edges_gdf=edges)
    assert (out["road"] > 0).all()  # all buildings sit above the street line
    assert (out["neigh"] >= 1).all()  # a building is its own neighbour at minimum
    assert (out["2dvis"] > 0).all()  # each building has a non-degenerate advance-visibility area


def test_visibility_score_uses_sight_lines_and_height():
    buildings = _buildings(3)
    buildings["height"] = [10.0, 20.0, 30.0]
    sight_lines = gpd.GeoDataFrame(
        {"nodeID": [1, 1, 2], "buildingID": [0, 0, 1]},
        geometry=[
            LineString([(0, 0), (0, 5)]),
            LineString([(0, 0), (0, 20)]),  # longest for building 0
            LineString([(0, 0), (0, 12)]),
        ],
        crs=CRS,
    )
    out = ci.visibility_score(buildings, sight_lines=sight_lines, method="longest")
    assert (out["fac"] > 0).all()  # facade area from height
    assert out["3dvis"].between(0.0, 1.0).all()

    combined = ci.visibility_score(buildings, sight_lines=sight_lines, method="combined")
    assert "3dvis" in combined.columns


def test_visibility_score_rejects_bad_method():
    buildings = _buildings(2)
    buildings["height"] = [10.0, 20.0]
    sight_lines = gpd.GeoDataFrame(
        {"nodeID": [1, 2], "buildingID": [0, 1]},
        geometry=[LineString([(0, 0), (0, 5)]), LineString([(0, 0), (0, 9)])],
        crs=CRS,
    )
    with pytest.raises(ValueError, match="method must be"):
        ci.visibility_score(buildings, sight_lines=sight_lines, method="nonsense")


def test_cultural_score_counts_intersecting_historic_elements():
    buildings = _buildings(3)
    historic = gpd.GeoDataFrame(
        {"importance": [1.0, 2.0]},
        geometry=[Point(5, 5), Point(20, 5)],  # inside buildings 0 and 1
        crs=CRS,
    )
    out = ci.cultural_score(buildings, historic_elements_gdf=historic)
    by_id = out.set_index("buildingID")["cult"]
    assert by_id[0] == 1.0 and by_id[1] == 1.0 and by_id[2] == 0.0


def test_cultural_score_sums_score_column():
    buildings = _buildings(2)
    historic = gpd.GeoDataFrame(
        {"importance": [2.0, 3.0]},
        geometry=[Point(5, 5), Point(6, 6)],  # both inside building 0
        crs=CRS,
    )
    out = ci.cultural_score(buildings, historic_elements_gdf=historic, score_column="importance")
    assert out.set_index("buildingID")["cult"][0] == 5.0


def test_cultural_score_crs_mismatch_and_from_osm_errors():
    buildings = _buildings(2)
    other_crs = gpd.GeoDataFrame({"x": [1]}, geometry=[Point(5, 5)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="CRS mismatch"):
        ci.cultural_score(buildings, historic_elements_gdf=other_crs)
    with pytest.raises(ValueError, match="historic"):
        ci.cultural_score(buildings, from_OSM=True)


def test_pragmatic_score_measures_unexpectedness_between_neighbours():
    buildings = _buildings(3)
    buildings["land_uses"] = [["residential"], ["residential"], ["retail"]]
    buildings["land_uses_overlap"] = [[1.0], [1.0], [1.0]]
    out = ci.pragmatic_score(buildings, search_radius=100)
    assert "prag" in out.columns
    assert out["prag"].between(0.0, 1.0).all()
    # The lone 'retail' building is more unexpected than the common 'residential' ones.
    prag = out.set_index("buildingID")["prag"]
    assert prag[2] >= prag[0]


def test_pragmatic_score_defaults_missing_land_use_columns():
    buildings = _buildings(2)  # no land_uses / overlap columns
    out = ci.pragmatic_score(buildings, search_radius=50)
    assert "prag" in out.columns
    assert out["prag"].notna().all()


def test_compute_global_scores_produces_component_and_global_scores():
    scored = ci.compute_global_scores(_scored_buildings(4), GLOBAL_INDEXES, GLOBAL_COMPONENTS)
    for col in ["vScore", "sScore", "cScore", "pScore", "gScore", "gScore_sc"]:
        assert col in scored.columns
    assert scored["gScore_sc"].between(0.0, 1.0).all()


def test_compute_global_scores_rejects_unnormalised_component_weights():
    with pytest.raises(ValueError, match="sum to 1.0"):
        ci.compute_global_scores(_scored_buildings(3), GLOBAL_INDEXES, {"sScore": 0.9})


def test_compute_local_scores_produces_rescaled_local_score():
    local_indexes = {
        "3dvis": 0.5,
        "fac": 0.3,
        "height": 0.2,
        "area": 0.4,
        "2dvis": 0.0,
        "neigh": 0.3,
        "road": 0.3,
    }
    out = ci.compute_local_scores(
        _scored_buildings(4), local_indexes, LOCAL_COMPONENTS, rescaling_radius=1500
    )
    assert "lScore" in out.columns and "lScore_sc" in out.columns
    assert out["lScore_sc"].between(0.0, 1.0).all()


def test_compute_local_scores_rejects_unnormalised_component_weights():
    with pytest.raises(ValueError, match="sum to 1.0"):
        ci.compute_local_scores(_scored_buildings(3), {}, {"sScore": 0.5})


def test_visibility_score_cleans_messy_height_values():
    buildings = _buildings(3)
    # A comma-decimal string, a list-wrapped value, and a NaN exercise the _clean_height branches.
    buildings["height"] = ["12,5", [10.0], float("nan")]

    out = ci.visibility_score(buildings)  # no sight lines -> fac derived from cleaned heights

    fac = out["fac"].tolist()
    assert fac[0] > 0 and fac[1] > 0  # "12,5" -> 12.5 and [10.0] -> 10.0 give a facade area
    assert fac[2] == 0.0  # NaN height -> no facade area


def test_pragmatic_score_accepts_tuple_set_and_array_land_use_cells():
    import numpy as np

    buildings = _buildings(3)
    buildings["land_uses"] = [("residential",), {"residential"}, np.array(["retail"])]
    buildings["land_uses_overlap"] = [[1.0], [1.0], [1.0]]

    out = ci.pragmatic_score(buildings, search_radius=100)

    assert "prag" in out.columns
    assert out["prag"].between(0.0, 1.0).all()


def test_pragmatic_score_handles_mismatched_and_nonnumeric_overlaps():
    buildings = _buildings(3)
    buildings["land_uses"] = [["a", "b"], ["c"], ["d"]]
    # First row's overlap length != labels (falls back to equal split); second is non-numeric.
    buildings["land_uses_overlap"] = [[1.0], ["not_a_number"], [1.0]]

    out = ci.pragmatic_score(buildings, search_radius=100)
    assert out["prag"].notna().all()


def test_cultural_score_from_osm_uses_historic_flag():
    buildings = _buildings(3)
    buildings["historic"] = ["castle", None, "yes"]
    out = ci.cultural_score(buildings, from_OSM=True)
    cult = out.set_index("buildingID")["cult"]
    assert cult[0] == 1.0 and cult[2] == 1.0  # historic tags flagged
    assert cult[1] == 0.0  # no historic tag


def test_cultural_score_rejects_unknown_score_column():
    buildings = _buildings(2)
    historic = gpd.GeoDataFrame({"importance": [1.0]}, geometry=[Point(5, 5)], crs=CRS)
    with pytest.raises(ValueError, match="score_column"):
        ci.cultural_score(buildings, historic_elements_gdf=historic, score_column="missing")


def test_cultural_score_empty_historic_geometries_returns_zero():
    buildings = _buildings(2)
    historic = gpd.GeoDataFrame({"importance": [1.0]}, geometry=[None], crs=CRS)
    out = ci.cultural_score(buildings, historic_elements_gdf=historic)
    assert (out["cult"] == 0.0).all()  # nothing to intersect
