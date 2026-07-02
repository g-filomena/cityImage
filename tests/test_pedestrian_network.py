"""Tests for cityImage pedestrian-network filtering.

These tests lock the cityImage-specific pedestrian filtering semantics while
leaving OSM feature retrieval delegated to OSMnx.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Polygon

import cityImage as ci
from cityImage.pedestrian import _is_pedestrian_row, _pedestrian_status

CRS = "EPSG:3857"


def _highway_features() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "highway": [
                "residential",
                "primary",
                "footway",
                "footway",
                "cycleway",
                "cycleway",
                "service",
                "path",
            ],
            "name": [
                "residential sidewalk exception",
                "primary foot no",
                "sidewalk footway",
                "crossing footway",
                "cycleway no foot",
                "cycleway foot yes",
                "private service",
                "area path",
            ],
            "foot": [None, "no", None, None, None, "yes", None, None],
            "footway": [None, None, "sidewalk", "crossing", None, None, None, None],
            "sidewalk": ["no", None, None, None, None, None, None, None],
            "access": [None, None, None, None, None, None, "private", None],
            "area": [None, None, None, None, None, None, None, "yes"],
            "lit": ["yes", "yes", "yes", "no", "no", "yes", "yes", "no"],
        },
        geometry=[
            LineString([(0, 0), (1, 0)]),
            LineString([(0, 1), (1, 1)]),
            LineString([(0, 2), (1, 2)]),
            LineString([(0, 3), (1, 3)]),
            LineString([(0, 4), (1, 4)]),
            LineString([(0, 5), (1, 5)]),
            LineString([(0, 6), (1, 6)]),
            LineString([(0, 7), (1, 7)]),
        ],
        crs=CRS,
    )


def test_filter_pedestrian_osm_features_preserves_cityimage_semantics():
    filtered = ci.filter_pedestrian_osm_features(_highway_features())

    assert filtered["name"].tolist() == [
        "residential sidewalk exception",
        "sidewalk footway",
        "crossing footway",
        "cycleway foot yes",
    ]


def test_pedestrian_network_from_osm_features_builds_cityimage_network_and_keeps_metadata():
    nodes, edges = ci.pedestrian_network_from_osm_features(_highway_features(), CRS)

    assert not nodes.empty
    ordered = edges.sort_values("edgeID")
    # Two distinct footways survive (footway=sidewalk is kept under the default
    # keep_both policy); the name column disambiguates them.
    assert ordered["name"].tolist() == [
        "residential sidewalk exception",
        "sidewalk footway",
        "crossing footway",
        "cycleway foot yes",
    ]
    assert ordered["highway"].tolist() == ["residential", "footway", "footway", "cycleway"]
    assert {"edgeID", "u", "v", "length", "name", "highway", "lit", "foot", "footway"}.issubset(
        edges.columns
    )
    assert "sidewalk" in edges.columns


def test_pedestrian_network_from_osm_features_returns_empty_network_for_no_line_matches():
    features = gpd.GeoDataFrame(
        {"highway": ["primary"], "foot": ["no"]},
        geometry=[LineString([(0, 0), (1, 0)])],
        crs=CRS,
    )

    nodes, edges = ci.pedestrian_network_from_osm_features(features, CRS)

    assert nodes.empty
    assert edges.empty
    assert {"nodeID", "x", "y"}.issubset(nodes.columns)
    assert {"edgeID", "u", "v", "length"}.issubset(edges.columns)


def test_filter_pedestrian_osm_features_drops_non_line_geometries():
    features = gpd.GeoDataFrame(
        {"highway": ["path"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs=CRS,
    )

    assert ci.filter_pedestrian_osm_features(features).empty


@pytest.mark.parametrize(
    "sidewalk_tags, expected",
    [
        ({"sidewalk": "yes"}, True),
        ({"sidewalk": "1"}, True),  # sloppy affirmative spellings
        ({"sidewalk": "y"}, True),
        ({"sidewalk": "true"}, True),
        ({"sidewalk": "left"}, True),  # positional
        ({"sidewalk": "separate"}, True),  # mapped as its own way
        ({"sidewalk:left": "1"}, True),  # affirmative on a sub-key
        ({"sidewalk": "no"}, False),
        ({"sidewalk": None}, False),
    ],
)
def test_ambiguous_highway_kept_only_with_sidewalk_evidence(sidewalk_tags, expected):
    # "service" is evidence-gated: kept only with a usable foot tag or a sidewalk.
    row = pd.Series({"highway": "service", **sidewalk_tags})
    assert _is_pedestrian_row(row) is expected


@pytest.mark.parametrize("highway", ["track", "bridleway"])
def test_track_and_bridleway_are_always_footable(highway):
    assert _pedestrian_status(pd.Series({"highway": highway})) == "yes"


def test_cycleway_kept_when_a_sidewalk_is_mapped():
    assert _pedestrian_status(pd.Series({"highway": "cycleway", "sidewalk": "yes"})) == "yes"
    assert _pedestrian_status(pd.Series({"highway": "cycleway", "foot": "permissive"})) == "yes"
    assert _pedestrian_status(pd.Series({"highway": "cycleway"})) is None


def test_footway_traffic_island_is_kept():
    row = pd.Series({"highway": "footway", "footway": "traffic_island"})
    assert _pedestrian_status(row) == "yes"


@pytest.mark.parametrize("highway", ["residential", "primary", "secondary", "tertiary"])
def test_flagged_highways_kept_and_flagged_by_evidence(highway):
    # No evidence: kept, but flagged as uncertain.
    assert _pedestrian_status(pd.Series({"highway": highway})) == "noEvidence"
    # Evidence present: flagged as walkable.
    assert _pedestrian_status(pd.Series({"highway": highway, "sidewalk": "both"})) == "yes"
    # foot=no still drops it entirely.
    assert _pedestrian_status(pd.Series({"highway": highway, "foot": "no"})) is None


def test_living_street_is_always_walkable():
    assert _pedestrian_status(pd.Series({"highway": "living_street"})) == "yes"


def test_explicit_foot_access_overrides_access_restriction():
    # access=private is overridden by a permissive mode-specific foot tag.
    assert (
        _pedestrian_status(
            pd.Series({"highway": "service", "access": "private", "foot": "designated"})
        )
        == "yes"
    )
    assert (
        _pedestrian_status(pd.Series({"highway": "footway", "access": "private", "foot": "yes"}))
        == "yes"
    )
    # ...but without an explicit foot tag, the access restriction still drops it.
    assert _pedestrian_status(pd.Series({"highway": "service", "access": "private"})) is None
    # ...and foot=no wins even when access would otherwise allow it.
    assert (
        _pedestrian_status(pd.Series({"highway": "footway", "access": "yes", "foot": "no"})) is None
    )


@pytest.mark.parametrize("foot_value", ["no", "use_sidepath", "discouraged", "private"])
def test_negative_foot_values_drop_the_way(foot_value):
    # Even an inherently pedestrian way is dropped when foot says "not here".
    assert _pedestrian_status(pd.Series({"highway": "footway", "foot": foot_value})) is None


def test_filter_adds_ped_column_with_expected_values():
    features = gpd.GeoDataFrame(
        {
            "highway": ["primary", "footway", "primary"],
            "sidewalk": [None, None, "both"],
        },
        geometry=[
            LineString([(0, 0), (1, 0)]),
            LineString([(0, 1), (1, 1)]),
            LineString([(0, 2), (1, 2)]),
        ],
        crs=CRS,
    )
    filtered = ci.filter_pedestrian_osm_features(features)
    assert "ped" in filtered.columns
    assert filtered["ped"].tolist() == ["noEvidence", "yes", "yes"]


def test_pedestrian_network_edges_carry_ped_column():
    _, edges = ci.pedestrian_network_from_osm_features(_highway_features(), CRS)
    assert "ped" in edges.columns
    assert set(edges["ped"].dropna().unique()).issubset({"yes", "noEvidence"})


def test_pedestrian_network_drops_all_null_metadata_columns():
    # Minimal input: no foot/footway/cycleway/access tags at all.
    features = gpd.GeoDataFrame(
        {"highway": ["residential", "footway"]},
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(0, 1), (1, 1)])],
        crs=CRS,
    )
    _, edges = ci.pedestrian_network_from_osm_features(features, CRS)
    # ped and highway are always populated, so they survive.
    assert {"ped", "highway"}.issubset(edges.columns)
    # Entirely-empty passthrough tag columns are not carried.
    for empty_col in ["foot", "footway", "cycleway", "access"]:
        assert empty_col not in edges.columns


def _sidewalk(highway, footway=None, sidewalk=None):
    tags = {"highway": highway}
    if footway is not None:
        tags["footway"] = footway
    if sidewalk is not None:
        tags["sidewalk"] = sidewalk
    return pd.Series(tags)


def test_keep_both_policy_keeps_sidewalk_and_centreline():
    # Default: separately-mapped sidewalk kept, and a road with sidewalk=separate kept.
    assert _pedestrian_status(_sidewalk("footway", footway="sidewalk")) == "yes"
    assert _pedestrian_status(_sidewalk("residential", sidewalk="separate")) == "yes"


def test_centrelines_policy_drops_separately_mapped_sidewalks():
    # The parallel sidewalk way is dropped; the street centreline is kept.
    assert _pedestrian_status(_sidewalk("footway", footway="sidewalk"), "centrelines") is None
    assert _pedestrian_status(_sidewalk("residential", sidewalk="separate"), "centrelines") == "yes"
    # A genuine standalone footway (not a sidewalk) is unaffected.
    assert _pedestrian_status(_sidewalk("footway", footway="crossing"), "centrelines") == "yes"


def test_sidewalks_policy_drops_centreline_when_sidewalk_is_separate():
    # The separately-mapped sidewalk is kept; the redundant centreline is dropped.
    assert _pedestrian_status(_sidewalk("footway", footway="sidewalk"), "sidewalks") == "yes"
    assert _pedestrian_status(_sidewalk("residential", sidewalk="separate"), "sidewalks") is None
    # A road whose sidewalk is attached (not separate) keeps its centreline.
    assert _pedestrian_status(_sidewalk("residential", sidewalk="both"), "sidewalks") == "yes"


def test_invalid_sidewalk_policy_raises():
    features = gpd.GeoDataFrame(
        {"highway": ["footway"]},
        geometry=[LineString([(0, 0), (1, 0)])],
        crs=CRS,
    )
    with pytest.raises(ValueError, match="sidewalk_policy"):
        ci.filter_pedestrian_osm_features(features, sidewalk_policy="nope")


def test_pedestrian_network_from_osm_is_available_with_core_install():
    # osmnx is a core dependency, so the live-download helper is always importable.
    import osmnx

    assert osmnx is not None
    assert callable(ci.pedestrian_network_from_osm)
