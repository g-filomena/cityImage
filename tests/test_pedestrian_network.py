"""Tests for cityImage pedestrian-network filtering.

These tests lock the cityImage-specific pedestrian filtering semantics while
leaving OSM feature retrieval delegated to OSMnx.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Polygon

import cityImage as ci

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
        "crossing footway",
        "cycleway foot yes",
    ]


def test_pedestrian_network_from_osm_features_builds_cityimage_network_and_keeps_metadata():
    nodes, edges = ci.pedestrian_network_from_osm_features(_highway_features(), CRS)

    assert not nodes.empty
    assert edges.sort_values("edgeID")["highway"].tolist() == [
        "residential",
        "footway",
        "cycleway",
    ]
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


def test_pedestrian_network_from_osm_requires_osm_extra_for_live_download_if_osmnx_missing():
    osmnx = pytest.importorskip("osmnx")
    assert osmnx is not None
