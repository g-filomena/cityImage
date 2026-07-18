"""Offline tests for ``cityImage.osm`` using a mocked OSMnx.

No live Overpass access: ``cityImage.osm.ox`` is monkeypatched with a fake that
returns synthetic feature frames, so the download-dispatch, conversion, and
error-handling branches run deterministically.
"""

from __future__ import annotations

import types

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

import cityImage.osm as osm
from cityImage.osm import (
    _distance_arg,
    _empty_network,
    _empty_osm_features,
    _normalise_crs,
    _unit_overlaps_for_land_uses,
    _validate_download_method,
    barriers_from_osm,
    buildings_from_osm,
    features_from_osm,
    network_from_osm,
)

UTM = "EPSG:32633"  # metric CRS so areas are meaningful


class InsufficientResponseError(Exception):
    """Name-compatible stand-in for OSMnx's InsufficientResponseError.

    ``features_from_osm`` matches on ``exc.__class__.__name__``, so the class name
    must be exactly this (no leading underscore).
    """


# --------------------------------------------------------------------------- pure helpers


def test_pure_helpers():
    assert list(_empty_osm_features({"building": True}).columns) == ["building", "geometry"]
    nodes, edges = _empty_network("EPSG:3857")
    assert nodes.empty and edges.empty

    assert _distance_arg(500, "OSMplace") == 500
    with pytest.raises(ValueError, match="distance is required"):
        _distance_arg(None, "distance_from_address")

    assert _unit_overlaps_for_land_uses(["a", "b"]) == [0.5, 0.5]
    assert _unit_overlaps_for_land_uses(None) == [1.0]
    assert _unit_overlaps_for_land_uses("solo") == [1.0]

    assert _normalise_crs(3857) == "EPSG:3857"
    assert _normalise_crs("EPSG:4326") == "EPSG:4326"

    _validate_download_method("OSMplace")  # no raise
    with pytest.raises(ValueError, match="download_method must be one of"):
        _validate_download_method("bogus")


# --------------------------------------------------------------------------- fake OSMnx


def _building_features(tags=None, **_):
    return gpd.GeoDataFrame(
        {"building": ["yes"]},
        geometry=[Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])],  # 400 m^2
        crs=UTM,
    )


def _empty_features(tags=None, **_):
    cols = list(tags.keys()) if isinstance(tags, dict) else []
    return gpd.GeoDataFrame({c: [] for c in cols}, geometry=[], crs="EPSG:4326")


def _fake_ox(feature_fn):
    return types.SimpleNamespace(
        features_from_place=lambda q, tags=None: feature_fn(tags),
        features_from_address=lambda q, tags=None, dist=None: feature_fn(tags),
        features_from_point=lambda q, tags=None, dist=None: feature_fn(tags),
        features_from_polygon=lambda q, tags=None: feature_fn(tags),
    )


@pytest.mark.parametrize(
    "download_method,query,distance",
    [
        ("OSMplace", "Place", None),
        ("distance_from_address", "Addr", 500),
        ("distance_from_point", (0.0, 0.0), 500),
        ("polygon", Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), None),
    ],
)
def test_features_from_osm_dispatches_each_method(monkeypatch, download_method, query, distance):
    monkeypatch.setattr(osm, "ox", _fake_ox(_building_features))
    out = features_from_osm(
        query, {"building": True}, download_method=download_method, distance=distance, crs=UTM
    )
    assert not out.empty and out.crs.to_epsg() == 32633


def test_features_from_osm_handles_insufficient_response(monkeypatch):
    def _raise(*a, **k):
        raise InsufficientResponseError("no data")

    monkeypatch.setattr(osm, "ox", types.SimpleNamespace(features_from_place=_raise))
    out = features_from_osm("Nowhere", {"building": True}, crs="EPSG:3857")
    assert out.empty and out.crs.to_epsg() == 3857  # empty frame reprojected


def test_features_from_osm_reraises_other_errors(monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(osm, "ox", types.SimpleNamespace(features_from_place=_raise))
    with pytest.raises(RuntimeError, match="boom"):
        features_from_osm("X", {"building": True})


# --------------------------------------------------------------------------- higher-level bridges


def test_buildings_from_osm_builds_cityimage_schema(monkeypatch):
    monkeypatch.setattr(osm, "ox", _fake_ox(_building_features))
    buildings = buildings_from_osm("Place", crs=UTM, min_area=200)
    assert "buildingID" in buildings.columns
    assert "land_uses" in buildings.columns
    assert len(buildings) == 1


def test_barriers_from_osm_with_no_features_returns_empty(monkeypatch):
    monkeypatch.setattr(osm, "ox", _fake_ox(_empty_features))
    barriers = barriers_from_osm("Place", crs=UTM)
    assert barriers.empty


def test_network_from_osm_returns_empty_on_insufficient_response(monkeypatch):
    def _raise(*a, **k):
        raise InsufficientResponseError("no data")

    monkeypatch.setattr(osm, "ox", types.SimpleNamespace(graph_from_place=_raise))
    nodes, edges = network_from_osm("Nowhere", crs="EPSG:3857")
    assert nodes.empty and edges.empty


def test_network_from_osm_validates_arguments(monkeypatch):
    # graph_from_address must exist as an attribute so its call-arguments (and thus the
    # distance check) are evaluated; it is never actually reached.
    monkeypatch.setattr(osm, "ox", types.SimpleNamespace(graph_from_address=lambda *a, **k: None))
    with pytest.raises(ValueError, match="download_method must be one of"):
        network_from_osm("X", download_method="bogus")
    with pytest.raises(ValueError, match="distance is required"):
        network_from_osm("X", download_method="distance_from_address")


# --------------------------------------------------------------------------- graph conversion


def _fake_network_ox():
    """Fake OSMnx exposing a graph pipeline over a tiny 3-node / 2-edge network."""
    nodes = gpd.GeoDataFrame(
        {"x": [0.0, 10.0, 20.0], "y": [0.0, 0.0, 10.0]},
        geometry=[Point(0, 0), Point(10, 0), Point(20, 10)],
        index=[10, 20, 30],  # OSM node ids
        crs=UTM,
    )
    edge_idx = pd.MultiIndex.from_tuples([(10, 20, 0), (20, 30, 0)], names=["u", "v", "key"])
    edges = gpd.GeoDataFrame(
        {"length": [10.0, 14.14], "highway": ["residential", "primary"], "name": ["A", "B"]},
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(10, 0), (20, 10)])],
        index=edge_idx,
        crs=UTM,
    )
    graph = types.SimpleNamespace(graph={"crs": UTM})

    def graph_to_gdfs(g, nodes=False, edges=False, **_):
        return nodes_frame.copy() if nodes else edges_frame.copy()

    nodes_frame, edges_frame = nodes, edges

    def graph_from(*_a, **_k):
        return graph

    return types.SimpleNamespace(
        graph_from_place=graph_from,
        graph_from_address=graph_from,
        graph_from_point=graph_from,
        graph_from_polygon=graph_from,
        project_graph=lambda g, to_crs=None: g,
        graph_to_gdfs=graph_to_gdfs,
        projection=types.SimpleNamespace(project_gdf=lambda gdf, **_: gdf),
        features_from_place=lambda q, tags=None: _building_features(tags),
    )


def test_network_from_osm_converts_graph_to_cityimage_schema(monkeypatch):
    monkeypatch.setattr(osm, "ox", _fake_network_ox())
    nodes, edges = network_from_osm("Place", crs=UTM, dict_columns={"road_type": "highway"})

    assert {"nodeID", "x", "y", "geometry"}.issubset(nodes.columns)
    assert {"u", "v", "edgeID", "length", "road_type"}.issubset(edges.columns)
    node_ids = set(nodes["nodeID"])
    assert set(edges["u"]).issubset(node_ids) and set(edges["v"]).issubset(node_ids)
    assert edges["road_type"].tolist() == ["residential", "primary"]  # dict_columns mapping applied


@pytest.mark.parametrize(
    "download_method,query,distance",
    [
        ("distance_from_address", "Addr", 500),
        ("distance_from_point", (0.0, 0.0), 500),
        ("polygon", Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), None),
    ],
)
def test_network_from_osm_dispatches_each_graph_method(
    monkeypatch, download_method, query, distance
):
    monkeypatch.setattr(osm, "ox", _fake_network_ox())
    nodes, edges = network_from_osm(
        query, crs=UTM, download_method=download_method, distance=distance
    )
    assert not nodes.empty and not edges.empty


def test_network_from_osm_dict_columns_missing_column_raises(monkeypatch):
    monkeypatch.setattr(osm, "ox", _fake_network_ox())
    with pytest.raises(ValueError, match="missing column"):
        network_from_osm("Place", crs=UTM, dict_columns={"road_type": "does_not_exist"})


def test_buildings_from_osm_projects_when_crs_is_none(monkeypatch):
    monkeypatch.setattr(osm, "ox", _fake_network_ox())
    buildings = buildings_from_osm(
        "Place", crs=None, min_area=200
    )  # triggers projection.project_gdf
    assert "buildingID" in buildings.columns and len(buildings) == 1


def test_buildings_from_osm_drops_below_min_area(monkeypatch):
    def _two_buildings(tags=None, **_):
        return gpd.GeoDataFrame(
            {"building": ["yes", "yes"]},
            geometry=[
                Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]),  # 400 m^2 -> kept
                Polygon([(100, 100), (101, 100), (101, 101), (100, 101)]),  # 1 m^2 -> dropped
            ],
            crs=UTM,
        )

    monkeypatch.setattr(osm, "ox", _fake_ox(_two_buildings))
    buildings = buildings_from_osm("Place", crs=UTM, min_area=200)
    assert len(buildings) == 1  # the 1 m^2 footprint is filtered out
