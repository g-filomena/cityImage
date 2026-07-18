"""Offline tests for the OSM-download dispatch in ``cityImage.pedestrian``.

These cover the argument-validation branches and the OSMnx modern/legacy shim
without performing any live download.
"""

from __future__ import annotations

import types

import pytest

import cityImage as ci
from cityImage.pedestrian import _call_osmnx_features

CRS = "EPSG:3857"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"download_method": "OSMplace", "query": None},
        {"download_method": "distance_from_address", "address": None, "query": None},
        {"download_method": "distance_from_point", "point": None, "query": None},
        {"download_method": "polygon", "polygon": None, "query": None},
    ],
)
def test_pedestrian_network_from_osm_requires_a_query_per_method(kwargs):
    with pytest.raises(ValueError, match="must be provided"):
        ci.pedestrian_network_from_osm(crs=CRS, **kwargs)


def test_pedestrian_network_from_osm_rejects_unknown_download_method():
    with pytest.raises(ValueError, match="download_method must be one of"):
        ci.pedestrian_network_from_osm("Somewhere", crs=CRS, download_method="bogus")


def test_call_osmnx_features_prefers_modern_api():
    fake_ox = types.SimpleNamespace(features_from_place=lambda *a, **k: "modern")
    assert _call_osmnx_features(fake_ox, "place", "X", tags={"highway": True}) == "modern"


def test_call_osmnx_features_falls_back_to_legacy_api():
    fake_ox = types.SimpleNamespace(geometries_from_place=lambda *a, **k: "legacy")
    assert _call_osmnx_features(fake_ox, "place", "X") == "legacy"


def test_call_osmnx_features_raises_without_supported_api():
    fake_ox = types.SimpleNamespace()
    with pytest.raises(ImportError, match="features_from_|geometries_from_"):
        _call_osmnx_features(fake_ox, "place", "X")
