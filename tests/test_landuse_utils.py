"""Offline tests for the pure token/normalisation helpers in ``cityImage.landuse.utils``."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

import cityImage as ci
from cityImage.landuse.utils import (
    _clean_tokens,
    _is_canonical_token,
    _is_missing_scalar,
    _is_truthy_osm_tag_value,
    _normalize_token,
    _to_list,
)


def test_is_missing_scalar_distinguishes_scalars_from_containers():
    assert _is_missing_scalar(None) is True
    assert _is_missing_scalar(pd.NA) is True
    assert _is_missing_scalar(float("nan")) is True
    assert _is_missing_scalar([1]) is False  # list-like is never "missing"
    assert _is_missing_scalar("x") is False


def test_is_truthy_osm_tag_value():
    assert _is_truthy_osm_tag_value("yes") is True
    assert _is_truthy_osm_tag_value("no") is False
    assert _is_truthy_osm_tag_value("0") is False
    assert _is_truthy_osm_tag_value(None) is False


def test_is_canonical_token_rules():
    assert _is_canonical_token("fire_station") is True
    assert _is_canonical_token("") is False
    assert _is_canonical_token("Fire") is False  # uppercase
    assert _is_canonical_token("a b") is False  # whitespace
    assert _is_canonical_token("a-b") is False  # hyphen
    assert _is_canonical_token("_x") is False  # leading underscore
    assert _is_canonical_token("a__b") is False  # doubled underscore
    assert _is_canonical_token("a!b") is False  # bad char


def test_to_list_normalises_every_supported_type():
    assert _to_list(None) == []
    assert _to_list("x") == ["x"]
    assert _to_list([1, 2]) == [1, 2]
    assert _to_list((1, 2)) == [1, 2]
    assert _to_list({1}) == [1]
    assert _to_list(np.array([1, 2])) == [1, 2]
    assert _to_list(pd.Series([1, 2])) == [1, 2]
    assert _to_list(float("nan")) == []
    assert _to_list(5) == [5]


def test_normalize_token_cleans_noise():
    assert _normalize_token("Fi:re-station ") == "fire_station"
    assert _normalize_token(None) is None
    assert _normalize_token("   ") is None


def test_clean_tokens_splits_dedups_and_drops_junk():
    assert _clean_tokens(None) == []
    assert _clean_tokens(["shop", "yes", "shop"]) == ["shop"]  # drop 'yes', dedup
    assert _clean_tokens("a;b;yes") == ["a", "b"]  # split ';', drop 'yes'
    assert _clean_tokens([None, "", "retail"]) == ["retail"]  # skip None/empty
    assert _clean_tokens(5) == ["5"]  # non-string scalar coerced


def test_find_land_use_values_matching_counts_and_unique():
    buildings = gpd.GeoDataFrame(
        {"land_uses": [["shop_bakery", "residential"], ["shop_bakery"], ["office"]]},
        geometry=[Point(i, 0) for i in range(3)],
        crs="EPSG:3857",
    )

    counts = ci.find_land_use_values_matching(buildings, pattern="^shop", return_counts=True)
    assert counts["shop_bakery"] == 2

    unique = ci.find_land_use_values_matching(buildings, pattern="^shop", return_counts=False)
    assert unique == ["shop_bakery"]
