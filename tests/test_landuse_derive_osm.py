"""Offline tests for ``derive_land_uses_raw_fromOSM`` (OSM tag -> triplet extraction)."""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Polygon

import cityImage as ci

CRS = "EPSG:3857"


def _square(i):
    return Polygon([(i, 0), (i + 1, 0), (i + 1, 1), (i, 1)])


def _osm_buildings():
    return gpd.GeoDataFrame(
        {
            "buildingID": [1, 2, 3, 4],
            "amenity": ["restaurant", None, None, None],
            "shop": [None, "bakery", None, None],
            "building:use:residential": [None, None, "yes", None],
        },
        geometry=[_square(i * 2) for i in range(4)],
        crs=CRS,
    )


def test_derive_land_uses_raw_from_osm_builds_triplets_and_defaults():
    out = ci.derive_land_uses_raw_fromOSM(_osm_buildings(), default="residential")

    raw = out.set_index("buildingID")["land_uses_raw"]
    assert all(isinstance(cell, list) for cell in raw)

    # tag-derived tokens are preserved in the "<token>:<group>:<domain>" triplets
    assert any("restaurant" in t for t in raw[1])
    assert any("bakery" in t for t in raw[2])
    # the building:use:* column feeds a 'building'-domain triplet
    assert any(t.endswith(":building") for t in raw[3])
    # a row with no usable tags falls back to the default land use
    assert raw[4] == ["residential:residential:building"]


def test_derive_then_classify_yields_macro_groups():
    derived = ci.derive_land_uses_raw_fromOSM(_osm_buildings(), default="residential")
    classified = ci.classify_land_uses_raws_into_OSMgroups(derived)

    groups = classified.set_index("buildingID")["land_uses"]
    assert groups[1] == ["shop_food_beverages"]  # restaurant -> food/beverages macro-group
    assert groups[4] == ["residential"]  # defaulted row


def test_derive_land_uses_raw_marks_unknown_tokens_unclassified():
    buildings = gpd.GeoDataFrame(
        {"buildingID": [1], "amenity": ["totally_unknown_xyz"]},
        geometry=[_square(0)],
        crs=CRS,
    )
    out = ci.derive_land_uses_raw_fromOSM(buildings, default="residential")
    # A token absent from the taxonomy keeps its provenance domain and is flagged UNCLASSIFIED.
    assert out["land_uses_raw"].iloc[0] == ["totally_unknown_xyz:UNCLASSIFIED:amenity"]
