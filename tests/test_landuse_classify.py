"""Offline tests for OSM-triplet land-use classification (``cityImage.landuse.classify``).

Covers the "<token>:<group>:<domain>" triplet helpers and the DMA classifier.
All inputs are synthetic; no OSM download or optional dependency is required.
"""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point

import cityImage as ci
from cityImage import landuse

CRS = "EPSG:3857"
U = landuse.UNCLASSIFIED


def _gdf(rows, column="land_uses_raw"):
    return gpd.GeoDataFrame(
        {column: rows},
        geometry=[Point(i, 0) for i in range(len(rows))],
        crs=CRS,
    )


def test_raws_into_osmgroups_dedups_by_group_and_keeps_unclassified():
    gdf = _gdf(
        [
            [
                "shop:commercial:shop",
                "cafe:sustenance:amenity",
                "x:commercial:building",  # duplicate group -> dropped
                "malformed_no_colons",  # not a 3-part triplet -> ignored
                f"y:{U}:amenity",  # UNCLASSIFIED is retained
            ]
        ]
    )

    out = ci.classify_land_uses_raws_into_OSMgroups(gdf)

    assert out["land_uses"].iloc[0] == ["commercial", "sustenance", U]


def test_find_unclassified_tokens_modes_and_counts():
    gdf = _gdf(
        [
            [f"foo:{U}:amenity", "bar:commercial:shop"],
            [f"foo:{U}:amenity"],  # 'foo' appears twice overall
        ]
    )

    # token mode, unique + sorted
    assert ci.find_unclassified_tokens_OSM_groups(gdf, return_counts=False, mode="token") == ["foo"]
    # token mode with counts
    counts = ci.find_unclassified_tokens_OSM_groups(gdf, return_counts=True, mode="token")
    assert counts["foo"] == 2
    # token_domain and full triplet modes
    assert ci.find_unclassified_tokens_OSM_groups(
        gdf, return_counts=False, mode="token_domain"
    ) == ["foo:amenity"]
    assert ci.find_unclassified_tokens_OSM_groups(gdf, return_counts=False, mode="triplet") == [
        f"foo:{U}:amenity"
    ]


def test_find_unclassified_tokens_rejects_bad_mode():
    with pytest.raises(ValueError, match="mode must be one of"):
        ci.find_unclassified_tokens_OSM_groups(_gdf([[f"foo:{U}:amenity"]]), mode="nonsense")


def test_apply_manual_triplet_overrides_only_rewrites_unclassified():
    gdf = _gdf(
        [
            [
                f"foo:{U}:amenity",  # eligible -> overridden
                "bar:commercial:shop",  # already classified -> untouched
                f"baz:{U}:building",  # eligible but no override for 'baz' -> untouched
            ]
        ]
    )

    out = ci.apply_manual_triplet_overrides(gdf, overrides={"foo": "commercial"})

    assert out["land_uses_raw"].iloc[0] == [
        "foo:commercial:amenity",
        "bar:commercial:shop",
        f"baz:{U}:building",
    ]


def test_apply_manual_triplet_overrides_validates_inputs():
    gdf = _gdf([[f"foo:{U}:amenity"]])
    with pytest.raises(ValueError, match="non-empty dict"):
        ci.apply_manual_triplet_overrides(gdf, overrides={})
    with pytest.raises(ValueError, match="Invalid override"):
        ci.apply_manual_triplet_overrides(gdf, overrides={"foo": "not_a_real_group"})


def test_classify_land_uses_into_dmas_maps_functions_and_combinations():
    gdf = gpd.GeoDataFrame(
        {
            "land_uses": [
                ["residential"],  # live
                ["commercial"],  # work
                ["tourism"],  # visit
                ["residential", "commercial"],  # live_work
                ["shop_bakery"],  # shop_* -> visit
                [U],  # ignored -> other
                [],  # empty -> other
            ]
        },
        geometry=[Point(i, 0) for i in range(7)],
        crs=CRS,
    )

    out = ci.classify_land_uses_intoDMAs(gdf)

    assert out["DMA"].tolist() == [
        "live",
        "work",
        "visit",
        "live_work",
        "visit",
        "other",
        "other",
    ]


def test_classify_land_uses_into_dmas_requires_the_column():
    gdf = gpd.GeoDataFrame({"other": [[]]}, geometry=[Point(0, 0)], crs=CRS)
    with pytest.raises(ValueError, match="must contain"):
        ci.classify_land_uses_intoDMAs(gdf, land_uses_column="land_uses")
