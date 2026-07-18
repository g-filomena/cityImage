"""Offline tests for barrier assignment and OSM-feature builder options.

Complements ``test_barriers_module.py`` by covering the park-assignment path,
``barriers_along`` directly, and the less-common builder branches.
"""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString, Polygon

import cityImage as ci

CRS = "EPSG:3857"


def _park_barriers():
    parks = gpd.GeoDataFrame(
        {"leisure": ["park"]},
        geometry=[Polygon([(0, 0), (400, 0), (400, 400), (0, 400)])],
        crs=CRS,
    )
    return ci.barriers_from_osm_features(parks_gdf=parks, crs=CRS)  # adds barrierID


def test_along_within_parks_flags_edges_inside_a_park():
    barriers = _park_barriers()
    edges = gpd.GeoDataFrame(
        {"edgeID": [1, 2]},
        geometry=[
            LineString([(50, 50), (350, 350)]),  # crosses the park
            LineString([(1000, 1000), (1010, 1010)]),  # far outside
        ],
        crs=CRS,
    )

    out = ci.along_within_parks(edges.copy(), barriers)

    by_id = out.set_index("edgeID")["w_parks"]
    assert len(by_id[1]) > 0  # inside the park -> at least one park barrier
    assert by_id[2] == []  # outside -> none


def test_along_within_parks_without_park_barriers_yields_empty_lists():
    barriers = gpd.GeoDataFrame(
        {"barrierID": [1], "barrier_type": ["water"]},
        geometry=[LineString([(0, 0), (10, 0)])],
        crs=CRS,
    )
    edges = gpd.GeoDataFrame({"edgeID": [1]}, geometry=[LineString([(0, 0), (5, 5)])], crs=CRS)

    out = ci.along_within_parks(edges.copy(), barriers)
    assert out["w_parks"].tolist() == [[]]


def test_barriers_along_returns_unblocked_nearby_barrier():
    barriers = gpd.GeoDataFrame(
        {"barrierID": [7], "barrier_type": ["water"]},
        geometry=[LineString([(0, 100), (300, 100)])],  # parallel, ~100 away
        crs=CRS,
    )
    edges = gpd.GeoDataFrame(
        {"edgeID": [10]},
        geometry=[LineString([(0, 0), (300, 0)])],
        crs=CRS,
    ).set_index("edgeID", drop=False)
    sindex = edges.sindex

    assert ci.barriers_along(10, edges, barriers, sindex, offset=200) == [7]


def test_barriers_along_empty_barriers_returns_empty():
    edges = gpd.GeoDataFrame(
        {"edgeID": [10]}, geometry=[LineString([(0, 0), (10, 0)])], crs=CRS
    ).set_index("edgeID", drop=False)
    empty = gpd.GeoDataFrame({"barrierID": [], "barrier_type": []}, geometry=[], crs=CRS)
    assert ci.barriers_along(10, edges, empty, edges.sindex, offset=100) == []


def test_road_barriers_include_secondary_and_drop_tunnels():
    roads = gpd.GeoDataFrame(
        {
            "highway": ["motorway", "secondary", "residential", "trunk"],
            "tunnel": [0, 0, 0, "yes"],
        },
        geometry=[LineString([(0, i), (10, i)]) for i in range(4)],
        crs=CRS,
    )

    barriers = ci.road_barriers_from_osm_features(roads, crs=CRS, include_secondary=True)

    # motorway + secondary kept; residential excluded; tunnelled trunk dropped.
    assert len(barriers) == 2
    assert set(barriers["barrier_type"]) == {"road"}


def test_railway_barriers_keep_light_rail_and_empty_when_none_match():
    railways = gpd.GeoDataFrame(
        {"railway": ["rail", "light_rail"]},
        geometry=[LineString([(0, 0), (100, 0)]), LineString([(0, 50), (100, 50)])],
        crs=CRS,
    )
    kept = ci.railway_barriers_from_osm_features(railways, crs=CRS, keep_light_rail=True)
    assert not kept.empty and set(kept["barrier_type"]) == {"railway"}

    none = gpd.GeoDataFrame(
        {"railway": ["subway"]}, geometry=[LineString([(0, 0), (1, 0)])], crs=CRS
    )
    assert ci.railway_barriers_from_osm_features(none, crs=CRS).empty


def test_assign_structuring_barriers_excludes_parks_and_secondary_roads():
    edges = gpd.GeoDataFrame({"edgeID": [1]}, geometry=[LineString([(0, 0), (10, 0)])], crs=CRS)
    barriers = gpd.GeoDataFrame(
        {"barrierID": [1, 2], "barrier_type": ["park", "secondary_road"]},
        geometry=[LineString([(5, -5), (5, 5)]), LineString([(7, -5), (7, 5)])],
        crs=CRS,
    )

    assigned = ci.assign_structuring_barriers(edges, barriers)
    # Both crossing barriers are of excluded types -> no structuring separation.
    assert assigned["sep_barr"].tolist() == [False]
