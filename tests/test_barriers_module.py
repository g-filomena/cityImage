"""Tests for hard barrier delegation.

These tests lock cityImage-specific barrier semantics while keeping OSM feature
retrieval outside the core library.
"""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString, Polygon

import cityImage as ci
from tests.fixtures.cityimage_minimal import york_network

CRS = "EPSG:3857"


def test_road_barriers_from_osm_features_filters_major_roads_and_tunnels():
    roads = gpd.GeoDataFrame(
        {
            "highway": ["motorway", "primary", "secondary", "residential", "trunk"],
            "tunnel": [0, 0, 0, 0, "yes"],
        },
        geometry=[
            LineString([(0, 0), (1, 0)]),
            LineString([(0, 1), (1, 1)]),
            LineString([(0, 2), (1, 2)]),
            LineString([(0, 3), (1, 3)]),
            LineString([(0, 4), (1, 4)]),
        ],
        crs=CRS,
    )

    barriers = ci.road_barriers_from_osm_features(
        roads,
        crs=CRS,
        include_primary=True,
        include_secondary=False,
    )

    assert barriers["barrier_type"].tolist() == ["road", "road"]
    assert len(barriers) == 2


def test_water_barriers_from_osm_features_combines_rivers_and_large_lakes():
    waterways = gpd.GeoDataFrame(
        {"waterway": ["river", "stream", "canal"]},
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(0, 1), (1, 1)]),
            LineString([(0, 2), (10, 2)]),
        ],
        crs=CRS,
    )
    water = gpd.GeoDataFrame(
        {"water": ["lake", "river"]},
        geometry=[
            Polygon([(0, 0), (100, 0), (100, 100), (0, 100)]),
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        ],
        crs=CRS,
    )

    barriers = ci.water_barriers_from_osm_features(
        waterways_gdf=waterways,
        water_gdf=water,
        crs=CRS,
        lakes_area=1000,
        min_lake_boundary_length=1,
    )

    assert not barriers.empty
    assert set(barriers["barrier_type"]) == {"water"}


def test_barriers_from_osm_features_assigns_barrier_ids_and_types():
    roads = gpd.GeoDataFrame(
        {"highway": ["motorway"]},
        geometry=[LineString([(0, 0), (1, 0)])],
        crs=CRS,
    )
    railways = gpd.GeoDataFrame(
        {"railway": ["rail"]},
        geometry=[LineString([(0, 1), (1, 1)])],
        crs=CRS,
    )
    parks = gpd.GeoDataFrame(
        {"leisure": ["park"]},
        geometry=[Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])],
        crs=CRS,
    )

    barriers = ci.barriers_from_osm_features(
        roads_gdf=roads,
        railways_gdf=railways,
        parks_gdf=parks,
        crs=CRS,
        parks_min_area=100,
    )

    assert barriers["barrierID"].tolist() == list(range(len(barriers)))
    assert {"road", "railway", "park"}.issubset(set(barriers["barrier_type"]))


def test_existing_barrier_assignment_semantics_still_work():
    edges = gpd.GeoDataFrame(
        {"edgeID": [1], "u": [1], "v": [2]},
        geometry=[LineString([(0, 0), (10, 0)])],
        crs=CRS,
    )
    barriers = gpd.GeoDataFrame(
        {"barrierID": [10], "barrier_type": ["water"]},
        geometry=[LineString([(5, -5), (5, 5)])],
        crs=CRS,
    )

    assigned = ci.assign_structuring_barriers(edges, barriers)

    assert assigned["sep_barr"].tolist() == [True]


def test_along_water_indexes_edges_by_edgeid_with_non_contiguous_ids():
    # Regression: barriers_along looks edges up by edgeID via .loc, so on the real York network -
    # whose cleaned edgeIDs are non-contiguous while the index is a plain RangeIndex - it must not
    # raise KeyError.
    _, edges_gdf = york_network()
    assert edges_gdf["edgeID"].tolist() != list(range(len(edges_gdf)))  # genuinely non-contiguous

    minx, miny, maxx, maxy = edges_gdf.total_bounds
    mid_y = (miny + maxy) / 2
    barriers = gpd.GeoDataFrame(
        {"barrierID": [10], "barrier_type": ["water"]},
        geometry=[LineString([(minx - 10, mid_y), (maxx + 10, mid_y)])],  # river crossing the area
        crs=edges_gdf.crs,
    )

    out = ci.along_water(edges_gdf.copy(), barriers)

    assert "a_rivers" in out.columns
    assert len(out) == len(edges_gdf)
