"""Minimal deterministic GeoDataFrames for behaviour-lock tests.

The fixtures intentionally avoid live OSM, raster, 3D mesh, and plotting dependencies.
They encode only the semantic outputs we want to preserve while refactoring.
"""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString, Point, Polygon

CRS = "EPSG:3857"


def minimal_buildings() -> gpd.GeoDataFrame:
    """Return three simple building footprints with stable IDs and attributes."""
    buildings = gpd.GeoDataFrame(
        {
            "buildingID": [1, 2, 3],
            "height": [10.0, 5.0, 15.0],
        },
        geometry=[
            Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]),
            Polygon([(3, 0), (5, 0), (5, 2), (3, 2)]),
            Polygon([(0, 3), (2, 3), (2, 5), (0, 5)]),
        ],
        crs=CRS,
    )
    buildings["area"] = buildings.geometry.area.astype(float)
    return buildings


def minimal_buildings_with_land_use() -> gpd.GeoDataFrame:
    """Return building fixtures with normalised land-use list columns."""
    buildings = minimal_buildings()
    buildings["land_uses"] = [["retail"], ["retail"], ["education"]]
    buildings["land_uses_overlap"] = [[1.0], [1.0], [1.0]]
    return buildings


def sparse_land_use_polygons() -> gpd.GeoDataFrame:
    """Return polygon land-use evidence that partially overlaps buildings."""
    return gpd.GeoDataFrame(
        {"raw_use": ["shop", "school", "shop"]},
        geometry=[
            Polygon([(0, 0), (1, 0), (1, 2), (0, 2)]),
            Polygon([(1, 0), (2, 0), (2, 2), (1, 2)]),
            Polygon([(3, 0), (5, 0), (5, 2), (3, 2)]),
        ],
        crs=CRS,
    )


def raw_sparse_buildings() -> gpd.GeoDataFrame:
    """Return rows used to test sparse non-OSM land-use classification."""
    return gpd.GeoDataFrame(
        {"raw": [["shop", "school", "shop"], None, "clinic"]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs=CRS,
    )


def minimal_network() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return a four-node square network with four equal edges."""
    nodes = gpd.GeoDataFrame(
        {
            "nodeID": [1, 2, 3, 4],
            "x": [0.0, 10.0, 10.0, 0.0],
            "y": [0.0, 0.0, 10.0, 10.0],
        },
        geometry=[Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)],
        crs=CRS,
    )
    edges = gpd.GeoDataFrame(
        {
            "edgeID": [101, 102, 103, 104],
            "u": [1, 2, 3, 4],
            "v": [2, 3, 4, 1],
            "length": [10.0, 10.0, 10.0, 10.0],
        },
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(10, 0), (10, 10)]),
            LineString([(10, 10), (0, 10)]),
            LineString([(0, 10), (0, 0)]),
        ],
        crs=CRS,
    )
    return nodes, edges


def path_network_with_opportunities() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Return a 3-node path network for centrality/reach behaviour locks."""
    nodes = gpd.GeoDataFrame(
        {
            "nodeID": [1, 2, 3],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "opportunity": [10.0, 20.0, 30.0],
        },
        geometry=[Point(0, 0), Point(1, 0), Point(2, 0)],
        crs=CRS,
    )
    edges = gpd.GeoDataFrame(
        {
            "edgeID": [201, 202],
            "u": [1, 2],
            "v": [2, 3],
            "length": [1.0, 1.0],
            "weight": [1.0, 1.0],
        },
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])],
        crs=CRS,
    )
    return nodes, edges


def structuring_barriers() -> gpd.GeoDataFrame:
    """Return two crossing line barriers with explicit barrier IDs."""
    return gpd.GeoDataFrame(
        {"barrierID": [10, 20], "barrier_type": ["water", "railway"]},
        geometry=[LineString([(5, -1), (5, 11)]), LineString([(-1, 5), (11, 5)])],
        crs=CRS,
    )


def sight_lines() -> gpd.GeoDataFrame:
    """Return sight lines for two of the three buildings."""
    return gpd.GeoDataFrame(
        {"nodeID": [1, 1, 2], "buildingID": [1, 1, 3]},
        geometry=[
            LineString([(0, 0), (0, 2)]),
            LineString([(0, 0), (0, 4)]),
            LineString([(0, 0), (0, 8)]),
        ],
        crs=CRS,
    )


def historic_elements() -> gpd.GeoDataFrame:
    """Return one cultural/historic point intersecting the first building."""
    return gpd.GeoDataFrame(
        {"importance": [2.0]},
        geometry=[Point(1, 1)],
        crs=CRS,
    )
