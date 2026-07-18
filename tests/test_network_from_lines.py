"""Offline tests for ``network_from_lines`` edge cases and the OSMnx list-attr resolver."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, MultiLineString

import cityImage as ci
from cityImage.network import _resolve_list_edges_gdf

CRS = "EPSG:3857"


def test_resolve_list_edges_gdf_flattens_and_scores_list_attributes():
    edges = gpd.GeoDataFrame(
        {
            "highway": [["primary", "secondary"]],  # first element kept
            "name": [["Main St"]],
            "lanes": [["2", "3"]],  # max kept
            "bridge": [["yes"]],  # any truthy -> 1
            "tunnel": [np.nan],  # missing -> 0
        },
        geometry=[LineString([(0, 0), (1, 0)])],
        crs=CRS,
    )

    out = _resolve_list_edges_gdf(edges)

    assert out["highway"].iloc[0] == "primary"  # first of the list
    assert out["name"].iloc[0] == "Main St"
    assert out["lanes"].iloc[0] == "3"  # max of the list
    assert out["bridge"].iloc[0] == 1  # truthy bridge flagged
    assert out["tunnel"].iloc[0] == 0  # NaN -> 0


def _lines():
    return gpd.GeoDataFrame(
        {"road": ["a", "b"]},
        geometry=[LineString([(0, 0), (10, 0)]), LineString([(10, 0), (10, 10)])],
        crs=CRS,
    )


def test_network_from_lines_rejects_non_geodataframe():
    with pytest.raises(TypeError, match="must be a GeoDataFrame"):
        ci.network_from_lines([1, 2, 3], CRS)


def test_network_from_lines_empty_returns_empty_network():
    empty = gpd.GeoDataFrame({"road": []}, geometry=[], crs=CRS)
    nodes, edges = ci.network_from_lines(empty, CRS)
    assert nodes.empty and edges.empty
    assert "nodeID" in nodes.columns and "edgeID" in edges.columns


def test_network_from_lines_dict_and_other_column_errors():
    with pytest.raises(ValueError, match="missing column"):
        ci.network_from_lines(_lines(), CRS, dict_columns={"type": "nonexistent"})
    with pytest.raises(ValueError, match="missing columns"):
        ci.network_from_lines(_lines(), CRS, other_columns=["nonexistent"])


def test_network_from_lines_builds_nodes_and_edges_with_mapped_columns():
    nodes, edges = ci.network_from_lines(
        _lines(), CRS, dict_columns={"road_type": "road"}, other_columns=["road"]
    )
    assert len(nodes) == 3 and len(edges) == 2
    assert {"u", "v", "edgeID", "length", "road_type", "road"}.issubset(edges.columns)
    node_ids = set(nodes["nodeID"])
    assert set(edges["u"]).issubset(node_ids) and set(edges["v"]).issubset(node_ids)


def test_network_from_lines_merges_multilinestring_geometry():
    gdf = gpd.GeoDataFrame(
        {"road": ["a"]},
        geometry=[MultiLineString([[(0, 0), (1, 0)], [(1, 0), (2, 0)]])],
        crs=CRS,
    )
    nodes, edges = ci.network_from_lines(gdf, CRS)
    assert (edges.geometry.geom_type == "LineString").all()  # multipart merged/exploded
