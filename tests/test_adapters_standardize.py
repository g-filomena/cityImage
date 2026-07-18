"""Offline tests for the schema-standardisation adapters (``cityImage.adapters``)."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

import cityImage as ci

CRS = "EPSG:3857"


def test_standardize_nodes_from_id_column_and_point_coords():
    nodes = gpd.GeoDataFrame({"osmid": [10, 20]}, geometry=[Point(0, 0), Point(1, 2)], crs=CRS)
    out = ci.standardize_nodes_gdf(nodes, node_id_column="osmid")
    assert out["nodeID"].tolist() == [10, 20]  # id taken from the named column
    assert out["x"].tolist() == [0.0, 1.0]  # x/y derived from Point geometry
    assert out["y"].tolist() == [0.0, 2.0]


def test_standardize_edges_from_osmnx_multiindex():
    idx = pd.MultiIndex.from_tuples([(1, 2, 0), (2, 3, 0)], names=["u", "v", "key"])
    edges = gpd.GeoDataFrame(
        {"length": [1.0, 1.0]},
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])],
        index=idx,
        crs=CRS,
    )
    out = ci.standardize_edges_gdf(edges)
    assert out["u"].tolist() == [1, 2] and out["v"].tolist() == [2, 3]  # pulled from the MultiIndex
    assert "edgeID" in out.columns


def test_standardize_edges_missing_v_raises():
    edges = gpd.GeoDataFrame(
        {"u": [1, 2], "length": [1.0, 1.0]},  # 'v' absent, plain index -> unresolvable
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])],
        crs=CRS,
    )
    with pytest.raises(ValueError, match="u/v"):
        ci.standardize_edges_gdf(edges)


def _one_building(**cols):
    return gpd.GeoDataFrame(
        {"buildingID": [1], **cols},
        geometry=[Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])],
        crs=CRS,
    )


def test_standardize_buildings_maps_raw_land_use_to_list_column():
    out = ci.standardize_buildings_gdf(
        _one_building(land_use=["church"]), land_uses_raw_column="land_use"
    )
    assert "land_uses_raw" in out.columns
    assert isinstance(out["land_uses_raw"].iloc[0], list)
    assert "area" in out.columns and out["area"].iloc[0] == 4.0


def test_standardize_buildings_overlap_without_land_uses_raises():
    with pytest.raises(ValueError, match="land_uses_overlap requires land_uses"):
        ci.standardize_buildings_gdf(_one_building(land_uses_overlap=[[1.0]]))


def test_standardize_cityimage_inputs_returns_only_provided():
    nodes = gpd.GeoDataFrame({"nodeID": [1]}, geometry=[Point(0, 0)], crs=CRS)
    edges = gpd.GeoDataFrame(
        {"u": [1], "v": [1], "length": [1.0]},
        geometry=[LineString([(0, 0), (1, 0)])],
        crs=CRS,
    )
    out = ci.standardize_cityimage_inputs(nodes_gdf=nodes, edges_gdf=edges)
    assert set(out.keys()) == {"nodes_gdf", "edges_gdf"}
    assert "nodeID" in out["nodes_gdf"].columns
    assert {"u", "v", "edgeID"}.issubset(out["edges_gdf"].columns)


def test_standardize_nodes_rejects_non_geodataframe():
    with pytest.raises(TypeError, match="must be a GeoDataFrame"):
        ci.standardize_nodes_gdf(pd.DataFrame({"nodeID": [1]}))


def test_standardize_edges_from_named_edge_id_column():
    edges = gpd.GeoDataFrame(
        {"myid": [7, 8], "u": [1, 2], "v": [2, 3], "length": [1.0, 1.0]},
        geometry=[LineString([(0, 0), (1, 0)]), LineString([(1, 0), (2, 0)])],
        crs=CRS,
    )
    out = ci.standardize_edges_gdf(edges, edge_id_column="myid")
    assert out["edgeID"].tolist() == [7, 8]


def test_standardize_buildings_reuses_existing_raw_land_use_column():
    out = ci.standardize_buildings_gdf(_one_building(land_uses_raw="church"))
    assert isinstance(out["land_uses_raw"].iloc[0], list)  # coerced to a list cell
