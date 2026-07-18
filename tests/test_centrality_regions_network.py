"""Offline tests for centrality helpers, polygon->node districting, and network builders."""

from __future__ import annotations

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, Polygon

import cityImage as ci
from cityImage.centrality import append_edges_metrics, weight_nodes
from cityImage.network import join_nodes_edges_by_coordinates, obtain_nodes_gdf

CRS = "EPSG:3857"


def _path_graph():
    graph = nx.Graph()
    graph.add_node(1, x=0.0, y=0.0)
    graph.add_node(2, x=1.0, y=0.0)
    graph.add_node(3, x=2.0, y=0.0)
    graph.add_edge(1, 2, weight=1.0, edgeID=10)
    graph.add_edge(2, 3, weight=1.0, edgeID=11)
    return graph


# --------------------------------------------------------------------------- centrality


def test_calculate_centrality_igraph_betweenness_and_closeness():
    pytest.importorskip("igraph")
    graph = _path_graph()

    betweenness = ci.calculate_centrality(graph, measure="betweenness")  # default igraph backend
    closeness = ci.calculate_centrality(graph, measure="closeness")

    assert betweenness[2] > betweenness[1]  # the middle node is most between
    assert closeness[2] > closeness[1]


def test_calculate_centrality_reach_requires_radius_and_attribute():
    pytest.importorskip("igraph")
    with pytest.raises(ValueError, match="radius and attribute"):
        ci.calculate_centrality(_path_graph(), measure="reach")


def test_calculate_centrality_rejects_unknown_measure():
    pytest.importorskip("igraph")
    with pytest.raises(ValueError, match="Unsupported centrality type"):
        ci.calculate_centrality(_path_graph(), measure="nonsense")


def test_weight_nodes_counts_services_within_radius():
    nodes = gpd.GeoDataFrame(
        {"nodeID": [1, 2]}, geometry=[Point(0, 0), Point(500, 0)], crs=CRS
    ).set_index("nodeID", drop=False)
    services = gpd.GeoDataFrame(
        {"s": [1, 2, 3]},
        geometry=[Point(1, 1), Point(2, 2), Point(499, 0)],
        crs=CRS,
    )
    graph = nx.Graph()
    graph.add_node(1)
    graph.add_node(2)

    out_graph = weight_nodes(nodes, services, graph, "opp", radius=50)

    assert nodes.at[1, "opp"] == 2 and nodes.at[2, "opp"] == 1
    assert out_graph.nodes[1]["opp"] == 2


def test_append_edges_metrics_merges_values_and_fills_missing():
    graph = _path_graph()
    # append_edges_metrics looks rows up by edgeID via .at, so index by edgeID
    # (with an unnamed index, as the real callers pass it).
    edges_gdf = pd.DataFrame({"edgeID": [10, 11, 99]}).set_index(
        "edgeID", drop=False
    )  # 99 unmatched
    edges_gdf.index.name = None
    metric = {(1, 2): 0.5, (2, 3): 0.7}

    out = append_edges_metrics(edges_gdf, graph, [metric], ["betw"])

    assert out.loc[10, "betw"] == 0.5
    assert out.loc[11, "betw"] == 0.7
    assert out.loc[99, "betw"] == 0.0  # missing edge filled with 0.0


# --------------------------------------------------------------------------- regions


def test_district_to_nodes_from_polygons_assigns_nearest_polygon():
    nodes = gpd.GeoDataFrame(
        {"nodeID": [1, 2, 3]},
        geometry=[Point(1, 1), Point(21, 1), Point(9, 9)],
        crs=CRS,
    )
    partitions = gpd.GeoDataFrame(
        {"p_topo": [100, 200]},
        geometry=[
            Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),  # district 100
            Polygon([(20, 0), (30, 0), (30, 10), (20, 10)]),  # district 200
        ],
        crs=CRS,
    )

    out = ci.district_to_nodes_from_polygons(nodes, partitions, "p_topo")

    by_id = out.set_index("nodeID")["p_topo"]
    assert by_id[1] == 100 and by_id[3] == 100  # inside / nearest to district 100
    assert by_id[2] == 200
    assert out["p_topo"].dtype.kind == "i"  # cast to int


# --------------------------------------------------------------------------- network builders


def _edges():
    return gpd.GeoDataFrame(
        {"edgeID": [1, 2]},
        geometry=[
            LineString([(0, 0), (10, 0)]),
            LineString([(10, 0), (10, 10)]),
        ],
        crs=CRS,
    )


def test_obtain_nodes_gdf_extracts_unique_endpoints():
    nodes = obtain_nodes_gdf(_edges(), crs=CRS)
    assert len(nodes) == 3  # (0,0), (10,0), (10,10)
    assert {"nodeID", "x", "y"}.issubset(nodes.columns)


def test_obtain_nodes_gdf_empty_edges_returns_empty_frame():
    empty = gpd.GeoDataFrame({"edgeID": []}, geometry=[], crs=CRS)
    nodes = obtain_nodes_gdf(empty, crs=CRS)
    assert nodes.empty and "nodeID" in nodes.columns


def test_join_nodes_edges_by_coordinates_and_reset_index_roundtrip():
    edges = _edges()
    nodes = obtain_nodes_gdf(edges, crs=CRS)

    nodes, edges = join_nodes_edges_by_coordinates(nodes, edges)
    assert edges["u"].notna().all() and edges["v"].notna().all()

    nodes2, edges2 = ci.reset_index_graph_gdfs(nodes, edges)
    # every edge endpoint still resolves to a surviving node
    node_ids = set(nodes2["nodeID"])
    assert set(edges2["u"]).issubset(node_ids)
    assert set(edges2["v"]).issubset(node_ids)
