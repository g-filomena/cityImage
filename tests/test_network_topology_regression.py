"""Regression tests for cityImage-owned network topology semantics."""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import LineString, Point

import cityImage.network_topology as nt
from tests.fixtures.cityimage_minimal import york_raw_network


def _nodes(rows, crs="EPSG:3857"):
    gdf = gpd.GeoDataFrame(
        rows,
        geometry=[Point(row["x"], row["y"]) for row in rows],
        crs=crs,
    )
    return gdf.set_index("nodeID", drop=False)


def _edges(rows, crs="EPSG:3857"):
    gdf = gpd.GeoDataFrame(rows, geometry=[row["geometry"] for row in rows], crs=crs)
    gdf["length"] = gdf.geometry.length
    return gdf.set_index("edgeID", drop=False)


def _as_nodes_edges(result, original_nodes):
    """Normalise topology functions that historically returned either edges or nodes/edges."""
    if isinstance(result, tuple):
        return result

    return original_nodes, result


def test_fix_network_topology_splits_only_existing_internal_vertices():
    nodes_gdf = _nodes(
        [
            {"nodeID": 1, "x": 0.0, "y": 0.0},
            {"nodeID": 2, "x": 10.0, "y": 0.0},
            {"nodeID": 3, "x": 5.0, "y": 0.0},
            {"nodeID": 4, "x": 5.0, "y": 5.0},
        ]
    )
    edges_gdf = _edges(
        [
            {
                "edgeID": 10,
                "u": 1,
                "v": 2,
                "geometry": LineString([(0, 0), (5, 0), (10, 0)]),
            },
            {
                "edgeID": 20,
                "u": 3,
                "v": 4,
                "geometry": LineString([(5, 0), (5, 5)]),
            },
        ]
    )

    fixed_nodes, fixed_edges = _as_nodes_edges(
        nt.fix_network_topology(nodes_gdf.copy(), edges_gdf.copy()),
        nodes_gdf,
    )

    assert len(fixed_edges) == 3
    assert sorted(round(length, 6) for length in fixed_edges.geometry.length) == [5.0, 5.0, 5.0]

    line_coords = {tuple(geom.coords) for geom in fixed_edges.geometry}
    assert ((0.0, 0.0), (10.0, 0.0)) not in line_coords
    assert any((5.0, 0.0) in tuple(geom.coords) for geom in fixed_edges.geometry)

    used_nodes = set(fixed_edges["u"]).union(fixed_edges["v"])
    node_lookup = fixed_nodes.set_index("nodeID")
    used_coords = {
        (float(node_lookup.loc[node_id, "x"]), float(node_lookup.loc[node_id, "y"]))
        for node_id in used_nodes
    }
    assert (5.0, 0.0) in used_coords


def test_fix_network_topology_ignores_endpoint_intersections():
    nodes_gdf = _nodes(
        [
            {"nodeID": 1, "x": 0.0, "y": 0.0},
            {"nodeID": 2, "x": 10.0, "y": 0.0},
            {"nodeID": 3, "x": 10.0, "y": 10.0},
        ]
    )
    edges_gdf = _edges(
        [
            {"edgeID": 10, "u": 1, "v": 2, "geometry": LineString([(0, 0), (10, 0)])},
            {"edgeID": 20, "u": 2, "v": 3, "geometry": LineString([(10, 0), (10, 10)])},
        ]
    )

    _, fixed_edges = _as_nodes_edges(
        nt.fix_network_topology(nodes_gdf.copy(), edges_gdf.copy()),
        nodes_gdf,
    )

    assert len(fixed_edges) == 2
    assert sorted(fixed_edges["edgeID"].tolist()) == [10, 20]


def test_fix_network_topology_ignores_crossings_that_are_not_existing_vertices():
    nodes_gdf = _nodes(
        [
            {"nodeID": 1, "x": 0.0, "y": 0.0},
            {"nodeID": 2, "x": 10.0, "y": 0.0},
            {"nodeID": 3, "x": 5.0, "y": -5.0},
            {"nodeID": 4, "x": 5.0, "y": 5.0},
        ]
    )
    edges_gdf = _edges(
        [
            {"edgeID": 10, "u": 1, "v": 2, "geometry": LineString([(0, 0), (10, 0)])},
            {"edgeID": 20, "u": 3, "v": 4, "geometry": LineString([(5, -5), (5, 5)])},
        ]
    )

    _, fixed_edges = _as_nodes_edges(
        nt.fix_network_topology(nodes_gdf.copy(), edges_gdf.copy()),
        nodes_gdf,
    )

    assert len(fixed_edges) == 2
    assert {tuple(geom.coords) for geom in fixed_edges.geometry} == {
        ((0.0, 0.0), (10.0, 0.0)),
        ((5.0, -5.0), (5.0, 5.0)),
    }


def test_simplify_graph_removes_degree_two_nodes_and_preserves_requested_nodes():
    nodes_gdf = _nodes(
        [
            {"nodeID": 1, "x": 0.0, "y": 0.0},
            {"nodeID": 2, "x": 5.0, "y": 0.0},
            {"nodeID": 3, "x": 10.0, "y": 0.0},
        ]
    )
    edges_gdf = _edges(
        [
            {"edgeID": 10, "u": 1, "v": 2, "geometry": LineString([(0, 0), (5, 0)])},
            {"edgeID": 11, "u": 2, "v": 3, "geometry": LineString([(5, 0), (10, 0)])},
        ]
    )

    simplified_nodes, simplified_edges = nt.simplify_graph(nodes_gdf.copy(), edges_gdf.copy())

    assert 2 not in simplified_nodes["nodeID"].tolist()
    assert len(simplified_edges) == 1
    assert sorted(simplified_edges.iloc[0][["u", "v"]].tolist()) == [1, 3]
    assert list(simplified_edges.iloc[0].geometry.coords) == [(0.0, 0.0), (5.0, 0.0), (10.0, 0.0)]

    kept_nodes, kept_edges = nt.simplify_graph(
        nodes_gdf.copy(),
        edges_gdf.copy(),
        nodes_to_keep_regardless=[2],
    )

    assert 2 in kept_nodes["nodeID"].tolist()
    assert len(kept_edges) == 2


def test_correct_edge_geometries_forces_linestring_endpoints_to_node_coordinates():
    nodes_gdf = _nodes(
        [
            {"nodeID": 1, "x": 0.0, "y": 0.0},
            {"nodeID": 2, "x": 10.0, "y": 0.0},
        ]
    )
    edges_gdf = _edges(
        [
            {
                "edgeID": 10,
                "u": 1,
                "v": 2,
                "geometry": LineString([(0.25, 0.25), (5, 1), (9.75, -0.25)]),
            },
        ]
    )

    corrected = nt.correct_edge_geometries(nodes_gdf.copy(), edges_gdf.copy())

    assert list(corrected.iloc[0].geometry.coords) == [
        (0.0, 0.0),
        (5.0, 1.0),
        (10.0, 0.0),
    ]


def test_clean_same_vertexes_edges_collapses_similar_duplicate_edges_to_center_line():
    nodes_gdf = _nodes(
        [
            {"nodeID": 1, "x": 0.0, "y": 0.0},
            {"nodeID": 2, "x": 10.0, "y": 0.0},
        ]
    )
    edges_gdf = _edges(
        [
            {"edgeID": 10, "u": 1, "v": 2, "geometry": LineString([(0, 0), (10, 0)])},
            {"edgeID": 11, "u": 1, "v": 2, "geometry": LineString([(0, 2), (10, 2)])},
        ]
    )

    clean_nodes, clean_edges = nt.clean_same_vertexes_edges(nodes_gdf.copy(), edges_gdf.copy())

    assert clean_nodes["nodeID"].tolist() == [1, 2]
    assert len(clean_edges) == 1
    assert list(clean_edges.iloc[0].geometry.coords) == [(0.0, 1.0), (10.0, 1.0)]


def test_clean_same_vertexes_edges_keeps_longer_duplicate_when_lengths_differ():
    nodes_gdf = _nodes(
        [
            {"nodeID": 1, "x": 0.0, "y": 0.0},
            {"nodeID": 2, "x": 10.0, "y": 0.0},
        ]
    )
    edges_gdf = _edges(
        [
            {"edgeID": 10, "u": 1, "v": 2, "geometry": LineString([(0, 0), (10, 0)])},
            {
                "edgeID": 11,
                "u": 1,
                "v": 2,
                "geometry": LineString([(0, 0), (5, 10), (10, 0)]),
            },
        ]
    )

    _, clean_edges = nt.clean_same_vertexes_edges(nodes_gdf.copy(), edges_gdf.copy())

    assert len(clean_edges) == 1
    assert clean_edges.iloc[0]["edgeID"] == 11
    assert list(clean_edges.iloc[0].geometry.coords) == [(0.0, 0.0), (5.0, 10.0), (10.0, 0.0)]


def test_consolidate_nodes_merges_close_nodes_and_returns_edges_when_requested():
    nodes = _nodes(
        [
            {"nodeID": 1, "x": 0.0, "y": 0.0},
            {"nodeID": 2, "x": 100.0, "y": 0.0},
            {"nodeID": 3, "x": 101.0, "y": 0.0},  # within tolerance of node 2 -> merged
            {"nodeID": 4, "x": 200.0, "y": 0.0},
        ]
    )
    edges = _edges(
        [
            {"edgeID": 1, "u": 1, "v": 2, "geometry": LineString([(0, 0), (100, 0)])},
            {"edgeID": 2, "u": 2, "v": 3, "geometry": LineString([(100, 0), (101, 0)])},
            {"edgeID": 3, "u": 3, "v": 4, "geometry": LineString([(101, 0), (200, 0)])},
        ]
    )

    cons_nodes, cons_edges = nt.consolidate_nodes(
        nodes, edges, consolidate_edges_too=True, tolerance=5
    )

    assert len(cons_nodes) < len(nodes)  # nodes 2 and 3 were merged
    assert (cons_edges["u"] != cons_edges["v"]).all()  # the 2-3 edge collapsed and was dropped


def test_clean_network_full_pass_yields_consistent_topology():
    # Run the full clean_network pass over a central subset of the real York street network. It must
    # dedupe, drop dead ends/islands, and leave a valid topology: every edge endpoint resolves to a
    # surviving node and no self-loops remain.
    nodes_gdf, edges_gdf = york_raw_network()

    clean_nodes, clean_edges = nt.clean_network(
        nodes_gdf,
        edges_gdf,
        dead_ends=True,
        remove_islands=True,
        same_vertexes_edges=True,
        self_loops=True,
        fix_topology=True,
    )

    node_ids = set(clean_nodes["nodeID"])
    assert len(clean_nodes) > 0 and len(clean_edges) > 0
    assert set(clean_edges["u"]).issubset(node_ids)
    assert set(clean_edges["v"]).issubset(node_ids)
    assert (clean_edges["u"] != clean_edges["v"]).all()  # no self-loops
