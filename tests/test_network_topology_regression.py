"""Regression tests for cityImage-owned network topology semantics."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pytest
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


def test_clean_same_vertexes_edges_keeps_one_edge_of_a_street_mapped_twice():
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

    # Two equally central copies: the shorter, then the lower edgeID, is kept as mapped.
    assert clean_nodes["nodeID"].tolist() == [1, 2]
    assert clean_edges["edgeID"].tolist() == [10]
    assert list(clean_edges.iloc[0].geometry.coords) == [(0.0, 0.0), (10.0, 0.0)]


def _crescent():
    # A straight street and a crescent joining the same two junctions, each with a stub beyond.
    nodes_gdf = _nodes(
        [
            {"nodeID": 1, "x": 0.0, "y": 0.0},
            {"nodeID": 2, "x": 10.0, "y": 0.0},
            {"nodeID": 3, "x": -10.0, "y": 0.0},
            {"nodeID": 4, "x": 20.0, "y": 0.0},
            {"nodeID": 5, "x": 0.0, "y": -10.0},
            {"nodeID": 6, "x": 10.0, "y": -10.0},
        ]
    )
    edges_gdf = _edges(
        [
            {"edgeID": 10, "u": 1, "v": 2, "geometry": LineString([(0, 0), (10, 0)])},
            {"edgeID": 11, "u": 2, "v": 1, "geometry": LineString([(10, 0), (5, 10), (0, 0)])},
            {"edgeID": 12, "u": 3, "v": 1, "geometry": LineString([(-10, 0), (0, 0)])},
            {"edgeID": 13, "u": 2, "v": 4, "geometry": LineString([(10, 0), (20, 0)])},
            {"edgeID": 14, "u": 1, "v": 5, "geometry": LineString([(0, 0), (0, -10)])},
            {"edgeID": 15, "u": 2, "v": 6, "geometry": LineString([(10, 0), (10, -10)])},
        ]
    )
    return nodes_gdf, edges_gdf


def test_clean_same_vertexes_edges_keeps_both_streets_as_parallel_edges():
    nodes_gdf, edges_gdf = _crescent()

    clean_nodes, clean_edges = nt.clean_same_vertexes_edges(nodes_gdf.copy(), edges_gdf.copy())

    # The straight street and the crescent both join nodes 1 and 2, unchanged; no node is added.
    assert sorted(clean_edges["edgeID"].tolist()) == [10, 11, 12, 13, 14, 15]
    assert sorted(clean_nodes["nodeID"].tolist()) == [1, 2, 3, 4, 5, 6]
    for edge_id in (10, 11):
        assert clean_edges.loc[edge_id].geometry.equals(edges_gdf.loc[edge_id].geometry)


def test_clean_network_keeps_a_crescent_and_terminates():
    nodes_gdf, edges_gdf = _crescent()

    _, clean_edges = nt.clean_network(
        nodes_gdf, edges_gdf, remove_islands=False, same_vertexes_edges=True
    )

    assert round(clean_edges.geometry.length.sum(), 6) == round(edges_gdf.geometry.length.sum(), 6)
    pairs = [frozenset(pair) for pair in zip(clean_edges["u"], clean_edges["v"], strict=False)]
    assert pairs.count(frozenset((1, 2))) == 2  # the straight street and the crescent


def test_fix_dead_ends_removes_a_dead_end_street_back_to_its_junction():
    # A square block with a two-segment street hanging off one corner.
    nodes_gdf = _nodes(
        [
            {"nodeID": 1, "x": 0.0, "y": 0.0},
            {"nodeID": 2, "x": 10.0, "y": 0.0},
            {"nodeID": 3, "x": 10.0, "y": 10.0},
            {"nodeID": 4, "x": 0.0, "y": 10.0},
            {"nodeID": 5, "x": 20.0, "y": 0.0},
            {"nodeID": 6, "x": 30.0, "y": 0.0},
        ]
    )
    edges_gdf = _edges(
        [
            {"edgeID": 10, "u": 1, "v": 2, "geometry": LineString([(0, 0), (10, 0)])},
            {"edgeID": 11, "u": 2, "v": 3, "geometry": LineString([(10, 0), (10, 10)])},
            {"edgeID": 12, "u": 3, "v": 4, "geometry": LineString([(10, 10), (0, 10)])},
            {"edgeID": 13, "u": 4, "v": 1, "geometry": LineString([(0, 10), (0, 0)])},
            {"edgeID": 14, "u": 2, "v": 5, "geometry": LineString([(10, 0), (20, 0)])},
            {"edgeID": 15, "u": 5, "v": 6, "geometry": LineString([(20, 0), (30, 0)])},
        ]
    )

    clean_nodes, clean_edges = nt.fix_dead_ends(nodes_gdf.copy(), edges_gdf.copy())

    assert sorted(clean_edges["edgeID"].tolist()) == [10, 11, 12, 13]
    assert sorted(clean_nodes["nodeID"].tolist()) == [1, 2, 3, 4]


def test_simplify_graph_gives_a_merged_edge_the_attribute_of_most_of_its_length():
    nodes_gdf = _nodes(
        [
            {"nodeID": 1, "x": 0.0, "y": 0.0},
            {"nodeID": 2, "x": 10.0, "y": 0.0},
            {"nodeID": 3, "x": 25.0, "y": 0.0},
            {"nodeID": 4, "x": 40.0, "y": 0.0},
        ]
    )
    edges_gdf = _edges(
        [
            {
                "edgeID": 10,
                "u": 1,
                "v": 2,
                "geometry": LineString([(0, 0), (10, 0)]),
                "highway": "primary",
                "name": None,
            },
            {
                "edgeID": 11,
                "u": 2,
                "v": 3,
                "geometry": LineString([(10, 0), (25, 0)]),
                "highway": "secondary",
                "name": "Via Roma",
            },
            {
                "edgeID": 12,
                "u": 3,
                "v": 4,
                "geometry": LineString([(25, 0), (40, 0)]),
                "highway": "secondary",
                "name": None,
            },
        ]
    )

    _, simplified = nt.simplify_graph(nodes_gdf.copy(), edges_gdf.copy())

    assert len(simplified) == 1
    assert simplified.iloc[0]["highway"] == "secondary"  # 30 m against 10 m of primary
    assert simplified.iloc[0]["name"] == "Via Roma"  # nulls do not outvote a value


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


def _line_network(n, spacing):
    """n nodes along the x axis, `spacing` metres apart, each joined to the next."""
    nodes = _nodes([{"nodeID": i, "x": i * spacing, "y": 0.0} for i in range(n)])
    edges = _edges(
        [
            {
                "edgeID": i,
                "u": i,
                "v": i + 1,
                "geometry": LineString([(i * spacing, 0), ((i + 1) * spacing, 0)]),
            }
            for i in range(n - 1)
        ]
    )
    return nodes, edges


def _max_cluster_span(cons_nodes, nodes):
    """Largest distance between two original nodes merged into the same consolidated node."""
    span = 0.0
    for old_ids in cons_nodes["old_nodeID"]:
        points = list(nodes.loc[old_ids].geometry)
        for a in points:
            for b in points:
                span = max(span, a.distance(b))
    return span


def test_consolidate_nodes_does_not_chain_along_closely_spaced_nodes():
    # 21 nodes 10 m apart span 200 m. With a 15 m tolerance only neighbours may merge; the
    # whole line must not collapse into one node through a chain of short gaps.
    nodes, edges = _line_network(21, 10.0)

    cons_nodes, cons_edges = nt.consolidate_nodes(
        nodes, edges, consolidate_edges_too=True, tolerance=15
    )

    assert _max_cluster_span(cons_nodes, nodes) <= 15
    assert len(cons_nodes) >= 11
    assert set(cons_edges["u"]) | set(cons_edges["v"]) <= set(cons_nodes["nodeID"])


def test_consolidate_nodes_tolerance_is_a_distance_not_a_buffer_radius():
    # Two nodes 20 m apart are farther than a 15 m tolerance and must stay separate.
    nodes, edges = _line_network(2, 20.0)

    cons_nodes = nt.consolidate_nodes(nodes, edges, tolerance=15)

    assert len(cons_nodes) == 2


def test_consolidate_nodes_every_cluster_fits_within_tolerance():
    # A dense 12 x 12 grid at 7 m spacing: every merged cluster must have all its members within
    # the tolerance of one another, whatever the density.
    spacing, size = 7.0, 12
    rows = [
        {"nodeID": i * size + j, "x": i * spacing, "y": j * spacing}
        for i in range(size)
        for j in range(size)
    ]
    edge_rows = [
        {
            "edgeID": i * size + j,
            "u": i * size + j,
            "v": i * size + j + 1,
            "geometry": LineString([(i * spacing, j * spacing), (i * spacing, (j + 1) * spacing)]),
        }
        for i in range(size)
        for j in range(size - 1)
    ]
    nodes, edges = _nodes(rows), _edges(edge_rows)

    cons_nodes = nt.consolidate_nodes(nodes, edges, tolerance=10)

    assert _max_cluster_span(cons_nodes, nodes) <= 10
    assert 1 < len(cons_nodes) < len(nodes)


def test_consolidate_nodes_with_z_column_keeps_2d_edges():
    # Nodes with a 'z' column consolidate to 3D points; edges drawn in 2D must stay 2D.
    nodes, edges = _line_network(6, 20.0)
    nodes["z"] = 10.0
    nodes.loc[2, "geometry"] = Point(22.0, 0.0)  # 2 m from node 1, so the two merge
    cons_nodes, cons_edges = nt.consolidate_nodes(
        nodes, edges, consolidate_edges_too=True, tolerance=5
    )

    assert len(cons_nodes) == 5
    assert not cons_edges.geometry.has_z.any()


def test_consolidate_nodes_does_not_depend_on_row_order():
    nodes, edges = _line_network(21, 10.0)

    first = nt.consolidate_nodes(nodes, edges, tolerance=15)
    second = nt.consolidate_nodes(nodes.sample(frac=1, random_state=1), edges, tolerance=15)

    def groups(cons):
        return sorted(sorted(ids) for ids in cons["old_nodeID"])

    assert groups(first) == groups(second)


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


def test_clean_network_keeps_a_street_joined_only_at_an_unnoded_crossing():
    # Two ways that cross at a shared internal vertex, as OSM ways do mid-way: neither ends there,
    # so the crossing is not a node until topology is fixed. The horizontal street is extended so
    # it forms the larger component; island removal must not see the vertical one as an island.
    from cityImage.network import network_from_lines

    lines = gpd.GeoDataFrame(
        geometry=[
            LineString([(0, 0), (50, 0), (100, 0)]),
            LineString([(100, 0), (200, 0)]),
            LineString([(50, -50), (50, 0), (50, 50)]),
        ],
        crs="EPSG:3857",
    )
    nodes, edges = network_from_lines(lines, "EPSG:3857")

    clean_nodes, clean_edges = nt.clean_network(
        nodes, edges, remove_islands=True, fix_topology=True, same_vertexes_edges=False
    )

    assert abs(clean_edges.geometry.length.sum() - 300.0) < 1e-6


def _star():
    # A loop-free network: three streets meeting at node 2.
    nodes_gdf = _nodes(
        [
            {"nodeID": 1, "x": 0.0, "y": 0.0},
            {"nodeID": 2, "x": 10.0, "y": 0.0},
            {"nodeID": 3, "x": 20.0, "y": 0.0},
            {"nodeID": 4, "x": 10.0, "y": 10.0},
        ]
    )
    edges_gdf = _edges(
        [
            {"edgeID": 10, "u": 1, "v": 2, "geometry": LineString([(0, 0), (10, 0)])},
            {"edgeID": 11, "u": 2, "v": 3, "geometry": LineString([(10, 0), (20, 0)])},
            {"edgeID": 12, "u": 2, "v": 4, "geometry": LineString([(10, 0), (10, 10)])},
        ]
    )
    return nodes_gdf, edges_gdf


def test_fix_dead_ends_leaves_a_loop_free_network_as_it_is():
    nodes_gdf, edges_gdf = _star()

    clean_nodes, clean_edges = nt.fix_dead_ends(nodes_gdf.copy(), edges_gdf.copy())

    assert sorted(clean_edges["edgeID"].tolist()) == [10, 11, 12]
    assert sorted(clean_nodes["nodeID"].tolist()) == [1, 2, 3, 4]


def test_clean_network_with_dead_ends_does_not_empty_a_loop_free_network():
    for remove_islands in (True, False):
        nodes_gdf, edges_gdf = _star()

        clean_nodes, clean_edges = nt.clean_network(
            nodes_gdf, edges_gdf, dead_ends=True, remove_islands=remove_islands
        )

        assert len(clean_edges) == 3
        assert len(clean_nodes) == 4


def _block_with_branch():
    # A square block with a two-segment street, 2-5-6, hanging off corner 2.
    nodes_gdf = _nodes(
        [
            {"nodeID": 1, "x": 0.0, "y": 0.0},
            {"nodeID": 2, "x": 10.0, "y": 0.0},
            {"nodeID": 3, "x": 10.0, "y": 10.0},
            {"nodeID": 4, "x": 0.0, "y": 10.0},
            {"nodeID": 5, "x": 20.0, "y": 0.0},
            {"nodeID": 6, "x": 30.0, "y": 0.0},
        ]
    )
    edges_gdf = _edges(
        [
            {"edgeID": 10, "u": 1, "v": 2, "geometry": LineString([(0, 0), (10, 0)])},
            {"edgeID": 11, "u": 2, "v": 3, "geometry": LineString([(10, 0), (10, 10)])},
            {"edgeID": 12, "u": 3, "v": 4, "geometry": LineString([(10, 10), (0, 10)])},
            {"edgeID": 13, "u": 4, "v": 1, "geometry": LineString([(0, 10), (0, 0)])},
            {"edgeID": 14, "u": 2, "v": 5, "geometry": LineString([(10, 0), (20, 0)])},
            {"edgeID": 15, "u": 5, "v": 6, "geometry": LineString([(20, 0), (30, 0)])},
        ]
    )
    return nodes_gdf, edges_gdf


def test_fix_dead_ends_stops_at_a_node_to_keep():
    nodes_gdf, edges_gdf = _block_with_branch()

    clean_nodes, clean_edges = nt.fix_dead_ends(
        nodes_gdf.copy(), edges_gdf.copy(), nodes_to_keep_regardless=[5]
    )

    # The segment beyond the kept node goes; the kept node and its way back to the block stay.
    assert sorted(clean_edges["edgeID"].tolist()) == [10, 11, 12, 13, 14]
    assert sorted(clean_nodes["nodeID"].tolist()) == [1, 2, 3, 4, 5]


def test_clean_network_keeps_a_protected_node_on_a_dead_end_street():
    nodes_gdf, edges_gdf = _block_with_branch()

    clean_nodes, _ = nt.clean_network(
        nodes_gdf, edges_gdf, dead_ends=True, nodes_to_keep_regardless=[5]
    )

    assert 5 in clean_nodes["nodeID"].tolist()


def _pair_with_detour(detour_coords, detour_u=1, detour_v=2):
    # A straight 10 m street from node 1 to node 2, and a second street joining the same pair.
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
                "u": detour_u,
                "v": detour_v,
                "geometry": LineString(detour_coords),
            },
        ]
    )
    return nodes_gdf, edges_gdf


def test_clean_same_vertexes_edges_keeps_a_parallel_edge_as_mapped():
    # Edge 11 runs from node 1 to node 2, its geometry drawn from node 2 to node 1.
    nodes_gdf, edges_gdf = _pair_with_detour([(10, 0), (5, 10), (0, 0)])

    _, clean_edges = nt.clean_same_vertexes_edges(
        nodes_gdf.copy(), edges_gdf.copy(), preserve_direction=True
    )

    assert sorted(clean_edges["edgeID"].tolist()) == [10, 11]
    parallel = clean_edges.loc[11]
    assert (parallel["u"], parallel["v"]) == (1, 2)
    assert parallel.geometry.equals(edges_gdf.loc[11].geometry)


def test_clean_same_vertexes_edges_keeps_an_edge_more_than_10_percent_longer():
    # The detour is about 10.5% longer than the straight street: a different street, kept.
    nodes_gdf, edges_gdf = _pair_with_detour([(0, 0), (5, 2.35), (10, 0)])
    assert 1.1 < edges_gdf.geometry.length.loc[11] / 10 < 1.11

    clean_nodes, clean_edges = nt.clean_same_vertexes_edges(nodes_gdf.copy(), edges_gdf.copy())

    assert len(clean_edges) == 2
    assert len(clean_nodes) == 2


def test_clean_same_vertexes_edges_leaves_self_loops_alone():
    nodes_gdf = _nodes([{"nodeID": 1, "x": 0.0, "y": 0.0}, {"nodeID": 2, "x": 10.0, "y": 0.0}])
    edges_gdf = _edges(
        [
            {"edgeID": 10, "u": 1, "v": 2, "geometry": LineString([(0, 0), (10, 0)])},
            {
                "edgeID": 11,
                "u": 1,
                "v": 1,
                "geometry": LineString([(0, 0), (0, 5), (5, 5), (0, 0)]),
            },
            {
                "edgeID": 12,
                "u": 1,
                "v": 1,
                "geometry": LineString([(0, 0), (0, -9), (-9, 0), (0, 0)]),
            },
        ]
    )

    clean_nodes, clean_edges = nt.clean_same_vertexes_edges(nodes_gdf.copy(), edges_gdf.copy())

    assert sorted(clean_edges["edgeID"].tolist()) == [10, 11, 12]
    assert sorted(clean_nodes["nodeID"].tolist()) == [1, 2]


def test_clean_same_vertexes_and_duplicate_edges_do_not_modify_their_input():
    nodes_gdf, edges_gdf = _crescent()
    columns = list(edges_gdf.columns)

    nt.clean_same_vertexes_edges(nodes_gdf, edges_gdf)
    nt.clean_duplicate_edges(nodes_gdf, edges_gdf)

    assert list(edges_gdf.columns) == columns


def _two_segments(**columns):
    # A 10 m segment and a 30 m one meeting at pseudo-node 2, with extra columns per segment.
    nodes_gdf = _nodes(
        [
            {"nodeID": 1, "x": 0.0, "y": 0.0},
            {"nodeID": 2, "x": 10.0, "y": 0.0},
            {"nodeID": 3, "x": 40.0, "y": 0.0},
        ]
    )
    edges_gdf = _edges(
        [
            {"edgeID": 10, "u": 1, "v": 2, "geometry": LineString([(0, 0), (10, 0)])},
            {"edgeID": 11, "u": 2, "v": 3, "geometry": LineString([(10, 0), (40, 0)])},
        ]
    )
    for name, values in columns.items():
        edges_gdf[name] = pd.Series(values, index=edges_gdf.index)
    return nodes_gdf, edges_gdf


def test_simplify_graph_does_not_let_a_missing_date_outvote_a_value():
    date = pd.Timestamp("2020-01-01")
    nodes_gdf, edges_gdf = _two_segments(surveyed=[date, pd.NaT])

    _, simplified = nt.simplify_graph(nodes_gdf, edges_gdf)

    assert len(simplified) == 1
    assert simplified.iloc[0]["surveyed"] == date  # 30 m of NaT does not win


def test_simplify_graph_keeps_column_dtypes():
    nodes_gdf, edges_gdf = _two_segments(
        highway=pd.Categorical(["primary", "secondary"]), lanes=[1, 2]
    )

    _, simplified = nt.simplify_graph(nodes_gdf, edges_gdf)

    assert isinstance(simplified["highway"].dtype, pd.CategoricalDtype)
    assert simplified.iloc[0]["highway"] == "secondary"
    assert simplified["lanes"].dtype == edges_gdf["lanes"].dtype
    assert simplified.iloc[0]["lanes"] == 2


def test_simplify_graph_merges_into_an_int32_column():
    # pandas refuses an object Series into an int32 column; the merged value must be cast back.
    nodes_gdf, edges_gdf = _two_segments(lanes=pd.array([1, 2], dtype="int32"))
    edges_gdf["lanes"] = edges_gdf["lanes"].astype("int32")

    _, simplified = nt.simplify_graph(nodes_gdf, edges_gdf)

    assert simplified["lanes"].dtype == "int32"
    assert simplified.iloc[0]["lanes"] == 2


def test_clean_same_vertexes_edges_collapses_a_longer_street_mapped_twice():
    # A straight street, and one crescent mapped twice about 1 m apart.
    nodes_gdf, edges_gdf = _pair_with_detour([(0, 0), (5, 10), (10, 0)])
    extra = _edges(
        [{"edgeID": 12, "u": 2, "v": 1, "geometry": LineString([(10, 0), (5, 11), (0, 0)])}]
    )
    edges_gdf = pd.concat([edges_gdf, extra])

    clean_nodes, clean_edges = nt.clean_same_vertexes_edges(nodes_gdf.copy(), edges_gdf.copy())

    # The straight street, and the crescent once.
    assert sorted(clean_edges["edgeID"].tolist()) in ([10, 11], [10, 12])
    assert len(clean_nodes) == 2


def test_clean_same_vertexes_edges_keeps_two_sides_of_a_block_apart():
    # Two streets of equal length between opposite corners of a block, 10 m apart.
    nodes_gdf = _nodes([{"nodeID": 1, "x": 0.0, "y": 0.0}, {"nodeID": 2, "x": 10.0, "y": 10.0}])
    edges_gdf = _edges(
        [
            {"edgeID": 10, "u": 1, "v": 2, "geometry": LineString([(0, 0), (0, 10), (10, 10)])},
            {"edgeID": 11, "u": 1, "v": 2, "geometry": LineString([(0, 0), (10, 0), (10, 10)])},
        ]
    )

    clean_nodes, clean_edges = nt.clean_same_vertexes_edges(nodes_gdf.copy(), edges_gdf.copy())

    assert sorted(clean_edges["edgeID"].tolist()) == [10, 11]  # both sides, as parallel edges

    # Within a larger tolerance they are taken as one street.
    _, merged = nt.clean_same_vertexes_edges(
        nodes_gdf.copy(), edges_gdf.copy(), same_vertexes_tolerance=15
    )
    assert len(merged) == 1


def test_clean_same_vertexes_edges_keeps_the_central_of_three_copies():
    # A centreline with a copy 4 m either side. The outer copies are the shortest and 8 m apart,
    # so only through the centreline are they the same street; the centreline is what is kept.
    nodes_gdf = _nodes([{"nodeID": 1, "x": 0.0, "y": 0.0}, {"nodeID": 2, "x": 100.0, "y": 0.0}])
    centre = [(0, 0), (20, 1), (40, -1), (60, 1), (80, -1), (100, 0)]
    edges_gdf = _edges(
        [
            {"edgeID": 10, "u": 1, "v": 2, "geometry": LineString([(0, 0), (50, 4), (100, 0)])},
            {"edgeID": 11, "u": 1, "v": 2, "geometry": LineString([(0, 0), (50, -4), (100, 0)])},
            {"edgeID": 12, "u": 1, "v": 2, "geometry": LineString(centre)},
        ]
    )
    assert edges_gdf.geometry.length.loc[12] > edges_gdf.geometry.length.loc[10]

    _, clean_edges = nt.clean_same_vertexes_edges(nodes_gdf.copy(), edges_gdf.copy())

    assert clean_edges["edgeID"].tolist() == [12]


def test_clean_network_merges_a_pseudo_node_into_a_parallel_edge():
    # Node 3 sits on a crescent between junctions 1 and 2, which a straight street also joins.
    nodes_gdf = _nodes(
        [
            {"nodeID": 1, "x": 0.0, "y": 0.0},
            {"nodeID": 2, "x": 10.0, "y": 0.0},
            {"nodeID": 3, "x": 5.0, "y": 10.0},
            {"nodeID": 4, "x": -10.0, "y": 0.0},
            {"nodeID": 5, "x": 20.0, "y": 0.0},
        ]
    )
    edges_gdf = _edges(
        [
            {"edgeID": 10, "u": 1, "v": 2, "geometry": LineString([(0, 0), (10, 0)])},
            {"edgeID": 11, "u": 1, "v": 3, "geometry": LineString([(0, 0), (5, 10)])},
            {"edgeID": 12, "u": 3, "v": 2, "geometry": LineString([(5, 10), (10, 0)])},
            {"edgeID": 13, "u": 4, "v": 1, "geometry": LineString([(-10, 0), (0, 0)])},
            {"edgeID": 14, "u": 2, "v": 5, "geometry": LineString([(10, 0), (20, 0)])},
        ]
    )

    clean_nodes, clean_edges = nt.clean_network(nodes_gdf, edges_gdf, remove_islands=False)

    assert 3 not in clean_nodes["nodeID"].tolist()
    pairs = [frozenset(pair) for pair in zip(clean_edges["u"], clean_edges["v"], strict=False)]
    assert pairs.count(frozenset((1, 2))) == 2
    assert round(clean_edges.geometry.length.sum(), 6) == round(edges_gdf.geometry.length.sum(), 6)


# A loop street leaves junction 1 and returns to it; 3-1-4 is the street it hangs off.
_LOOP_MAPPINGS = {
    "through one node": (
        {2: (10.0, 10.0)},
        [(1, 2, [(0, 0), (0, 10), (10, 10)]), (2, 1, [(10, 10), (10, 0), (0, 0)])],
    ),
    "through two nodes": (
        {2: (10.0, 10.0), 5: (10.0, 0.0)},
        [
            (1, 2, [(0, 0), (0, 10), (10, 10)]),
            (2, 5, [(10, 10), (10, 0)]),
            (5, 1, [(10, 0), (0, 0)]),
        ],
    ),
    "as one edge": ({}, [(1, 1, [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])]),
}


@pytest.mark.parametrize("mapping", sorted(_LOOP_MAPPINGS))
@pytest.mark.parametrize("self_loops", [True, False])
def test_clean_network_loop_street_leaves_no_pseudo_node(mapping, self_loops):
    loop_nodes, loop_edges = _LOOP_MAPPINGS[mapping]
    coords = {1: (0.0, 0.0), 3: (-10.0, 0.0), 4: (0.0, -10.0), **loop_nodes}
    nodes_gdf = _nodes([{"nodeID": n, "x": x, "y": y} for n, (x, y) in coords.items()])
    rows = [(3, 1, [(-10, 0), (0, 0)]), (1, 4, [(0, 0), (0, -10)]), *loop_edges]
    edges_gdf = _edges(
        [
            {"edgeID": 10 + i, "u": u, "v": v, "geometry": LineString(line)}
            for i, (u, v, line) in enumerate(rows)
        ]
    )

    clean_nodes, clean_edges = nt.clean_network(
        nodes_gdf, edges_gdf, remove_islands=False, self_loops=self_loops
    )

    assert 2 not in nt.nodes_degree(clean_edges).values()
    loops = clean_edges[clean_edges["u"] == clean_edges["v"]]
    if self_loops:
        assert loops.empty
        assert round(clean_edges.geometry.length.sum(), 6) == 20.0
    else:
        assert len(loops) == 1
        assert round(loops.geometry.length.iloc[0], 6) == 40.0
        assert round(clean_edges.geometry.length.sum(), 6) == 60.0


def _through_junction():
    # Edge 1000 runs from 100 to 300 through junction 200 without being split there; 200 is where
    # edge 1001 starts. Nodes carry an attribute and non-contiguous IDs.
    nodes_gdf = _nodes(
        [
            {"nodeID": 100, "x": 0.0, "y": 0.0, "elev": 1.0},
            {"nodeID": 200, "x": 10.0, "y": 0.0, "elev": 2.0},
            {"nodeID": 300, "x": 20.0, "y": 0.0, "elev": 3.0},
            {"nodeID": 400, "x": 10.0, "y": 10.0, "elev": 4.0},
        ]
    )
    edges_gdf = _edges(
        [
            {
                "edgeID": 1000,
                "u": 100,
                "v": 300,
                "name": "High St",
                "geometry": LineString([(0, 0), (10, 0), (20, 0)]),
            },
            {
                "edgeID": 1001,
                "u": 200,
                "v": 400,
                "name": "Mill Ln",
                "geometry": LineString([(10, 0), (10, 10)]),
            },
        ]
    )
    return nodes_gdf, edges_gdf


def test_fix_fake_self_loops_keeps_ids_and_attributes():
    nodes_gdf, edges_gdf = _through_junction()

    fixed_nodes, fixed_edges = nt.fix_fake_self_loops(nodes_gdf, edges_gdf)

    pd.testing.assert_frame_equal(fixed_nodes, nodes_gdf)
    rows = {
        (edge_id, u, v, name)
        for edge_id, u, v, name in zip(
            fixed_edges["edgeID"],
            fixed_edges["u"],
            fixed_edges["v"],
            fixed_edges["name"],
            strict=True,
        )
    }
    assert rows == {
        (1000, 100, 200, "High St"),
        (1002, 200, 300, "High St"),
        (1001, 200, 400, "Mill Ln"),
    }
    assert list(fixed_edges.index) == list(fixed_edges["edgeID"])
    assert fixed_edges["edgeID"].dtype == edges_gdf["edgeID"].dtype


def test_fix_network_topology_adds_one_shared_node_at_a_crossing_and_keeps_ids():
    # Two ways cross at (5, 0), a vertex of both, where no node exists.
    nodes_gdf = _nodes(
        [
            {"nodeID": 7, "x": 0.0, "y": 0.0, "elev": 1.0},
            {"nodeID": 8, "x": 10.0, "y": 0.0, "elev": 1.0},
            {"nodeID": 9, "x": 5.0, "y": -5.0, "elev": 1.0},
            {"nodeID": 11, "x": 5.0, "y": 5.0, "elev": 1.0},
        ]
    )
    edges_gdf = _edges(
        [
            {"edgeID": 3, "u": 7, "v": 8, "geometry": LineString([(0, 0), (5, 0), (10, 0)])},
            {"edgeID": 5, "u": 9, "v": 11, "geometry": LineString([(5, -5), (5, 0), (5, 5)])},
        ]
    )

    fixed_nodes, fixed_edges = nt.fix_network_topology(nodes_gdf, edges_gdf)

    assert sorted(fixed_nodes["nodeID"]) == [7, 8, 9, 11, 12]
    new_node = fixed_nodes.loc[12]
    assert (new_node.geometry.x, new_node.geometry.y) == (5.0, 0.0)
    assert fixed_nodes.loc[[7, 8, 9, 11], "elev"].tolist() == [1.0] * 4
    assert sorted(fixed_edges["edgeID"]) == [3, 5, 6, 7]
    pairs = {frozenset(pair) for pair in zip(fixed_edges["u"], fixed_edges["v"], strict=True)}
    assert pairs == {frozenset(p) for p in [(7, 12), (12, 8), (9, 12), (12, 11)]}


def test_clean_network_keeps_protected_ids_when_a_way_is_split():
    # Edge 1000 is split at junction 200 before anything else, which used to renumber every node,
    # so nodes_to_keep_regardless pointed at another node. Mill Ln continues 400-500-600; 400 is a
    # protected pseudo-node, 500 an unprotected one.
    nodes_gdf, edges_gdf = _through_junction()
    nodes_gdf = pd.concat(
        [
            nodes_gdf,
            _nodes(
                [
                    {"nodeID": 500, "x": 10.0, "y": 20.0, "elev": 5.0},
                    {"nodeID": 600, "x": 10.0, "y": 30.0, "elev": 6.0},
                ]
            ),
        ]
    )
    edges_gdf = pd.concat(
        [
            edges_gdf,
            _edges(
                [
                    {
                        "edgeID": 1003,
                        "u": 400,
                        "v": 500,
                        "name": "Mill Ln",
                        "geometry": LineString([(10, 10), (10, 20)]),
                    },
                    {
                        "edgeID": 1004,
                        "u": 500,
                        "v": 600,
                        "name": "Mill Ln",
                        "geometry": LineString([(10, 20), (10, 30)]),
                    },
                ]
            ),
        ]
    )

    clean_nodes, _ = nt.clean_network(
        nodes_gdf, edges_gdf, remove_islands=False, nodes_to_keep_regardless=[400]
    )

    assert sorted(clean_nodes["nodeID"]) == [100, 200, 300, 400, 600]
    assert clean_nodes.loc[100, "elev"] == 1.0


def test_clean_network_drops_a_node_left_only_with_a_removed_self_loop():
    # A loop street on its own, mapped as two streets between 9 and 19: merging 19 closes a
    # self-loop at 9, which self_loops=True then removes. Node 9 must not be left behind.
    nodes_gdf = _nodes(
        [
            {"nodeID": 1, "x": 0.0, "y": 0.0},
            {"nodeID": 2, "x": 10.0, "y": 0.0},
            {"nodeID": 3, "x": 20.0, "y": 0.0},
            {"nodeID": 9, "x": 100.0, "y": 0.0},
            {"nodeID": 19, "x": 110.0, "y": 10.0},
        ]
    )
    edges_gdf = _edges(
        [
            {"edgeID": 10, "u": 1, "v": 2, "geometry": LineString([(0, 0), (10, 0)])},
            {"edgeID": 11, "u": 2, "v": 3, "geometry": LineString([(10, 0), (20, 0)])},
            {
                "edgeID": 12,
                "u": 9,
                "v": 19,
                "geometry": LineString([(100, 0), (100, 10), (110, 10)]),
            },
            {
                "edgeID": 13,
                "u": 19,
                "v": 9,
                "geometry": LineString([(110, 10), (110, 0), (100, 0)]),
            },
        ]
    )

    clean_nodes, clean_edges = nt.clean_network(
        nodes_gdf, edges_gdf, remove_islands=False, self_loops=True
    )

    assert set(clean_nodes["nodeID"]) == set(clean_edges["u"]).union(clean_edges["v"]) == {1, 3}


def test_fix_fake_self_loops_splits_a_3d_edge_and_keeps_z():
    nodes_gdf, edges_gdf = _through_junction()
    edges_gdf["geometry"] = [
        LineString([(0, 0, 5), (10, 0, 6), (20, 0, 7)]),
        LineString([(10, 0, 6), (10, 10, 8)]),
    ]

    _, fixed_edges = nt.fix_fake_self_loops(nodes_gdf, edges_gdf)

    assert sorted(fixed_edges["edgeID"]) == [1000, 1001, 1002]
    assert [list(geometry.coords) for geometry in fixed_edges.sort_index().geometry] == [
        [(0.0, 0.0, 5.0), (10.0, 0.0, 6.0)],
        [(10.0, 0.0, 6.0), (10.0, 10.0, 8.0)],
        [(10.0, 0.0, 6.0), (20.0, 0.0, 7.0)],
    ]


def test_split_does_not_leave_a_zero_length_piece_at_a_repeated_end_vertex():
    nodes_gdf = _nodes([{"nodeID": 1, "x": 0.0, "y": 0.0}, {"nodeID": 2, "x": 10.0, "y": 0.0}])
    edges_gdf = _edges(
        [{"edgeID": 5, "u": 1, "v": 2, "geometry": LineString([(0, 0), (5, 1), (10, 0), (10, 0)])}]
    )

    _, fixed_edges = nt.fix_fake_self_loops(nodes_gdf, edges_gdf)

    assert (fixed_edges.geometry.length > 0).all()
    assert (fixed_edges["u"] != fixed_edges["v"]).all()


def test_split_joins_an_edge_end_whose_node_point_is_off_by_float_noise():
    # Node 3's point is 1e-7 off the end of edge 20, which ends on edge 10's internal vertex.
    nodes_gdf = _nodes(
        [
            {"nodeID": 1, "x": 0.0, "y": 0.0},
            {"nodeID": 2, "x": 10.0, "y": 0.0},
            {"nodeID": 3, "x": 5.0000001, "y": 0.0},
            {"nodeID": 4, "x": 5.0, "y": 5.0},
        ]
    )
    edges_gdf = _edges(
        [
            {"edgeID": 10, "u": 1, "v": 2, "geometry": LineString([(0, 0), (5, 0), (10, 0)])},
            {"edgeID": 20, "u": 4, "v": 3, "geometry": LineString([(5, 5), (5, 0)])},
        ]
    )

    fixed_nodes, fixed_edges = nt.fix_fake_self_loops(nodes_gdf, edges_gdf)

    assert sorted(fixed_nodes["nodeID"]) == [1, 2, 3, 4]
    assert nt.nodes_degree(fixed_edges)[3] == 3


def test_split_orients_pieces_of_an_edge_stored_against_its_labels():
    # Edge 10 is labelled 1 -> 2 but its line runs from node 2 to node 1, through node 3.
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
            {"edgeID": 10, "u": 1, "v": 2, "geometry": LineString([(10, 0), (5, 0), (0, 0)])},
            {"edgeID": 20, "u": 3, "v": 4, "geometry": LineString([(5, 0), (5, 5)])},
        ]
    )

    fixed_nodes, fixed_edges = nt.fix_fake_self_loops(nodes_gdf, edges_gdf)

    for u, v, line in zip(fixed_edges["u"], fixed_edges["v"], fixed_edges.geometry, strict=True):
        assert line.coords[0] == fixed_nodes.loc[u].geometry.coords[0]
        assert line.coords[-1] == fixed_nodes.loc[v].geometry.coords[0]


def test_split_keeps_edge_and_node_dtypes():
    nodes_gdf = _nodes(
        [
            {"nodeID": 7, "x": 0.0, "y": 0.0, "degree": 1},
            {"nodeID": 8, "x": 10.0, "y": 0.0, "degree": 1},
            {"nodeID": 9, "x": 5.0, "y": -5.0, "degree": 1},
            {"nodeID": 11, "x": 5.0, "y": 5.0, "degree": 1},
        ]
    )
    edges_gdf = _edges(
        [
            {"edgeID": 3, "u": 7, "v": 8, "geometry": LineString([(0, 0), (5, 0), (10, 0)])},
            {"edgeID": 5, "u": 9, "v": 11, "geometry": LineString([(5, -5), (5, 0), (5, 5)])},
        ]
    )
    edges_gdf["highway"] = pd.Categorical(["primary", "residential"])
    edges_gdf["oneway"] = pd.array([True, False], dtype="boolean")

    fixed_nodes, fixed_edges = nt.fix_network_topology(nodes_gdf, edges_gdf)

    assert isinstance(fixed_edges["highway"].dtype, pd.CategoricalDtype)
    assert fixed_edges["oneway"].dtype == "boolean"
    assert fixed_edges["edgeID"].dtype == edges_gdf["edgeID"].dtype
    assert fixed_nodes["degree"].dtype == "Int64"  # the new junction has no degree yet
    assert fixed_nodes["nodeID"].dtype == nodes_gdf["nodeID"].dtype


def test_new_junction_takes_z_interpolated_between_the_edge_ends():
    nodes_gdf = _nodes(
        [
            {"nodeID": 1, "x": 0.0, "y": 0.0, "z": 10.0},
            {"nodeID": 2, "x": 10.0, "y": 0.0, "z": 20.0},
            {"nodeID": 3, "x": 2.5, "y": -5.0, "z": 0.0},
            {"nodeID": 4, "x": 2.5, "y": 5.0, "z": 0.0},
        ]
    )
    edges_gdf = _edges(
        [
            {"edgeID": 10, "u": 1, "v": 2, "geometry": LineString([(0, 0), (2.5, 0), (10, 0)])},
            {"edgeID": 20, "u": 3, "v": 4, "geometry": LineString([(2.5, -5), (2.5, 0), (2.5, 5)])},
        ]
    )

    fixed_nodes, _ = nt.fix_network_topology(nodes_gdf, edges_gdf)

    assert fixed_nodes.loc[5, "z"] == 12.5


def test_fix_functions_return_copies_when_nothing_is_split():
    nodes_gdf, edges_gdf = _star()

    for fix in (nt.fix_fake_self_loops, nt.fix_network_topology):
        fixed_nodes, fixed_edges = fix(nodes_gdf, edges_gdf)
        assert fixed_nodes is not nodes_gdf
        assert fixed_edges is not edges_gdf
        pd.testing.assert_frame_equal(fixed_edges, edges_gdf)
