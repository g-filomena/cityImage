"""The network layers, after a GeoPackage round trip, as GeoMason-light reads them.

GeoMason's ``Graph.fromStreetJunctionsSegments`` builds the graph from the segments' end
coordinates, not from ``u``/``v``: an end creates or finds the node at exactly that (x, y), and a
junction's attributes (``nodeID``, district, centrality) reach a node only when the junction's point
sits exactly there. The dual graph is built the same way from the dual edges and the dual nodes'
points. So every end must land exactly on its own node, no two nodes may share a point, and every
segment must be a single LineString with two distinct points - otherwise GeoMason adds bare nodes,
merges two nodes into one, or drops a segment, all without an error.

The York network goes through the preparation paths a pipeline uses (cleaning with different
options, node consolidation, the dual graph), is written to and read back from a GeoPackage, and
is checked against each of those requirements.
"""

from __future__ import annotations

from collections import Counter

import geopandas as gpd
import numpy as np
import pytest

import cityImage as ci
from tests.fixtures.cityimage_minimal import YORK_CRS, york_raw_network

PATHS = {
    "clean": {"clean": {}, "consolidate": None},
    "clean_topology_dead_ends": {
        "clean": {"fix_topology": True, "dead_ends": True},
        "consolidate": None,
    },
    "pipeline": {  # as PedSimCity's 00_city_preparation does
        "clean": {
            "dead_ends": True,
            "remove_islands": True,
            "same_vertexes_edges": True,
            "self_loops": True,
            "fix_topology": True,
        },
        "consolidate": 15.0,
    },
    "pipeline_keep_loops_direction": {
        "clean": {"fix_topology": True, "self_loops": False, "preserve_direction": True},
        "consolidate": 15.0,
    },
    "wide_consolidation": {"clean": {"fix_topology": True}, "consolidate": 40.0},
}


def _round_trip(gdf, path, layer):
    gdf.to_file(path, layer=layer, driver="GPKG")
    return gpd.read_file(path, layer=layer)


@pytest.fixture(
    scope="module",
    params=["centre", pytest.param("whole_town", marks=pytest.mark.slow)],
)
def raw(request):
    return york_raw_network(whole_town=request.param == "whole_town")


@pytest.fixture(scope="module", params=list(PATHS), ids=list(PATHS))
def layers(request, raw, tmp_path_factory):
    path = PATHS[request.param]
    nodes, edges = ci.clean_network(raw[0].copy(), raw[1].copy(), **path["clean"])
    if path["consolidate"] is not None:
        nodes, edges = ci.consolidate_nodes(
            nodes, edges, consolidate_edges_too=True, tolerance=path["consolidate"]
        )
        nodes = nodes.drop(columns="old_nodeID")  # a list column, not a GeoPackage field
    nodes_dual, edges_dual = ci.dual_gdf(nodes, edges, YORK_CRS)
    nodes_dual = nodes_dual.drop(columns="intersecting")

    gpkg = tmp_path_factory.mktemp(request.param) / "network.gpkg"
    return {
        "nodes": _round_trip(nodes, gpkg, "nodes"),
        "edges": _round_trip(edges, gpkg, "edges"),
        "nodesDual": _round_trip(nodes_dual, gpkg, "nodesDual"),
        "edgesDual": _round_trip(edges_dual, gpkg, "edgesDual"),
    }


def _xy(point):
    return tuple(point.coords[0][:2])


def _ends(lines):
    return [(tuple(line.coords[0][:2]), tuple(line.coords[-1][:2])) for line in lines]


def _node_at(points, ids):
    return dict(zip((_xy(point) for point in points), ids, strict=True))


def _is_integer(series):
    return series.notna().all() and np.array_equal(series, series.round())


# --- primal ---------------------------------------------------------------------------------


def test_segments_are_single_lines_and_junctions_points(layers):
    assert set(layers["edges"].geom_type) == {"LineString"}
    assert set(layers["nodes"].geom_type) == {"Point"}


def test_ids_are_unique_integers(layers):
    nodes, edges = layers["nodes"], layers["edges"]
    assert nodes["nodeID"].is_unique and edges["edgeID"].is_unique
    for column in (nodes["nodeID"], edges["edgeID"], edges["u"], edges["v"]):
        assert _is_integer(column)


def test_no_two_junctions_share_a_point(layers):
    points = Counter(_xy(point) for point in layers["nodes"].geometry)
    assert max(points.values()) == 1


def test_x_and_y_columns_are_the_junction_point(layers):
    nodes = layers["nodes"]
    assert np.array_equal(nodes["x"], nodes.geometry.x)
    assert np.array_equal(nodes["y"], nodes.geometry.y)


def test_every_segment_has_two_distinct_points(layers):
    for line in layers["edges"].geometry:
        assert len({coord[:2] for coord in line.coords}) >= 2


def test_every_segment_end_is_exactly_its_own_junction(layers):
    nodes, edges = layers["nodes"], layers["edges"]
    node_at = _node_at(nodes.geometry, nodes["nodeID"])
    for (start, end), u, v in zip(_ends(edges.geometry), edges["u"], edges["v"], strict=True):
        assert (node_at.get(start), node_at.get(end)) == (u, v)  # exact, and running u -> v


def test_every_junction_is_reached_by_a_segment(layers):
    nodes, edges = layers["nodes"], layers["edges"]
    assert set(nodes["nodeID"]) == set(edges["u"]) | set(edges["v"])


def test_no_segment_runs_along_another(layers):
    # Two segments on exactly the same coordinates are one street twice, and share a dual point.
    keys = Counter()
    for line in layers["edges"].geometry:
        coords = tuple(coord[:2] for coord in line.coords)
        keys[min(coords, coords[::-1])] += 1
    assert max(keys.values()) == 1


def test_lengths_are_the_geometry(layers):
    edges = layers["edges"]
    assert np.allclose(edges["length"], edges.geometry.length)


# --- dual -----------------------------------------------------------------------------------


def test_every_segment_is_one_dual_node(layers):
    nodes_dual, edges = layers["nodesDual"], layers["edges"]
    assert nodes_dual["edgeID"].is_unique
    assert set(nodes_dual["edgeID"]) == set(edges["edgeID"])
    assert set(nodes_dual.geom_type) == {"Point"}


def test_no_two_dual_nodes_share_a_point(layers):
    points = Counter(_xy(point) for point in layers["nodesDual"].geometry)
    assert max(points.values()) == 1


def test_every_dual_edge_end_is_exactly_its_own_dual_node(layers):
    nodes_dual, edges_dual = layers["nodesDual"], layers["edgesDual"]
    assert set(edges_dual.geom_type) == {"LineString"}
    node_at = _node_at(nodes_dual.geometry, nodes_dual["edgeID"])
    for (start, end), u, v in zip(
        _ends(edges_dual.geometry), edges_dual["u"], edges_dual["v"], strict=True
    ):
        assert (node_at.get(start), node_at.get(end)) == (u, v)


def test_every_dual_edge_joins_two_segments_at_a_junction(layers):
    edges, edges_dual = layers["edges"], layers["edgesDual"]
    ends = {i: {u, v} for i, u, v in zip(edges["edgeID"], edges["u"], edges["v"], strict=True)}
    for u, v in zip(edges_dual["u"], edges_dual["v"], strict=True):
        assert u != v and ends[u] & ends[v]
    assert not edges_dual[["u", "v"]].apply(frozenset, axis=1).duplicated().any()


def test_every_pair_of_segments_at_a_junction_is_a_dual_edge(layers):
    edges, edges_dual = layers["edges"], layers["edgesDual"]
    at_node = {}
    for i, u, v in zip(edges["edgeID"], edges["u"], edges["v"], strict=True):
        for node in {u, v}:
            at_node.setdefault(node, []).append(i)
    expected = {frozenset((a, b)) for ids in at_node.values() for a in ids for b in ids if a != b}
    assert set(edges_dual[["u", "v"]].apply(frozenset, axis=1)) == expected


def test_dual_edge_attributes_are_complete(layers):
    edges_dual = layers["edgesDual"]
    assert edges_dual["deg"].between(0, 180).all()
    assert set(edges_dual["oneway"]) <= {0, 1}
    assert np.isfinite(edges_dual["length"]).all()
