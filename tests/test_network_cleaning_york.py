"""clean_network on a real street network (York, Ontario), checked against the rules in
docs/network_topology.md rather than against fixed counts.

The network comes from whole OSM ways, so most junctions are un-noded: it exercises pass-through
splitting, topology fixing, loop streets (closed service ways), parallel streets, islands and dead
ends on genuine data. The central subset runs in a few seconds; the whole town (1,005 ways, 108 km)
is marked slow.
"""

from __future__ import annotations

from collections import Counter

import networkx as nx
import pytest
from shapely.geometry import Point

import cityImage as ci
from tests.fixtures.cityimage_minimal import york_raw_network

OPTIONS = {
    "defaults": {},
    "fix_topology": {"fix_topology": True},
    "fix_topology_dead_ends": {"fix_topology": True, "dead_ends": True},
    "everything_on": {
        "fix_topology": True,
        "dead_ends": True,
        "self_loops": True,
        "remove_islands": True,
    },
    "keep_islands_no_same_vertex": {"remove_islands": False, "same_vertexes_edges": False},
    "preserve_direction": {"fix_topology": True, "preserve_direction": True},
}


def _key(coord):
    return (round(float(coord[0]), 6), round(float(coord[1]), 6))


def _degree(edges):
    return Counter(list(edges["u"]) + list(edges["v"]))


@pytest.fixture(
    scope="module",
    params=["centre", pytest.param("whole_town", marks=pytest.mark.slow)],
)
def raw(request):
    return york_raw_network(whole_town=request.param == "whole_town")


@pytest.fixture(scope="module", params=list(OPTIONS), ids=list(OPTIONS))
def cleaned(request, raw):
    options = OPTIONS[request.param]
    nodes, edges = ci.clean_network(raw[0].copy(), raw[1].copy(), **options)
    return options, nodes, edges


def test_no_vertex_is_invented_except_on_centre_lines(raw, cleaned):
    # A new vertex only comes from the centre line of a street mapped an even number of times,
    # so it lies within half of same_vertexes_tolerance (5 m) of the input.
    _, _, edges = cleaned
    raw_lines = raw[1].geometry
    raw_vertices = {_key(c) for line in raw_lines for c in line.coords}
    new = {_key(c) for line in edges.geometry for c in line.coords} - raw_vertices
    sindex = raw_lines.sindex
    for x, y in new:
        point = Point(x, y)
        nearest = raw_lines.iloc[sindex.query(point.buffer(2.5))]
        assert nearest.distance(point).min() <= 2.5 + 1e-6


def test_edges_and_nodes_reference_each_other(cleaned):
    _, nodes, edges = cleaned
    ids = set(nodes["nodeID"])
    assert set(edges["u"]) | set(edges["v"]) == ids
    assert nodes["nodeID"].is_unique and edges["edgeID"].is_unique
    assert list(nodes.index) == list(nodes["nodeID"])
    assert list(edges.index) == list(edges["edgeID"])


def test_edge_ends_lie_on_their_nodes(cleaned):
    _, nodes, edges = cleaned
    xy = {n: _key(p.coords[0]) for n, p in zip(nodes["nodeID"], nodes.geometry, strict=True)}
    for u, v, line in zip(edges["u"], edges["v"], edges.geometry, strict=True):
        assert _key(line.coords[0]) == xy[u]
        assert _key(line.coords[-1]) == xy[v]


def test_no_pseudo_node_zero_length_or_duplicate_edge_is_left(cleaned):
    options, _, edges = cleaned
    degree = _degree(edges)
    loop_only = {
        u for u, v in zip(edges["u"], edges["v"], strict=True) if u == v and degree[u] == 2
    }
    assert [n for n, d in degree.items() if d == 2 and n not in loop_only] == []
    assert (edges.geometry.length > 0).all()
    assert not edges.geometry.apply(lambda line: line.normalize().wkb).duplicated().any()
    if options.get("self_loops"):
        assert (edges["u"] != edges["v"]).all()


def test_no_street_is_left_mapped_twice(cleaned):
    # Parallel edges are kept only when they are different streets: no pair may still pass the
    # same-street rule (within 10 % in length and 5 m Hausdorff). With preserve_direction, u->v and
    # v->u are different edges, such as the two directions of a street mapped as one-way ways.
    options, _, edges = cleaned
    if options.get("same_vertexes_edges") is False:
        pytest.skip("same-vertex cleaning is off")
    by_pair = {}
    for u, v, line in zip(edges["u"], edges["v"], edges.geometry, strict=True):
        if u != v:
            pair = (u, v) if options.get("preserve_direction") else frozenset((u, v))
            by_pair.setdefault(pair, []).append(line)
    for lines in by_pair.values():
        for i, a in enumerate(lines):
            for b in lines[i + 1 :]:
                short, long = sorted((a.length, b.length))
                assert not (long <= short * 1.1 and a.hausdorff_distance(b) <= 5.0)


def test_islands_are_removed_only_when_asked(cleaned):
    options, _, edges = cleaned
    graph = nx.MultiGraph(list(zip(edges["u"], edges["v"], strict=True)))
    components = nx.number_connected_components(graph)
    if options.get("remove_islands", True):
        assert components == 1
    else:
        assert components > 1  # the clip leaves fragments; they must survive


def test_ids_are_kept_and_new_ones_come_after_the_input(raw, cleaned):
    _, nodes, edges = cleaned
    raw_nodes, raw_edges = raw
    new_nodes = set(nodes["nodeID"]) - set(raw_nodes["nodeID"])
    new_edges = set(edges["edgeID"]) - set(raw_edges["edgeID"])
    assert all(n > raw_nodes["nodeID"].max() for n in new_nodes)
    assert all(e > raw_edges["edgeID"].max() for e in new_edges)


def test_attributes_come_from_the_input(raw, cleaned):
    _, _, edges = cleaned
    for column in ("name", "highway"):
        assert set(edges[column].dropna()) <= set(raw[1][column].dropna())


def test_cleaning_twice_changes_nothing(cleaned):
    options, nodes, edges = cleaned
    nodes_again, edges_again = ci.clean_network(nodes.copy(), edges.copy(), **options)
    assert set(nodes_again["nodeID"]) == set(nodes["nodeID"])
    assert set(edges_again["edgeID"]) == set(edges["edgeID"])
    assert edges_again.geometry.length.sum() == pytest.approx(edges.geometry.length.sum())


def test_nothing_is_lost_when_nothing_is_asked_to_remove_it(raw):
    raw_nodes, raw_edges = raw
    _, edges = ci.clean_network(
        raw_nodes.copy(),
        raw_edges.copy(),
        remove_islands=False,
        same_vertexes_edges=False,
        self_loops=False,
    )
    assert edges.geometry.length.sum() == pytest.approx(raw_edges.geometry.length.sum())


def test_loop_streets_are_removed_unless_self_loops_is_false(raw):
    raw_nodes, raw_edges = raw
    _, kept = ci.clean_network(raw_nodes.copy(), raw_edges.copy(), self_loops=False)
    _, dropped = ci.clean_network(raw_nodes.copy(), raw_edges.copy())
    assert (kept["u"] == kept["v"]).any()  # closed service ways and footways
    assert (dropped["u"] != dropped["v"]).all()


def test_protected_nodes_survive_peeling_and_merging(raw):
    raw_nodes, raw_edges = raw
    # Protect pseudo-nodes and dead-end tips of the main network, where island removal
    # (which protection does not stop) cannot take them.
    _, main = ci.clean_network(raw_nodes.copy(), raw_edges.copy(), fix_topology=True)
    main_nodes = set(main["u"]) | set(main["v"])
    degree = _degree(raw_edges)
    pseudo = [n for n, d in sorted(degree.items()) if d == 2 and n in main_nodes][:5]
    tips = [n for n, d in sorted(degree.items()) if d == 1 and n in main_nodes][:5]
    assert pseudo and tips

    nodes, _ = ci.clean_network(
        raw_nodes.copy(),
        raw_edges.copy(),
        fix_topology=True,
        dead_ends=True,
        nodes_to_keep_regardless=pseudo + tips,
    )

    assert set(pseudo + tips) <= set(nodes["nodeID"])


@pytest.mark.slow
def test_a_wider_tolerance_collapses_the_two_carriageways_of_a_parkway():
    # Murray Ross Parkway is mapped as two carriageways about 9 m apart between the same two
    # junctions: different streets at the default 5 m, one street at 10 m.
    raw_nodes, raw_edges = york_raw_network(whole_town=True)

    def parkway_pairs(tolerance):
        _, edges = ci.clean_network(
            raw_nodes.copy(), raw_edges.copy(), same_vertexes_tolerance=tolerance
        )
        parkway = edges[edges["name"] == "Murray Ross Parkway"]
        pairs = Counter(frozenset(p) for p in zip(parkway["u"], parkway["v"], strict=True))
        return sum(count > 1 for count in pairs.values())

    assert parkway_pairs(5.0) >= 1
    assert parkway_pairs(10.0) == 0
