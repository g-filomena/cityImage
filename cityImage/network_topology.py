"""Network topology preparation utilities.

This module is the hard replacement for the old split modules:

* ``graph_clean.py``
* ``graph_consolidate.py``
* ``graph_topology.py``

The purpose is to keep all legacy graph-preparation behaviour in one explicit
boundary module, while the core cityImage semantics rely on already-prepared
``nodes_gdf``/``edges_gdf`` inputs.

No live OSM/file loading is owned here. For new work, prefer external network
preparation tools first, then pass cleaned GeoDataFrames into cityImage. These
helpers are retained for workflows that need the historical cityImage topology
operations.
"""

from __future__ import annotations

from collections import defaultdict

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely import STRtree
from shapely.geometry import LineString, Point

from .data_utils import convert_numeric_columns
from .geometry import split_line_at_MultiPoint
from .graph import _is_missing_scalar, graph_fromGDF, nodes_degree
from .network import join_nodes_edges_by_coordinates, obtain_nodes_gdf

pd.set_option("display.precision", 3)


# -----------------------------------------------------------------------------
# Topology fixing
# -----------------------------------------------------------------------------
def fix_network_topology(nodes_gdf, edges_gdf):
    """
    Node the network at shared, un-noded vertices.

    An edge is split at one of its own internal vertices only when that vertex **coincides with a
    vertex** (endpoint or internal) of another edge: two edges genuinely meet there but the junction
    was never noded. Crossings that do **not** share a vertex are left intact — this covers
    grade-separated bridges/tunnels and ways that merely cross in 2D, which must not be noded, as
    well as an edge whose vertex happens to fall on another edge's interior (the split would only be
    undone by the pseudo-node simplification anyway). No new vertices are ever introduced.

    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.

    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The (possibly unchanged) nodes and the updated edges.
    """
    edges_gdf = edges_gdf.copy()
    coords_list = [list(geometry.coords) for geometry in edges_gdf.geometry]
    edges_gdf["coords"] = coords_list

    def _coord_key(coord, ndigits=10):
        return (round(float(coord[0]), ndigits), round(float(coord[1]), ndigits))

    # coordinate -> set of edge positions carrying it as any vertex (endpoint or internal)
    vertex_edges = defaultdict(set)
    for pos, coords in enumerate(coords_list):
        for coord in coords:
            vertex_edges[_coord_key(coord)].add(pos)

    # An internal vertex that is also a vertex of a *different* edge is a shared, un-noded junction
    # -> a split point. Endpoints are already u/v nodes, so only internal vertices are considered.
    to_fix_points = []
    for pos, coords in enumerate(coords_list):
        endpoints = {_coord_key(coords[0]), _coord_key(coords[-1])}
        points, seen = [], set()
        for coord in coords[1:-1]:
            key = _coord_key(coord)
            if key in endpoints or key in seen:
                continue
            if vertex_edges[key] - {pos}:  # the vertex belongs to another edge too
                points.append(Point(coord[0], coord[1]))
                seen.add(key)
        to_fix_points.append(points)

    edges_gdf["to_fix"] = to_fix_points
    edges_gdf["fixing"] = [len(item) > 0 for item in to_fix_points]

    to_fix = edges_gdf[edges_gdf["fixing"]].copy()
    edges_gdf = edges_gdf[~edges_gdf["fixing"]]
    if len(to_fix) == 0:
        # Nothing to split: drop temp columns and return the unchanged nodes alongside the edges,
        # matching the (nodes_gdf, edges_gdf) contract callers unpack.
        edges_gdf = edges_gdf.drop(columns=["coords", "to_fix", "fixing"], errors="ignore")
        return nodes_gdf, edges_gdf
    return _add_fixed_edges(edges_gdf, to_fix)


def fix_fake_self_loops(nodes_gdf, edges_gdf):
    """
    Fix the network topology by removing (fake) self-loops and adding fixed edges.

    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.

    Returns
    -------
    LineString GeoDataFrame
        The updated edges GeoDataFrame.
    """

    edges_gdf = edges_gdf.copy()
    edges_gdf["coords"] = [list(geometry.coords) for geometry in edges_gdf.geometry]
    # all the coordinates but the from and to vertices' ones.
    edges_gdf["coords"] = [coords[1:-1] for coords in edges_gdf.coords]

    # convert nodes_gdf['x'] and nodes_gdf['y'] to numpy arrays for faster computation
    x = list(nodes_gdf["x"])
    y = list(nodes_gdf["y"])
    # create a set of all coordinates in nodes. This essentially correspond to the from and to nodes of the edges currently in the edges_gdf
    nodes_set = set(zip(x, y, strict=False))

    to_fix = []
    # loop through the coordinates in edges_gdf.coords and check if they are in the nodes_set. This means that one of the edges coords (not from and to),
    # coincide with some other edge from or to vertex (indicating some sort of loop)
    for coords in edges_gdf.coords:
        fix_coords = []
        for coord in coords:
            if coord in nodes_set:
                fix_coords.append(coord)
        to_fix.append(fix_coords)

    # assign the results to self_loops['to_fix']
    edges_gdf["to_fix"] = to_fix
    edges_gdf["fixing"] = [len(to_fix) > 0 for to_fix in edges_gdf["to_fix"]]
    to_fix = edges_gdf[edges_gdf["fixing"]].copy()
    edges_gdf = edges_gdf[~edges_gdf["fixing"]]
    if len(to_fix) == 0:
        return nodes_gdf, edges_gdf
    return _add_fixed_edges(edges_gdf, to_fix)


def _add_fixed_edges(edges_gdf, to_fix_gdf):
    """
    Add fixed edges to the edges GeoDataFrame.

    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    to_fix_gdf: GeoDataFrame
        The GeoDataFrame containing the edges to be fixed.

    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The cleaned junctions and street segments GeoDataFrames.
    """
    dfs = []

    def _split_row_geometry(row):
        split_points = [point if isinstance(point, Point) else Point(point) for point in row.to_fix]
        return split_line_at_MultiPoint(row.geometry, split_points, z=None)

    new_geometries = to_fix_gdf.apply(_split_row_geometry, axis=1)
    new_geometries = pd.DataFrame(new_geometries, columns=["lines"])

    def append_new_geometries(row):
        for n, line in enumerate(row):  # assigning the resulting geometries
            ix = row.name
            index = ix if n == 0 else max(edges_gdf.index) + 1

            # copy attributes
            row = to_fix_gdf.loc[ix].copy()
            # and assign geometry an new edgeID
            row["edgeID"] = index
            row["geometry"] = line
            dfs.append(row.to_frame().T)

    new_geometries.apply(lambda row: append_new_geometries(row), axis=1)
    rows = pd.concat(dfs, ignore_index=True)
    rows = rows.explode(column="geometry")

    # concatenate the dataframes and assign to edges_gdf
    edges_gdf = pd.concat([edges_gdf, rows], ignore_index=True)
    edges_gdf.drop(["u", "v", "to_fix", "fixing", "coords"], inplace=True, axis=1)
    edges_gdf["length"] = edges_gdf.geometry.length
    edges_gdf["edgeID"] = edges_gdf.index
    nodes_gdf = obtain_nodes_gdf(edges_gdf, edges_gdf.crs)
    nodes_gdf, edges_gdf = join_nodes_edges_by_coordinates(nodes_gdf, edges_gdf)

    return nodes_gdf, edges_gdf


def remove_disconnected_islands(nodes_gdf, edges_gdf):
    """
    Remove disconnected islands from a graph.

    Parameters:
    -----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.

    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The updated junctions and street segments GeoDataFrame.
    """
    Ng = graph_fromGDF(nodes_gdf, edges_gdf)
    if not nx.is_connected(Ng):
        largest_component = max(nx.connected_components(Ng), key=len)
        # Create a subgraph of Ng consisting only of this component:
        G = Ng.subgraph(largest_component)
        to_keep = list(G.nodes())
        nodes_gdf = nodes_gdf[nodes_gdf["nodeID"].isin(to_keep)]
        edges_gdf = edges_gdf[
            (edges_gdf.u.isin(nodes_gdf["nodeID"])) & (edges_gdf.v.isin(nodes_gdf["nodeID"]))
        ]

    return nodes_gdf, edges_gdf


# -----------------------------------------------------------------------------
# Network cleaning
# -----------------------------------------------------------------------------
def clean_network(
    nodes_gdf,
    edges_gdf,
    dead_ends=False,
    remove_islands=True,
    same_vertexes_edges=True,
    self_loops=False,
    fix_topology=False,
    preserve_direction=False,
    nodes_to_keep_regardless=None,
    same_vertexes_tolerance=5.0,
):
    """
    Cleans a street network by applying a series of topology and geometry corrections to nodes and edges GeoDataFrames.


    This function can:
        - Remove pseudo-nodes
        - Remove duplicate nodes and edges (by geometry or node pairing)
        - Remove disconnected islands (optional)
        - Remove edges with the same vertexes but different geometry (optional)
        - Remove dead-ends (optional)
        - Remove self-loops (optional)
        - Fix topology by breaking lines at intersections (optional)

    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        Point GeoDataFrame containing network nodes (junctions), must include a unique node ID column.
    edges_gdf : GeoDataFrame
        LineString GeoDataFrame containing street segments, must include columns for start/end node IDs and geometry.
    dead_ends : bool, optional
        If True, removes dead-end nodes and corresponding edges. Default is False.
    remove_islands : bool, optional
        If True, removes disconnected components ("islands") in the network. Default is True.
    same_vertexes_edges : bool, optional
        If True, resolves multiple edges between the same pair of nodes: a street mapped more
        than once (within 10% in length and `same_vertexes_tolerance` in distance) is reduced to
        its most central edge, and different streets are kept as parallel edges (see
        clean_same_vertexes_edges). Default is True.
    same_vertexes_tolerance : float, optional
        Largest distance, in CRS units, between two edges joining the same nodes for them to be
        taken as one street mapped twice (see clean_same_vertexes_edges). Default is 5.
    self_loops : bool, optional
        If True, removes self-loop edges (where start and end node are the same). Default is False.
    fix_topology : bool, optional
        If True, breaks lines at intersections with other lines in the streets GeoDataFrame. Default is False.
    preserve_direction : bool, optional
        If True, considers edge direction: edges with the same coordinates but opposite directions are not considered duplicates.
        If False, such edges are treated as duplicates. Default is False.
    nodes_to_keep_regardless : list, optional
        List of node IDs to always keep, even if they would otherwise be removed (e.g. for transport stations). Default is empty list.

    Returns
    -------
    nodes_gdf : GeoDataFrame
        Cleaned nodes GeoDataFrame.
    edges_gdf : GeoDataFrame
        Cleaned edges GeoDataFrame.
    """

    if nodes_to_keep_regardless is None:
        nodes_to_keep_regardless = []

    crs = nodes_gdf.crs
    nodes_gdf, edges_gdf = _prepare_dataframes(nodes_gdf, edges_gdf)
    # removes fake self-loops wrongly coded by the data source
    nodes_gdf, edges_gdf = fix_fake_self_loops(nodes_gdf, edges_gdf)

    # Topology first: ways that cross at a shared vertex without either ending there are not joined
    # until it is fixed, so a dead-end or island test before it sees streets as cut off that are not.
    if fix_topology:
        nodes_gdf, edges_gdf = fix_network_topology(nodes_gdf, edges_gdf)
    if dead_ends:
        nodes_gdf, edges_gdf = fix_dead_ends(nodes_gdf, edges_gdf, nodes_to_keep_regardless)
    if remove_islands:
        nodes_gdf, edges_gdf = remove_disconnected_islands(nodes_gdf, edges_gdf)

    cycle = 0
    while (
        (
            same_vertexes_edges
            and not _are_edges_simplified(edges_gdf, preserve_direction, same_vertexes_tolerance)
        )
        | (not _are_nodes_simplified(nodes_gdf, edges_gdf, nodes_to_keep_regardless))
        | (cycle == 0)
    ):
        edges_gdf["length"] = edges_gdf[
            "geometry"
        ].length  # recomputing length, to account for small changes
        cycle += 1

        nodes_gdf, edges_gdf = clean_duplicate_nodes(nodes_gdf, edges_gdf)
        # eliminate loops
        if self_loops:
            edges_gdf = edges_gdf[edges_gdf["u"] != edges_gdf["v"]]
        if dead_ends:
            nodes_gdf, edges_gdf = fix_dead_ends(nodes_gdf, edges_gdf, nodes_to_keep_regardless)

        nodes_gdf, edges_gdf = clean_duplicate_edges(nodes_gdf, edges_gdf, preserve_direction)

        # edges with different geometries but same u-v nodes pairs
        if same_vertexes_edges:
            nodes_gdf, edges_gdf = clean_same_vertexes_edges(
                nodes_gdf, edges_gdf, preserve_direction, same_vertexes_tolerance
            )

        # simplify the graph
        nodes_gdf, edges_gdf = simplify_graph(nodes_gdf, edges_gdf, nodes_to_keep_regardless)

        # repreat eliminate loops
        if self_loops:
            edges_gdf = edges_gdf[edges_gdf["u"] != edges_gdf["v"]]
        if dead_ends:
            nodes_gdf, edges_gdf = fix_dead_ends(nodes_gdf, edges_gdf, nodes_to_keep_regardless)

    # No second island removal here: the loop (node/edge de-duplication, dead-end and pseudo-node
    # removal) can only merge or peel, never split a component, so any islands were already removed
    # before the loop.
    nodes_gdf["x"], nodes_gdf["y"] = list(
        zip(*[(r.coords[0][0], r.coords[0][1]) for r in nodes_gdf.geometry], strict=False)
    )
    edges_gdf = correct_edge_geometries(nodes_gdf, edges_gdf)  # correct edges coordinates
    return _finalize_dataframes(nodes_gdf, edges_gdf, crs)


def _prepare_dataframes(nodes_gdf, edges_gdf):
    """
    Prepare nodes and edges dataframes for further analysis by extracting the x,y coordinates of the nodes
    and adding new columns to the edges dataframe.

    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    crs : str, or pyproj.CRS
        Coordinate Reference System for the output GeoDataFrames. Can be a string (e.g. 'EPSG:32633'), or a pyproj.CRS object.

    Returns:
    ----------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The cleaned junctions and street segments GeoDataFrames.
    """

    nodes_gdf = nodes_gdf.copy().set_index("nodeID", drop=False)
    edges_gdf = edges_gdf.copy().set_index("edgeID", drop=False)

    nodes_gdf.index.name, edges_gdf.index.name = None, None
    nodes_gdf["x"], nodes_gdf["y"] = nodes_gdf.geometry.x, nodes_gdf.geometry.y
    edges_gdf.sort_index(inplace=True)

    if "highway" in edges_gdf.columns:
        edges_gdf = edges_gdf[edges_gdf["highway"] != "elevator"]

    return nodes_gdf, edges_gdf


def _finalize_dataframes(nodes_gdf, edges_gdf, crs):
    """
    Final steps to output clean dataframes.

    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    crs : str, or pyproj.CRS
        Coordinate Reference System for the output GeoDataFrames. Can be a string (e.g. 'EPSG:32633'), or a pyproj.CRS object.

    Returns:
    ----------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The cleaned junctions and street segments GeoDataFrames.
    """

    nodes_gdf.drop(["wkt"], axis=1, inplace=True, errors="ignore")  # remove temporary columns
    edges_gdf.drop(
        ["coords", "tmp", "code", "wkt", "fixing", "to_fix"], axis=1, inplace=True, errors="ignore"
    )  # remove temporary columns
    edges_gdf["length"] = edges_gdf["geometry"].length
    edges_gdf.set_index("edgeID", drop=False, inplace=True, append=False)
    nodes_gdf.set_index("nodeID", drop=False, inplace=True, append=False)
    nodes_gdf.index.name = None
    edges_gdf.index.name = None
    nodes_gdf = convert_numeric_columns(nodes_gdf)
    edges_gdf = convert_numeric_columns(edges_gdf)
    nodes_gdf.set_crs(crs, inplace=True)
    edges_gdf.set_crs(crs, inplace=True)
    return nodes_gdf, edges_gdf


def _are_nodes_simplified(nodes_gdf, edges_gdf, nodes_to_keep_regardless=None):
    """

    The function checks the presence of pseudo-junctions, by using the edges_gdf GeoDataFrame.

    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.

    Returns
    -------
    bool
        Whether the nodes of the network are simplified or not.
    """

    if nodes_to_keep_regardless is None:
        nodes_to_keep_regardless = []

    degree = nodes_degree(edges_gdf)
    to_edit = [node for node, deg in degree.items() if deg == 2]

    # Exclude nodes to keep regardless
    if nodes_to_keep_regardless:
        to_edit = [node for node in to_edit if node not in nodes_to_keep_regardless]
    if not to_edit:
        return True

    # A pseudo-node whose two segments both lead to the same node is the far end of a loop street:
    # merging it would turn the street into a self-loop, so it stays (see simplify_graph).
    neighbours = defaultdict(list)
    for u, v in zip(edges_gdf["u"], edges_gdf["v"], strict=False):
        neighbours[u].append(v)
        neighbours[v].append(u)
    for node in to_edit:
        a, b = neighbours[node]
        if a != b:
            return False
    return True


def _are_edges_simplified(edges_gdf, preserve_direction, same_vertexes_tolerance=5.0):
    """

    The function checks whether any street is mapped more than once between the same two nodes,
    by the rule of clean_same_vertexes_edges. Parallel edges that are different streets are
    simplified.

    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    preserve_direction: bool
        Whether edges (u, v) and (v, u) are distinct.
    same_vertexes_tolerance: float
        See clean_same_vertexes_edges.

    Returns
    -------
    simplified: bool
        Whether the edges of the network are simplified or not.
    """

    edges_gdf = edges_gdf.copy()
    edges_gdf["code"] = _pair_codes(edges_gdf, preserve_direction)
    return not _same_streets(edges_gdf, same_vertexes_tolerance)


def clean_duplicate_nodes(nodes_gdf, edges_gdf):
    """
    Removes duplicate nodes in a network based on coincident geometry, updating both nodes and edges GeoDataFrames.

    Nodes with exactly matching geometries are considered duplicates and merged into a single node.
    All references to duplicate node IDs in the edges GeoDataFrame are updated to the retained node ID.

    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        Point GeoDataFrame containing network nodes (junctions).
    edges_gdf : GeoDataFrame
        LineString GeoDataFrame containing street segments.

    Returns
    -------
    nodes_gdf : GeoDataFrame
        Cleaned nodes GeoDataFrame with duplicates removed.
    edges_gdf : GeoDataFrame
        Edges GeoDataFrame with references to duplicate node IDs updated.
    """

    nodes_gdf = nodes_gdf.copy().set_index("nodeID", drop=False)
    nodes_gdf.index.name = None

    # detecting duplicate geometries
    nodes_gdf["wkt"] = nodes_gdf["geometry"].apply(lambda geom: geom.wkt)
    # Detect duplicates
    subset_cols = ["wkt", "z"] if "z" in nodes_gdf.columns else ["wkt"]
    new_nodes = nodes_gdf.drop_duplicates(subset=subset_cols).copy()

    # assign univocal nodeID to edges which have 'u' or 'v' referring to duplicate nodes
    # Identify duplicate nodes
    to_edit = set(nodes_gdf.index) - set(new_nodes.index)

    if not to_edit:
        return nodes_gdf.drop(columns="wkt"), edges_gdf  # No changes needed

    # Map duplicates to their new nodeIDs
    node_mapping = {
        old_node: new_nodes[new_nodes["geometry"] == nodes_gdf.loc[old_node, "geometry"]].index[0]
        for old_node in to_edit
    }

    # readjusting edges' nodes too, accordingly
    edges_gdf[["u", "v"]] = edges_gdf[["u", "v"]].replace(node_mapping)

    return new_nodes.drop(columns="wkt"), edges_gdf


def simplify_graph(
    nodes_gdf,
    edges_gdf,
    nodes_to_keep_regardless=None,
):
    """

    The function identify pseudo-nodes, namely nodes that represent intersection between only 2 segments.
    The segments geometries are merged and the node is removed from the nodes_gdf GeoDataFrame.
    The merged segment may join two nodes already joined by another segment: parallel segments
    are kept (see clean_same_vertexes_edges). A pseudo-node whose two segments both lead to the
    same node, the far end of a loop street, is kept, since merging would leave a self-loop. Each
    attribute of a merged segment takes the non-null value covering the greatest length among the
    segments merged into it.

    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    nodes_to_keep_regardless: list
        List of nodeIDs representing nodes to keep, even when pseudo-nodes (e.g. stations, when modelling transport networks).

    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The cleaned junctions and street segments GeoDataFrames.
    """
    if nodes_to_keep_regardless is None:
        nodes_to_keep_regardless = []

    nodes_gdf = nodes_gdf.copy()
    edges_gdf = edges_gdf.copy()
    to_edit = list(set(n for n, d in nodes_degree(edges_gdf).items() if d == 2))

    if len(to_edit) == 0:
        return (nodes_gdf, edges_gdf)

    if nodes_to_keep_regardless:
        to_edit_list = list(to_edit)
        tmp_nodes = nodes_gdf[
            (nodes_gdf["nodeID"].isin(to_edit_list))
            & (~nodes_gdf["nodeID"].isin(nodes_to_keep_regardless))
        ].copy()
        to_edit = list(tmp_nodes["nodeID"])

    # Mutable edge state and a node->edges incidence map, so each pseudo-node is resolved with O(1)
    # dict lookups instead of the old per-node full-frame scan and per-merge .drop (both O(edges),
    # making the whole pass O(pseudo-nodes x edges)). The merge rules — which two incident edges
    # (lowest frame position first) and the orientation of the merged line — are unchanged.
    u_of = edges_gdf["u"].to_dict()
    v_of = edges_gdf["v"].to_dict()
    geom_of = edges_gdf["geometry"].to_dict()
    order = {eid: pos for pos, eid in enumerate(edges_gdf.index)}

    incidence = defaultdict(set)
    for eid in edges_gdf.index:
        incidence[u_of[eid]].add(eid)
        incidence[v_of[eid]].add(eid)

    dropped_edges: set = set()
    dropped_nodes: set = set()
    pieces = {eid: [eid] for eid in edges_gdf.index}

    def _coord_key(coord, ndigits=10):
        return tuple(round(float(value), ndigits) for value in coord[:2])

    for nodeID in to_edit:
        incident = sorted(
            (e for e in incidence[nodeID] if e not in dropped_edges), key=order.__getitem__
        )
        if len(incident) == 0:
            dropped_nodes.add(nodeID)
            continue
        if len(incident) == 1:
            continue  # possible dead end

        first, second = incident[0], incident[1]
        u1, v1, u2, v2 = u_of[first], v_of[first], u_of[second], v_of[second]
        coords_first, coords_second = list(geom_of[first].coords), list(geom_of[second].coords)

        if u1 == u2:  # meeting at u
            new_u, new_v = v1, v2
            line_a, line_b = coords_first[::-1], coords_second
        elif u1 == v2:  # meeting at u and v
            new_u, new_v = u2, v1
            line_a, line_b = coords_second, coords_first
        elif v1 == u2:  # meeting at v and u
            new_u, new_v = u1, v2
            line_a, line_b = coords_first, coords_second
        else:  # meeting at v and v
            new_u, new_v = u1, u2
            line_a, line_b = coords_first, coords_second[::-1]

        if new_u == new_v:
            continue  # the far end of a loop street: merging would leave a self-loop

        # detach both edges from their endpoints and remove the pseudo-node and second segment
        incidence[u1].discard(first)
        incidence[v1].discard(first)
        incidence[u2].discard(second)
        incidence[v2].discard(second)
        dropped_edges.add(second)
        dropped_nodes.add(nodeID)

        if _coord_key(line_a[-1]) == _coord_key(line_b[0]):
            merged_line = line_a + line_b[1:]
        else:
            merged_line = line_a + line_b

        u_of[first], v_of[first] = new_u, new_v
        geom_of[first] = LineString(merged_line)
        incidence[new_u].add(first)
        incidence[new_v].add(first)
        pieces[first] += pieces.pop(second)

    original = edges_gdf
    surviving = [eid for eid in edges_gdf.index if eid not in dropped_edges]
    edges_gdf = edges_gdf.loc[surviving].copy()
    edges_gdf["u"] = edges_gdf.index.map(u_of)
    edges_gdf["v"] = edges_gdf.index.map(v_of)
    edges_gdf["geometry"] = edges_gdf.index.map(geom_of)
    merged = {eid: pieces[eid] for eid in surviving if len(pieces[eid]) > 1}
    if merged:
        edges_gdf = _merge_attributes(edges_gdf, original, merged)
    edges_gdf = edges_gdf[edges_gdf["u"] != edges_gdf["v"]]  # eliminate node-lines

    if dropped_nodes:
        nodes_gdf = nodes_gdf.drop(
            index=[n for n in dropped_nodes if n in nodes_gdf.index], errors="ignore"
        )

    return nodes_gdf, edges_gdf


_STRUCTURAL_EDGE_COLUMNS = frozenset(
    {"edgeID", "u", "v", "geometry", "length", "coords", "code", "tmp", "wkt", "fixing", "to_fix"}
)


def _merge_attributes(edges_gdf, original, merged):
    """Give each merged edge, per attribute, the non-null value covering the most length.

    ``merged`` maps a surviving edgeID to the edgeIDs of ``original`` merged into it. Ties go to
    the value met first along that list. Only the merged rows are written, so the other rows and
    each column's dtype are left as they are.
    """
    lengths = original.geometry.length
    long = pd.DataFrame(
        [(target, piece) for target, group in merged.items() for piece in group],
        columns=["_target", "_piece"],
    )
    long["_w"] = lengths.reindex(long["_piece"]).to_numpy()
    edges_gdf = edges_gdf.copy()
    for column in original.columns:
        if column in _STRUCTURAL_EDGE_COLUMNS or column == original.geometry.name:
            continue
        values = original[column].reindex(long["_piece"]).to_numpy(dtype=object)
        present = np.array([not _is_missing_scalar(value) for value in values], dtype=bool)
        frame = long[present].assign(_v=values[present])
        if frame.empty:
            continue  # no merged edge has a value; each already carries a missing one
        frame["_k"] = [repr(value) for value in frame["_v"]]
        weight = frame.groupby(["_target", "_k"], sort=False)["_w"].sum()
        best = weight.groupby(level=0, sort=False).idxmax()
        first_value = frame.drop_duplicates(["_target", "_k"]).set_index(["_target", "_k"])["_v"]
        targets = list(best.index)
        edges_gdf.loc[targets, column] = pd.Series(
            [first_value.loc[key] for key in best], index=targets
        )
    return edges_gdf


def fix_dead_ends(nodes_gdf, edges_gdf, nodes_to_keep_regardless=None):
    """

    The function removes dead-ends: nodes from where only one segment originates, and that segment.
    It repeats until none is left, so a street ending in a dead end is removed back to the junction
    where it meets the rest of the network. Removal stops at a node in `nodes_to_keep_regardless`,
    which is kept together with the segments leading from it back to that junction. A component
    with no loop has no such junction and is left as it is. Nodes no segment references are dropped.

    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame
    nodes_to_keep_regardless: list
        List of nodeIDs never removed as dead ends (e.g. stations, when modelling transport
        networks).

    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The cleaned junctions and street segments GeoDataFrames.
    """
    keep = set(nodes_to_keep_regardless or [])
    edge_ids = edges_gdf.index.to_list()
    us, vs = edges_gdf["u"].to_list(), edges_gdf["v"].to_list()

    # Peel dead ends from a queue, updating degrees as each segment goes: O(edges) in all.
    degree = defaultdict(int)
    incident = defaultdict(list)
    for pos, (u, v) in enumerate(zip(us, vs, strict=True)):
        degree[u] += 1
        degree[v] += 1
        incident[u].append(pos)
        incident[v].append(pos)

    removed = [False] * len(edge_ids)
    queue = [node for node, deg in degree.items() if deg == 1 and node not in keep]
    while queue:
        node = queue.pop()
        if degree[node] != 1:
            continue
        pos = next(p for p in incident[node] if not removed[p])
        removed[pos] = True
        other = vs[pos] if us[pos] == node else us[pos]
        degree[node] -= 1
        degree[other] -= 1
        if degree[other] == 1 and other not in keep:
            queue.append(other)

    # A component whose segments all went is loop-free: restore it rather than delete it.
    if any(removed):
        component = nx.Graph()
        component.add_edges_from(zip(us, vs, strict=True))
        survivors = {us[pos] for pos, gone in enumerate(removed) if not gone}
        for nodes in nx.connected_components(component):
            if nodes.isdisjoint(survivors):
                for pos in (p for node in nodes for p in incident[node]):
                    removed[pos] = False

    edges_gdf = edges_gdf[[not gone for gone in removed]].copy()
    return _drop_unused_nodes(nodes_gdf.copy(), edges_gdf), edges_gdf


def _drop_unused_nodes(nodes_gdf, edges_gdf):
    """Keep only the nodes that some edge references as its u or v."""
    used = set(edges_gdf["u"]).union(edges_gdf["v"])
    return nodes_gdf[nodes_gdf["nodeID"].isin(used)]


def clean_same_vertexes_edges(
    nodes_gdf, edges_gdf, preserve_direction=False, same_vertexes_tolerance=5.0
):
    """
    Resolves edges that share the same start and end nodes (same vertexes).

    Edges joining the same pair of nodes are either one street mapped more than once, or different
    streets between the same two junctions (a crescent, a loop round a block). Two edges are taken
    as the same street when the longer is at most 10% longer than the shorter and they lie within
    `same_vertexes_tolerance` of each other (Hausdorff distance). A street is every edge linked to
    another through a chain of such matches, so the grouping does not depend on edge order.

    Of each street mapped more than once, only its most central edge is kept: the one with the
    smallest total distance to the others (of two, the shorter). Different streets are all kept,
    as parallel edges between the same nodes; no node is added.

    Self-loops (u equal to v) are left as they are: removing them is up to clean_duplicate_edges
    or the `self_loops` option of clean_network.

    If `preserve_direction` is False, treats edges as undirected (edges (u,v) and (v,u) are considered duplicates).
    If True, edges in opposite directions are not treated as duplicates.

    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        GeoDataFrame of nodes (junctions), must include unique node IDs.
    edges_gdf : GeoDataFrame
        GeoDataFrame of street segments (edges), must include 'u', 'v', 'geometry', and 'length' columns.
    preserve_direction : bool
        Whether to preserve edge direction (see above).
    same_vertexes_tolerance : float
        Largest distance, in CRS units, between two edges taken as the same street (default: 5).
        It keeps apart distinct streets of similar length, such as the two sides of a block.

    Returns
    -------
    nodes_gdf : GeoDataFrame
        Filtered nodes, only those referenced by remaining edges.
    edges_gdf : GeoDataFrame
        The edges, with each street mapped more than once reduced to one edge.
    """
    edges_gdf = edges_gdf.copy()
    edges_gdf["code"] = _pair_codes(edges_gdf, preserve_direction)
    edges_gdf["length"] = edges_gdf.geometry.length
    to_drop = [
        index
        for street in _same_streets(edges_gdf, same_vertexes_tolerance)
        for index in street[1:]
    ]
    if not to_drop:
        return nodes_gdf, edges_gdf
    edges_gdf = edges_gdf.drop(to_drop, axis=0)
    return _drop_unused_nodes(nodes_gdf, edges_gdf), edges_gdf


def _same_streets(edges_gdf, tolerance):
    """Groups of edges that are one street mapped more than once, the edge to keep first.

    Edges are compared only within the same ``code`` (node pair); self-loops are skipped. Two
    edges match when the longer is at most 10% longer than the shorter and their Hausdorff distance
    is within ``tolerance``; a group is a connected set of matches. Only groups of two or more are
    returned, each led by its most central edge (smallest total Hausdorff distance to the others,
    then the shortest, then the lowest edgeID).
    """
    candidates = edges_gdf[
        edges_gdf["code"].duplicated(keep=False) & (edges_gdf["u"] != edges_gdf["v"])
    ]
    streets = []
    for _, group in candidates.groupby("code", sort=True):
        ids = list(group.index)
        lengths = group.geometry.length.to_numpy()
        lines = list(group.geometry)
        distance = np.zeros((len(ids), len(ids)))
        matches = nx.Graph()
        matches.add_nodes_from(range(len(ids)))
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                distance[i, j] = distance[j, i] = lines[i].hausdorff_distance(lines[j])
                short, long = sorted((lengths[i], lengths[j]))
                if long <= short * 1.1 and distance[i, j] <= tolerance:
                    matches.add_edge(i, j)

        for component in nx.connected_components(matches):
            if len(component) < 2:
                continue
            group_members = sorted(component)
            order = sorted(
                group_members,
                key=lambda i: (distance[i, group_members].sum(), lengths[i], ids[i]),
            )
            streets.append([ids[i] for i in order])
    return streets


def _pair_codes(edges_gdf, preserve_direction):
    """A 'u-v' code per edge, with the lower nodeID first unless direction is preserved."""
    if preserve_direction:
        return edges_gdf["u"].astype(str) + "-" + edges_gdf["v"].astype(str)
    return np.where(
        edges_gdf["v"] >= edges_gdf["u"],
        edges_gdf["u"].astype(str) + "-" + edges_gdf["v"].astype(str),
        edges_gdf["v"].astype(str) + "-" + edges_gdf["u"].astype(str),
    )


def clean_duplicate_edges(
    nodes_gdf,
    edges_gdf,
    preserve_direction=False,
):
    """
    Cleans and deduplicates network edges, and removes unused nodes.


    The function performs the following:
      - Generates a unique 'code' for each edge, based on node IDs, with or without preserving direction.
      - Removes self-loop edges (edges from a node to itself).
      - Drops duplicate edges based on geometry (including reversal if direction is not preserved).
      - Removes edges that are geometrically duplicates, even if node order is reversed (for undirected graphs).
      - Updates the node GeoDataFrame to keep only those nodes actually used by the remaining edges.

    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        GeoDataFrame containing nodes, must include a 'nodeID' column.
    edges_gdf : GeoDataFrame
        GeoDataFrame containing edges, must include 'u', 'v', and 'geometry' columns.
    preserve_direction : bool, optional
        If True, edge direction is preserved; edges (u,v) and (v,u) are considered distinct.
        If False, edges are treated as undirected and geometric duplicates (with reversed coords) are removed.
        Default is False.

    Returns
    -------
    nodes_gdf : GeoDataFrame
        Filtered nodes GeoDataFrame, containing only nodes referenced by the cleaned edges.
    edges_gdf : GeoDataFrame
        Cleaned edges GeoDataFrame, deduplicated and without self-loops.
    """
    edges_gdf = edges_gdf.copy()
    edges_gdf["code"] = _pair_codes(edges_gdf, preserve_direction)

    # eliminate node-lines
    edges_gdf = edges_gdf[edges_gdf["u"] != edges_gdf["v"]]

    # dropping duplicate-geometries edges
    geometries = edges_gdf["geometry"].apply(lambda geom: geom.wkb)
    edges_gdf = edges_gdf.loc[geometries.drop_duplicates().index]

    # dropping edges with same geometry but with coords in different orders (depending on their directions)
    # Reordering coordinates to allow for comparison between edges
    edges_gdf["coords"] = [list(c.coords) for c in edges_gdf.geometry]
    if not preserve_direction:
        condition = (edges_gdf.u.astype(str) + "-" + edges_gdf.v.astype(str)) != edges_gdf.code
        edges_gdf.loc[condition, "coords"] = pd.Series(
            [x[::-1] for x in edges_gdf.loc[condition]["coords"]],
            index=edges_gdf.loc[condition].index,
        )

    edges_gdf["tmp"] = edges_gdf["coords"].apply(tuple)
    edges_gdf.drop_duplicates(["tmp"], keep="first", inplace=True)

    return _drop_unused_nodes(nodes_gdf, edges_gdf), edges_gdf


def correct_edge_geometries(nodes_gdf, edges_gdf):
    """

    The function adjusts the edges LineString coordinates consistently with their relative u and v nodes' coordinates.
    It might be necessary to run the function after having cleaned the network.

    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.

    Returns
    -------
    edges_gdf: LineString GeoDataFrame
        The updated street segments GeoDataFrame.
    """

    def _update_line_geometry_coords(u, v, nodes_gdf, line_geometry):
        """
        It supports the correct_edges function checks that the edges coordinates are consistent with their relative u and v nodes'coordinates.
        It can be necessary to run the function after having cleaned the network.
        """
        line_coords = list(line_geometry.coords)
        line_coords[0] = (nodes_gdf.loc[u]["x"], nodes_gdf.loc[u]["y"])
        line_coords[-1] = (nodes_gdf.loc[v]["x"], nodes_gdf.loc[v]["y"])
        new_line_geometry = LineString([coor for coor in line_coords])
        return new_line_geometry

    edges_gdf["geometry"] = edges_gdf.apply(
        lambda row: _update_line_geometry_coords(row["u"], row["v"], nodes_gdf, row["geometry"]),
        axis=1,
    )
    return edges_gdf


# -----------------------------------------------------------------------------
# Node/edge consolidation
# -----------------------------------------------------------------------------
def _clusters_within_tolerance(nodes_gdf, tolerance):
    """Cluster labels, one per node, such that all nodes sharing a label lie within `tolerance`
    of each other.

    Nodes are taken as seeds in decreasing order of how many neighbours they have within
    `tolerance` (ties broken by nodeID), so dense junction cores seed first. Each seed collects
    its unassigned neighbours, nearest first, admitting one only if it is within `tolerance` of
    every member already admitted. The result does not depend on row order.
    """
    geometries = nodes_gdf.geometry.to_numpy()
    xy = np.column_stack([nodes_gdf.geometry.x.to_numpy(), nodes_gdf.geometry.y.to_numpy()])
    node_ids = nodes_gdf["nodeID"].to_numpy()

    tree = STRtree(geometries)
    left, right = tree.query(geometries, predicate="dwithin", distance=tolerance)
    neighbours = defaultdict(list)
    for i, j in zip(left, right, strict=True):
        if i != j:
            neighbours[i].append(j)

    order = sorted(range(len(xy)), key=lambda i: (-len(neighbours[i]), node_ids[i]))
    labels = np.full(len(xy), -1, dtype=int)
    next_label = 0
    for seed in order:
        if labels[seed] >= 0:
            continue
        members = [seed]
        candidates = [j for j in neighbours[seed] if labels[j] < 0]
        candidates.sort(key=lambda j: (np.hypot(*(xy[j] - xy[seed])), node_ids[j]))
        for j in candidates:
            if np.all(np.hypot(*(xy[members] - xy[j]).T) <= tolerance):
                members.append(j)
        labels[members] = next_label
        next_label += 1
    return labels


def consolidate_nodes(
    nodes_gdf,
    edges_gdf,
    consolidate_edges_too=False,
    tolerance=20,
):
    """
    Consolidates nodes in a spatial network that are within a given distance (tolerance), preserving topology and unclustered nodes.

    Nodes are clustered so that every pair in a cluster lies within `tolerance` of each other,
    and each cluster is represented by a single consolidated node at the mean of its members.
    For clusters containing disconnected components, each connected component is further split into its own consolidated node.
    Optionally, edges can be updated to reference the new consolidated node IDs and geometries.

    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        GeoDataFrame of nodes, must include columns 'nodeID' and 'geometry'. If present, 'z' is averaged for clusters.
    edges_gdf : GeoDataFrame
        GeoDataFrame of edges for checking network connectivity.
    consolidate_edges_too : bool, optional
        If True, also returns the updated edges GeoDataFrame (default: False).
    tolerance : float, optional
        Distance threshold for clustering nodes (in CRS units): the largest distance between any
        two nodes merged into one (default: 20).

    Returns
    -------
    consolidated_nodes_gdf : GeoDataFrame
        GeoDataFrame of consolidated nodes. Columns include:
            - 'old_nodeIDs': list of merged node IDs
            - 'x', 'y': centroid coordinates
            - 'z' (optional): averaged elevation for the cluster
            - 'nodeID': new node ID
            - 'geometry': consolidated node Point geometry
    consolidated_edges_gdf : GeoDataFrame (optional)
        Only returned if `consolidate_edges_too` is True.
        Edges with endpoints mapped to new consolidated node IDs and geometries
    """

    nodes_gdf = nodes_gdf.copy().set_index("nodeID", drop=False)
    nodes_gdf.index.name = None
    nodes_gdf.drop(columns=["x", "y"], inplace=True, errors="ignore")
    graph = graph_fromGDF(nodes_gdf, edges_gdf)

    # Steps 1-2: Cluster nodes so that every pair in a cluster lies within tolerance
    new_column = "new_nodeID"
    labels = _clusters_within_tolerance(nodes_gdf, tolerance)
    gdf = pd.DataFrame(nodes_gdf.drop(columns="geometry"))
    gdf[new_column] = labels
    centroids = (
        pd.DataFrame(
            {"x": nodes_gdf.geometry.x.to_numpy(), "y": nodes_gdf.geometry.y.to_numpy()},
            index=nodes_gdf.index,
        )
        .groupby(labels)
        .mean()
    )
    gdf["x"] = gdf[new_column].map(centroids["x"])
    gdf["y"] = gdf[new_column].map(centroids["y"])
    new_nodeID = gdf[new_column].max() + 1

    # Step 3: Split non-connected components in clusters
    for _cluster_label, nodes_subset in gdf.groupby(new_column):
        if len(nodes_subset) > 1:  # Skip unclustered nodes
            wccs = list(nx.connected_components(graph.subgraph(nodes_subset.index)))
            if len(wccs) > 1:
                for wcc in wccs:
                    idx = list(wcc)
                    subcluster_centroid = nodes_gdf.loc[idx].geometry.unary_union.centroid
                    gdf.loc[idx, ["x", "y"]] = subcluster_centroid.x, subcluster_centroid.y
                    gdf.loc[idx, new_column] = new_nodeID
                    new_nodeID += 1

    # Step 4: Consolidate nodes, but preserve unclustered ones
    consolidated_nodes = []
    has_z = "z" in nodes_gdf.columns
    oldIDs_column = "old_nodeID"

    for new_nodeID, nodes_subset in gdf.groupby(new_column):
        old_nodeIDs = nodes_subset["nodeID"].to_list()
        cluster_x, cluster_y = nodes_subset.iloc[0][["x", "y"]]

        new_node = {
            oldIDs_column: old_nodeIDs,
            "x": cluster_x,
            "y": cluster_y,
            "nodeID": new_nodeID,
        }

        if has_z:
            new_node["z"] = (
                nodes_gdf.loc[old_nodeIDs, "z"].mean()
                if len(old_nodeIDs) > 1
                else nodes_gdf.loc[old_nodeIDs[0], "z"]
            )

        consolidated_nodes.append(new_node)

    # Convert list of dicts to DataFrame
    consolidated_nodes_df = pd.DataFrame(consolidated_nodes)

    # Create final GeoDataFrame
    consolidated_nodes_gdf = gpd.GeoDataFrame(
        consolidated_nodes_df,
        geometry=gpd.points_from_xy(
            consolidated_nodes_df["x"],
            consolidated_nodes_df["y"],
            consolidated_nodes_df["z"] if "z" in consolidated_nodes_df.columns else None,
        ),
        crs=nodes_gdf.crs,
    )

    if consolidate_edges_too:
        return consolidated_nodes_gdf, consolidate_edges(edges_gdf, consolidated_nodes_gdf)

    return consolidated_nodes_gdf


def consolidate_edges(edges_gdf, consolidated_nodes_gdf):
    """Consolidate edge geometries after node consolidation.

    Parameters
    ----------
    nodes_gdf : geopandas.GeoDataFrame
        cityImage node table.
    edges_gdf : geopandas.GeoDataFrame
        cityImage edge table.
    consolidation_map : dict
        Mapping from original node IDs to consolidated node IDs.

    Returns
    -------
    geopandas.GeoDataFrame
        Edge table with updated endpoints and geometries.

    Notes
    -----
    This helper preserves cityImage endpoint and identifier semantics while applying
    geometry consolidation. It is not a generic line-merge wrapper.
    """

    oldIDs_column = "old_nodeID"
    # Create a mapping from old_nodeIDs to their corresponding nodeID and geometry
    nodes_mapping = consolidated_nodes_gdf.explode(oldIDs_column)[
        [oldIDs_column, "geometry", "nodeID"]
    ].set_index(oldIDs_column)

    def _update_edge(row):

        old_u, old_v, geom = row["u"], row["v"], row["geometry"]

        # Map old_u and old_v to their corresponding new nodeIDs
        new_u_id = nodes_mapping.loc[old_u, "nodeID"]
        new_v_id = nodes_mapping.loc[old_v, "nodeID"]

        # Get the new geometries for u and v
        new_u_geom = nodes_mapping.loc[old_u, "geometry"]
        new_v_geom = nodes_mapping.loc[old_v, "geometry"]

        # Update the geometry (replace first and last coordinates), keeping the edge's own
        # dimensionality: consolidated nodes carry z when the input nodes had a 'z' column.
        if isinstance(geom, LineString):
            dims = 3 if geom.has_z else 2
            new_coords = (
                [new_u_geom.coords[0][:dims]]
                + list(geom.coords[1:-1])
                + [new_v_geom.coords[0][:dims]]
            )
            geom = LineString(new_coords)

        return pd.Series({"u": new_u_id, "v": new_v_id, "geometry": geom})

    # Apply updates to the edges
    consolidated_edges = edges_gdf.copy()
    consolidated_edges[["u", "v", "geometry"]] = consolidated_edges.apply(_update_edge, axis=1)
    consolidated_edges = consolidated_edges[consolidated_edges.u != consolidated_edges.v]
    consolidated_edges.index = consolidated_edges["edgeID"]
    consolidated_edges.index.name = None

    return consolidated_edges
