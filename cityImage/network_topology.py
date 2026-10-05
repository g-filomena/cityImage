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

from .angles import _round_coord as _coord_key
from .data_utils import convert_numeric_columns
from .geometry import center_line
from .graph import _is_missing_scalar, _oneway_flags, graph_fromGDF, nodes_degree

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

    Existing nodes and edges keep their IDs, attributes and dtypes: a split edge keeps its edgeID
    on its first piece, and a split point takes the node already there or, if there is none, a new
    node (see _split_edges).

    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.

    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        Copies of the nodes, with any new junction added, and of the updated edges.
    """
    coords_list = [list(geometry.coords) for geometry in edges_gdf.geometry]

    # coordinate -> set of edge positions carrying it as any vertex (endpoint or internal)
    vertex_edges = defaultdict(set)
    for pos, coords in enumerate(coords_list):
        for coord in coords:
            vertex_edges[_coord_key(coord)].add(pos)

    # An internal vertex that is also a vertex of a *different* edge is a shared, un-noded junction
    # -> a split point. Endpoints are already u/v nodes, so only internal vertices are considered.
    split_keys = []
    for pos, coords in enumerate(coords_list):
        endpoints = {_coord_key(coords[0]), _coord_key(coords[-1])}
        internal = {_coord_key(coord) for coord in coords[1:-1]} - endpoints
        split_keys.append({key for key in internal if vertex_edges[key] - {pos}})

    return _split_edges(nodes_gdf, edges_gdf, split_keys)


def fix_fake_self_loops(nodes_gdf, edges_gdf):
    """
    Split edges that run through an existing node without being split there.

    An internal vertex of an edge that lies on a node (another edge's end, or the edge's own start
    or end, as in a way that loops back through its first node) becomes a split point. A node's
    position is its point and the end coordinates of its edges, so float noise between the two
    does not hide it. Existing nodes and edges keep their IDs, attributes and dtypes (see
    _split_edges).

    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.

    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        Copies of the nodes and of the updated edges.
    """
    node_keys = {_coord_key(point.coords[0]) for point in nodes_gdf.geometry}
    for geometry in edges_gdf.geometry:
        node_keys.add(_coord_key(geometry.coords[0]))
        node_keys.add(_coord_key(geometry.coords[-1]))
    split_keys = [
        {_coord_key(coord) for coord in geometry.coords[1:-1]} & node_keys
        for geometry in edges_gdf.geometry
    ]
    return _split_edges(nodes_gdf, edges_gdf, split_keys)


def _split_at_vertices(line, keys):
    """Split a line at each internal vertex whose ``_coord_key`` is in ``keys``, keeping z.

    A piece that would have no length (a vertex repeated at a split point, or at the line's end)
    is left out: the pieces either side of it already meet there.
    """
    coords = list(line.coords)
    lines, current = [], [coords[0]]
    start, moved = _coord_key(coords[0]), False
    for coord in coords[1:-1]:
        current.append(coord)
        key = _coord_key(coord)
        moved = moved or key != start
        if key in keys:
            if moved:
                lines.append(LineString(current))
            current, start, moved = [coord], key, False
    current.append(coords[-1])
    if moved or _coord_key(coords[-1]) != start or not lines:
        lines.append(LineString(current))
    return lines


def _oriented_ends(nodes_gdf, edges_gdf):
    """Per edge, the nodes at the first and at the last coordinate of its geometry.

    That is ``(u, v)``, or ``(v, u)`` for a line stored against its labels: whichever pairing puts
    the labelled nodes nearer to the line's ends.
    """
    xy = {
        node_id: point.coords[0]
        for node_id, point in zip(nodes_gdf["nodeID"], nodes_gdf.geometry, strict=True)
    }

    def _d2(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    ends = []
    for u, v, line in zip(edges_gdf["u"], edges_gdf["v"], edges_gdf.geometry, strict=True):
        first, last = line.coords[0], line.coords[-1]
        reversed_ = (
            u != v
            and u in xy
            and v in xy
            and _d2(first, xy[v]) + _d2(last, xy[u]) < _d2(first, xy[u]) + _d2(last, xy[v])
        )
        ends.append((v, u) if reversed_ else (u, v))
    return ends


def _split_edges(nodes_gdf, edges_gdf, split_keys):
    """Split edges at internal vertices, keeping every existing nodeID, edgeID, attribute and dtype.

    ``split_keys`` holds, per edge row, the ``_coord_key`` of the vertices to split that edge at,
    wherever the edge passes them. The pieces of a split edge are copies of its row, oriented by
    its geometry: the first starts at the node at the line's first coordinate and keeps the edgeID,
    the last ends at the node at its last coordinate, and the others get new edgeIDs after the
    largest. A split point takes the node at its coordinates (a node's point, or an edge end
    labelled with it) or, if there is none, a new node after the largest nodeID, shared by every
    edge split there. A new node's ``z`` is the vertex's own, or else interpolated along the edge
    between its end nodes; its other columns are left missing (integer and boolean columns become
    nullable to hold that).
    """
    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()
    if not any(split_keys):
        return nodes_gdf, edges_gdf

    ends = _oriented_ends(nodes_gdf, edges_gdf)
    node_at = {}
    for node_id, point in zip(nodes_gdf["nodeID"], nodes_gdf.geometry, strict=True):
        node_at.setdefault(_coord_key(point.coords[0]), node_id)
    for (start, end), line in zip(ends, edges_gdf.geometry, strict=True):
        node_at.setdefault(_coord_key(line.coords[0]), start)
        node_at.setdefault(_coord_key(line.coords[-1]), end)

    if "z" in nodes_gdf.columns:
        node_z = dict(zip(nodes_gdf["nodeID"], nodes_gdf["z"], strict=True))
    elif nodes_gdf.geometry.has_z.any():
        node_z = dict(zip(nodes_gdf["nodeID"], nodes_gdf.geometry.z, strict=True))
    else:
        node_z = {}
    node_dims = 3 if nodes_gdf.geometry.has_z.any() else 2
    next_node = nodes_gdf["nodeID"].max() + 1 if len(nodes_gdf) else 0
    next_edge = edges_gdf["edgeID"].max() + 1
    new_nodes = []

    def _node_for(coord, fraction, start, end):
        nonlocal next_node
        key = _coord_key(coord)
        if key in node_at:
            return node_at[key]
        z = coord[2] if len(coord) > 2 else None
        z_start, z_end = node_z.get(start), node_z.get(end)
        if z is None and not (_is_missing_scalar(z_start) or _is_missing_scalar(z_end)):
            z = z_start + fraction * (z_end - z_start)
        point = coord[:2] if z is None else (coord[0], coord[1], z)
        node = {"nodeID": next_node, "geometry": Point(point[:node_dims])}
        for column, value in zip(("x", "y", "z"), point, strict=False):
            if column in nodes_gdf.columns:
                node[column] = value
        new_nodes.append(node)
        node_at[key] = next_node
        next_node += 1
        return node_at[key]

    take, geometries, edge_ids, us, vs, order = [], [], [], [], [], []
    for pos, keys in enumerate(split_keys):
        if not keys:
            continue
        line, (start, end) = edges_gdf.geometry.iloc[pos], ends[pos]
        lines = _split_at_vertices(line, keys)
        travelled, piece_u = 0.0, start
        for n, piece in enumerate(lines):
            travelled += piece.length
            if n == len(lines) - 1:
                piece_v = end
            else:
                fraction = travelled / line.length if line.length else 0.0
                piece_v = _node_for(piece.coords[-1], fraction, start, end)
            if n == 0:
                edge_ids.append(edges_gdf["edgeID"].iloc[pos])
            else:
                edge_ids.append(next_edge)
                next_edge += 1
            take.append(pos)
            geometries.append(piece)
            us.append(piece_u)
            vs.append(piece_v)
            order.append((pos, n))
            piece_u = piece_v

    # Pieces are copies of their edge's row, so every column keeps its dtype.
    pieces = edges_gdf.iloc[take].reset_index(drop=True)
    pieces["edgeID"] = pd.Series(edge_ids).astype(edges_gdf["edgeID"].dtype)
    pieces["u"] = pd.Series(us).astype(edges_gdf["u"].dtype)
    pieces["v"] = pd.Series(vs).astype(edges_gdf["v"].dtype)
    pieces[edges_gdf.geometry.name] = gpd.GeoSeries(geometries, crs=edges_gdf.crs)

    # The pieces of a split edge take its place in the frame, so row order is kept.
    kept = [pos for pos, keys in enumerate(split_keys) if not keys]
    new_edges = pd.concat([edges_gdf.iloc[kept].reset_index(drop=True), pieces], ignore_index=True)
    sort_key = [(pos, 0) for pos in kept] + order
    new_edges = new_edges.iloc[sorted(range(len(sort_key)), key=sort_key.__getitem__)]
    new_edges["length"] = new_edges.geometry.length
    new_edges = _index_like(new_edges, edges_gdf, "edgeID")

    if new_nodes:
        added = gpd.GeoDataFrame(new_nodes, geometry="geometry", crs=nodes_gdf.crs)
        if nodes_gdf.geometry.name != "geometry":
            added = added.rename_geometry(nodes_gdf.geometry.name)
        combined = _keep_dtypes(pd.concat([nodes_gdf, added]), nodes_gdf)
        nodes_gdf = _index_like(combined, nodes_gdf, "nodeID")

    return nodes_gdf, new_edges


def _keep_dtypes(gdf, original):
    """Cast ``gdf``'s columns back to ``original``'s dtypes after rows with missing values were
    added: integer and boolean columns become their nullable counterparts where a value is missing.
    """
    for column, dtype in original.dtypes.items():
        if column == original.geometry.name or gdf[column].dtype == dtype:
            continue
        if gdf[column].isna().any():
            if pd.api.types.is_bool_dtype(dtype):
                dtype = "boolean"
            elif pd.api.types.is_integer_dtype(dtype) and not isinstance(
                dtype, pd.api.extensions.ExtensionDtype
            ):
                dtype = "Int64"
        gdf[column] = gdf[column].astype(dtype)
    return gdf


def _index_like(gdf, original, id_column):
    """Index ``gdf`` by its ID column if ``original`` is indexed by its IDs, else by position."""
    if original.index.equals(pd.Index(original[id_column])):
        gdf = gdf.set_index(id_column, drop=False)
        gdf.index.name = None
        return gdf
    return gdf.reset_index(drop=True)


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
    if Ng.number_of_nodes() == 0:
        return nodes_gdf, edges_gdf  # an empty network has no islands (and NetworkX raises)
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
    self_loops=True,
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
        If True, removes self-loop edges (where start and end node are the same). A loop street,
        which leaves a junction and returns to it, is one such edge once its pseudo-nodes are
        merged, however many nodes it was mapped with. False keeps it. Default is True.
    fix_topology : bool, optional
        If True, breaks lines at intersections with other lines in the streets GeoDataFrame. Default is False.
    preserve_direction : bool, optional
        If True, considers edge direction: edges with the same coordinates but opposite directions are not considered duplicates,
        and a pseudo-node where the direction changes is kept (see simplify_graph), so `oneway` stays exact.
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
        | (
            not _are_nodes_simplified(
                nodes_gdf, edges_gdf, nodes_to_keep_regardless, preserve_direction
            )
        )
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
            nodes_gdf = _drop_unused_nodes(nodes_gdf, edges_gdf)
        if dead_ends:
            nodes_gdf, edges_gdf = fix_dead_ends(nodes_gdf, edges_gdf, nodes_to_keep_regardless)

        nodes_gdf, edges_gdf = clean_duplicate_edges(
            nodes_gdf, edges_gdf, preserve_direction, self_loops=self_loops
        )

        # edges with different geometries but same u-v nodes pairs
        if same_vertexes_edges:
            nodes_gdf, edges_gdf = clean_same_vertexes_edges(
                nodes_gdf, edges_gdf, preserve_direction, same_vertexes_tolerance
            )

        # simplify the graph
        nodes_gdf, edges_gdf = simplify_graph(
            nodes_gdf, edges_gdf, nodes_to_keep_regardless, preserve_direction
        )

        # repreat eliminate loops
        if self_loops:
            edges_gdf = edges_gdf[edges_gdf["u"] != edges_gdf["v"]]
            nodes_gdf = _drop_unused_nodes(nodes_gdf, edges_gdf)
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
        ["coords", "tmp", "code", "wkt"], axis=1, inplace=True, errors="ignore"
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


def _are_nodes_simplified(
    nodes_gdf, edges_gdf, nodes_to_keep_regardless=None, preserve_direction=False
):
    """

    The function checks the presence of pseudo-junctions, by using the edges_gdf GeoDataFrame,
    by the rule of simplify_graph.

    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    preserve_direction: bool
        Whether a pseudo-node where the direction changes is kept (see simplify_graph).

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

    oneway_of = _oneway_of(edges_gdf) if preserve_direction else None
    incident = defaultdict(list)
    for eid, u, v in zip(edges_gdf.index, edges_gdf["u"], edges_gdf["v"], strict=True):
        incident[u].append((eid, True))
        incident[v].append((eid, False))

    def mergeable(node):
        (first, first_at_u), (second, second_at_u) = incident[node]
        if first == second:  # a node whose only edge is a self-loop: nothing to merge with
            return False
        return not (
            preserve_direction
            and _direction_breaks(oneway_of[first], oneway_of[second], first_at_u, second_at_u)
        )

    return not any(mergeable(node) for node in to_edit)


def _oneway_of(edges_gdf):
    """Each edge's ``oneway`` as 1 or 0 (see graph._oneway_flags), by index; 0 without the column."""
    if "oneway" not in edges_gdf.columns:
        return dict.fromkeys(edges_gdf.index, 0)
    return _oneway_flags(edges_gdf["oneway"]).to_dict()


def _direction_breaks(first_oneway, second_oneway, first_at_u, second_at_u):
    """Whether two segments meeting at a pseudo-node cannot be merged without losing direction:
    one is one-way and the other is not, or both are one-way and do not run on from one into the
    other (they both start, or both end, at the node).
    """
    if first_oneway != second_oneway:
        return True
    return first_oneway == 1 and first_at_u == second_at_u


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
    preserve_direction=False,
):
    """

    The function identify pseudo-nodes, namely nodes that represent intersection between only 2 segments.
    The segments geometries are merged and the node is removed from the nodes_gdf GeoDataFrame.
    The merged segment may join two nodes already joined by another segment: parallel segments
    are kept (see clean_same_vertexes_edges). Merging the last pseudo-node of a loop street leaves
    a self-loop, which is kept: whether it stays is clean_network's `self_loops`. Each
    attribute of a merged segment takes the non-null value covering the greatest length among the
    segments merged into it.

    With `preserve_direction`, a pseudo-node is kept where merging would lose the direction of a
    one-way street (``oneway`` True, 1 or "yes"): one segment is one-way and the other is not, or
    both are one-way but do not run on from one into the other. One-way segments that do are
    merged in their direction.

    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    nodes_to_keep_regardless: list
        List of nodeIDs representing nodes to keep, even when pseudo-nodes (e.g. stations, when modelling transport networks).
    preserve_direction: bool
        Whether a pseudo-node where the direction changes is kept. Default is False.

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
    # A merged edge keeps its first piece's edgeID and, merging only equal values, its oneway.
    oneway_of = _oneway_of(edges_gdf) if preserve_direction else None

    incidence = defaultdict(set)
    for eid in edges_gdf.index:
        incidence[u_of[eid]].add(eid)
        incidence[v_of[eid]].add(eid)

    dropped_edges: set = set()
    dropped_nodes: set = set()
    pieces = {eid: [eid] for eid in edges_gdf.index}

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

        # Which end of each segment is the pseudo-node: two segments of a loop street share both
        # their ends, so the shared end alone does not say where they meet.
        first_at_u, second_at_u = u1 == nodeID, u2 == nodeID
        if preserve_direction and _direction_breaks(
            oneway_of[first], oneway_of[second], first_at_u, second_at_u
        ):
            continue
        if first_at_u and second_at_u:  # meeting at u
            new_u, new_v = v1, v2
            line_a, line_b = coords_first[::-1], coords_second
        elif first_at_u:  # meeting at u and v
            new_u, new_v = u2, v1
            line_a, line_b = coords_second, coords_first
        elif second_at_u:  # meeting at v and u
            new_u, new_v = u1, v2
            line_a, line_b = coords_first, coords_second
        else:  # meeting at v and v
            new_u, new_v = u1, u2
            line_a, line_b = coords_first, coords_second[::-1]

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

    if dropped_nodes:
        nodes_gdf = nodes_gdf.drop(
            index=[n for n in dropped_nodes if n in nodes_gdf.index], errors="ignore"
        )

    return nodes_gdf, edges_gdf


_STRUCTURAL_EDGE_COLUMNS = frozenset(
    {"edgeID", "u", "v", "geometry", "length", "coords", "code", "tmp", "wkt"}
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
        # The piece each winning value was first met on: taking the value from the column by
        # that piece keeps the column's dtype.
        first_piece = frame.drop_duplicates(["_target", "_k"]).set_index(["_target", "_k"])[
            "_piece"
        ]
        targets = list(best.index)
        chosen = original[column].loc[[first_piece.loc[key] for key in best]]
        chosen.index = targets
        edges_gdf.loc[targets, column] = chosen
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
    return nodes_gdf[nodes_gdf["nodeID"].isin(used)].copy()


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

    Each street mapped more than once becomes one edge, the middle of its copies, ranked by their
    total distance to the others: with an odd number of copies the most central one, as mapped;
    with an even number the centre line of the two most central (see center_line), which keeps the
    row, edgeID and attributes of the more central, or shorter, of the two. Different streets are
    all kept, as parallel edges between the same nodes; no node is added.

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
    streets = _same_streets(edges_gdf, same_vertexes_tolerance)
    if not streets:
        return nodes_gdf, edges_gdf

    for street in streets:
        if len(street) % 2 == 0:
            # No middle copy: the centre line of the two most central, in the kept edge's direction.
            kept, other = street[0], street[1]
            edges_gdf.loc[kept, "geometry"] = center_line(
                [edges_gdf.loc[kept, "geometry"], edges_gdf.loc[other, "geometry"]]
            )
            edges_gdf.loc[kept, "length"] = edges_gdf.loc[kept, "geometry"].length
    edges_gdf = edges_gdf.drop([index for street in streets for index in street[1:]], axis=0)
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
    self_loops=False,
):
    """
    Cleans and deduplicates network edges, and removes unused nodes.


    The function performs the following:
      - Generates a unique 'code' for each edge, based on node IDs, with or without preserving direction.
      - Removes self-loop edges (edges from a node to itself), when ``self_loops`` is True.
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
    self_loops : bool, optional
        If True, removes self-loop edges. Default is False; clean_network passes its own
        ``self_loops``.

    Returns
    -------
    nodes_gdf : GeoDataFrame
        Filtered nodes GeoDataFrame, containing only nodes referenced by the cleaned edges.
    edges_gdf : GeoDataFrame
        Cleaned edges GeoDataFrame, deduplicated, and without self-loops when ``self_loops``.
    """
    edges_gdf = edges_gdf.copy()
    edges_gdf["code"] = _pair_codes(edges_gdf, preserve_direction)

    if self_loops:
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
        # Only x and y are moved to the node: a 3D line keeps its own z at each end, so every
        # vertex keeps the same number of dimensions: a line cannot mix 2D and 3D vertices.
        line_coords[0] = (nodes_gdf.loc[u]["x"], nodes_gdf.loc[u]["y"], *line_coords[0][2:])
        line_coords[-1] = (nodes_gdf.loc[v]["x"], nodes_gdf.loc[v]["y"], *line_coords[-1][2:])
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
        Edges with endpoints mapped to new consolidated node IDs and geometries, `length`
        recomputed; an edge whose ends merged is dropped, and of edges that now run along exactly
        the same coordinates (either way) only the first is kept.
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
    if "length" in consolidated_edges.columns:
        consolidated_edges["length"] = consolidated_edges.geometry.length

    # Two edges whose ends merged into the same nodes become one line when neither has an inner
    # vertex (a street and the sidewalk beside it): the same street twice, so the first is kept,
    # as clean_duplicate_edges does. Left in, they would also share a dual-node point, which
    # GeoMason, keying nodes by coordinate, merges into one node.
    consolidated_edges = consolidated_edges[~_same_line_twice(consolidated_edges)]
    consolidated_edges.index = consolidated_edges["edgeID"]
    consolidated_edges.index.name = None

    return consolidated_edges


def _same_line_twice(edges_gdf):
    """Mark every edge after the first that runs along exactly the same coordinates, either way."""
    keys = []
    for line in edges_gdf.geometry:
        coords = tuple(tuple(coord) for coord in line.coords)
        keys.append(min(coords, coords[::-1]))
    return pd.Series(keys, index=edges_gdf.index).duplicated(keep="first")
