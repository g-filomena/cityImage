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

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from .data_utils import convert_numeric_columns
from .geometry import center_line, split_line_at_MultiPoint
from .graph import graph_fromGDF, nodes_degree
from .network import join_nodes_edges_by_coordinates, obtain_nodes_gdf

pd.set_option("display.precision", 3)


# -----------------------------------------------------------------------------
# Topology fixing
# -----------------------------------------------------------------------------
def fix_network_topology(nodes_gdf, edges_gdf, edgeID_column = 'edgeID'):
    """
    Fix the network topology by splitting intersecting edges and adding fixed edges to the network.

    This function considers as segments to be fixed only segments that are actually fully intersecting, thus sharing coordinates, excluding their 
    from and to vertices coordinates, but withouth actually generating, in the given GeoDataFrame, the right number of features.

    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    edgeID_column : str, optional
        Column name for edge unique identifiers in `edges_gdf`. Default is 'edgeID'.
    
    Returns
    -------
    LineString GeoDataFrame
        The updated edges GeoDataFrame.
    """
    edges_gdf = edges_gdf.copy()
    edges_gdf['coords'] = [list(geometry.coords) for geometry in edges_gdf.geometry]
    # spatial index
    sindex = edges_gdf.sindex 

    def find_intersections(ix, line_geometry, coords):
        """
        Find intersection points between a line and other intersecting edges.

        Parameters
        ----------
        ix : int
            The index of the line to check for intersections.
        line_geometry : LineString
            The LineString geometry of the line to check for intersections.
        coords : list
            The list of coordinates of the line's geometry.

        Returns
        -------
        list
            A list of actual intersection points between the line and other intersecting edges.
        """
        
        possible_matches_index = list(sindex.intersection(line_geometry.buffer(5).bounds))
        possible_matches = edges_gdf.iloc[possible_matches_index].copy()
        # lines intersecting the given line
        tmp = possible_matches[possible_matches.intersects(line_geometry)]
        tmp = tmp.drop(ix, axis = 0)
        union = tmp.union_all()
        
        # find actual intersections
        actual_intersections = []
        intersections = line_geometry.intersection(union)
        if intersections is None:
            return actual_intersections      
        if intersections.geom_type == 'LineString': 
            # probably overlapping (to resolve)
            return actual_intersections     
        
        if intersections.geom_type == 'Point': 
            intersections = [intersections]
        else:
            intersections = intersections.geoms
        
        # Obtain all intersecting Points.
        intersection_points = [
            intersection for intersection in intersections if intersection.geom_type == 'Point'
        ]

        def _coord_key(coord, ndigits=10):
            return tuple(round(float(value), ndigits) for value in coord[:2])

        # Keep only intersection points that are existing internal vertices
        # of the current segment. Endpoints are already represented by u/v.
        segment_vertices = {_coord_key(coords[0]), _coord_key(coords[-1])}
        segment_coords = {_coord_key(coord) for coord in coords}

        seen = set()
        for point in intersection_points:
            point_coord = _coord_key(point.coords[0])

            if point_coord not in segment_coords:
                continue
            if point_coord in segment_vertices:
                continue
            if point_coord in seen:
                continue

            actual_intersections.append(point)
            seen.add(point_coord)

        return actual_intersections
    
    # verify which street segment needs to be fixed
    edges_gdf['to_fix'] = edges_gdf.apply(lambda row: find_intersections(row.name, row.geometry, row.coords), axis=1)
    # verify which street segment needs to be fixed
    edges_gdf["fixing"] = [len(to_fix) > 0 for to_fix in edges_gdf["to_fix"]]
    
    to_fix = edges_gdf[edges_gdf['fixing']].copy()
    edges_gdf = edges_gdf[~edges_gdf["fixing"]]
    if len(to_fix) == 0:
        return edges_gdf    
    return _add_fixed_edges(edges_gdf, to_fix, edgeID_column)
    
def fix_fake_self_loops(nodes_gdf, edges_gdf, edgeID_column = 'edgeID'):
    """
    Fix the network topology by removing (fake) self-loops and adding fixed edges.

    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    edgeID_column : str, optional
        Column name for edge unique identifiers in `edges_gdf`. Default is 'edgeID'.

    Returns
    -------
    LineString GeoDataFrame
        The updated edges GeoDataFrame.
    """
    
    edges_gdf = edges_gdf.copy()
    edges_gdf['coords'] = [list(geometry.coords) for geometry in edges_gdf.geometry]
    # all the coordinates but the from and to vertices' ones.
    edges_gdf['coords'] = [coords[1:-1] for coords in edges_gdf.coords]
    
    # convert nodes_gdf['x'] and nodes_gdf['y'] to numpy arrays for faster computation
    x = list(nodes_gdf['x'])
    y = list(nodes_gdf['y'])
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
    edges_gdf['to_fix'] = to_fix
    edges_gdf["fixing"] = [len(to_fix) > 0 for to_fix in edges_gdf["to_fix"]]
    to_fix = edges_gdf[edges_gdf['fixing']].copy()
    edges_gdf = edges_gdf[~edges_gdf["fixing"]]
    if len(to_fix) == 0:
        return nodes_gdf, edges_gdf
    return _add_fixed_edges(edges_gdf, to_fix, edgeID_column)    
 
def _add_fixed_edges(edges_gdf, to_fix_gdf, edgeID_column = 'edgeID'):
    """
    Add fixed edges to the edges GeoDataFrame.

    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    to_fix_gdf: GeoDataFrame
        The GeoDataFrame containing the edges to be fixed.
    edgeID_column : str, optional
        Column name for edge unique identifiers in `edges_gdf`. Default is 'edgeID'.

    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The cleaned junctions and street segments GeoDataFrames.
    """
    dfs = []

    def _split_row_geometry(row):
        split_points = [
            point if isinstance(point, Point) else Point(point)
            for point in row.to_fix
        ]
        return split_line_at_MultiPoint(row.geometry, split_points, z=None)

    new_geometries = to_fix_gdf.apply(_split_row_geometry, axis=1)
    new_geometries = pd.DataFrame(new_geometries, columns = ['lines'])
    
    def append_new_geometries(row):
        for n, line in enumerate(row): # assigning the resulting geometries
            ix = row.name
            index = ix if n == 0 else max(edges_gdf.index) + 1  

            # copy attributes
            row = to_fix_gdf.loc[ix].copy()
            # and assign geometry an new edgeID 
            row[edgeID_column] = index 
            row['geometry'] = line 
            dfs.append(row.to_frame().T)

    new_geometries.apply(lambda row: append_new_geometries(row), axis = 1)
    rows = pd.concat(dfs, ignore_index = True)
    rows = rows.explode(column = 'geometry')
    
    # concatenate the dataframes and assign to edges_gdf
    edges_gdf = pd.concat([edges_gdf, rows], ignore_index=True)
    edges_gdf.drop(['u', 'v', 'to_fix', 'fixing', 'coords'], inplace=True, axis=1)
    edges_gdf['length'] = edges_gdf.geometry.length
    edges_gdf[edgeID_column] = edges_gdf.index
    nodes_gdf = obtain_nodes_gdf(edges_gdf, edges_gdf.crs)
    nodes_gdf, edges_gdf = join_nodes_edges_by_coordinates(nodes_gdf, edges_gdf)
    
    return nodes_gdf, edges_gdf
 
def remove_disconnected_islands(nodes_gdf, edges_gdf, nodeID_column = 'nodeID'):
    """
    Remove disconnected islands from a graph.

    Parameters:
    -----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    nodeID_column : str, optional
        Column name for node unique identifiers in `nodes_gdf`. Default is 'nodeID'.

    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The updated junctions and street segments GeoDataFrame.
    """
    Ng = graph_fromGDF(nodes_gdf, edges_gdf, nodeID_column)
    if not nx.is_connected(Ng):  
        largest_component = max(nx.connected_components(Ng), key=len)
        # Create a subgraph of Ng consisting only of this component:
        G = Ng.subgraph(largest_component)
        to_keep = list(G.nodes())
        nodes_gdf = nodes_gdf[nodes_gdf[nodeID_column].isin(to_keep)]
        edges_gdf = edges_gdf[(edges_gdf.u.isin(nodes_gdf[nodeID_column])) & (edges_gdf.v.isin(nodes_gdf[nodeID_column]))]
        
    return nodes_gdf, edges_gdf


# -----------------------------------------------------------------------------
# Network cleaning
# -----------------------------------------------------------------------------
def clean_network(nodes_gdf, edges_gdf, dead_ends = False, remove_islands = True, same_vertexes_edges = True, self_loops = False, fix_topology = False, 
                  preserve_direction = False, nodeID_column = 'nodeID', edgeID_column  ='edgeID', nodes_to_keep_regardless = None):
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
        If True, treats multiple edges between the same pair of nodes as duplicates. Removes the longer edge if one is >1% longer,
        otherwise replaces them with a center line. Default is True.
    self_loops : bool, optional
        If True, removes self-loop edges (where start and end node are the same). Default is False.
    fix_topology : bool, optional
        If True, breaks lines at intersections with other lines in the streets GeoDataFrame. Default is False.
    preserve_direction : bool, optional
        If True, considers edge direction: edges with the same coordinates but opposite directions are not considered duplicates.
        If False, such edges are treated as duplicates. Default is False.
    nodeID_column : str, optional
        Column name for node unique identifiers in `nodes_gdf`. Default is 'nodeID'.
    edgeID_column : str, optional
        Column name for edge unique identifiers in `edges_gdf`. Default is 'edgeID'.
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
    nodes_gdf, edges_gdf = _prepare_dataframes(nodes_gdf, edges_gdf, nodeID_column, edgeID_column)  
    # removes fake self-loops wrongly coded by the data source
    nodes_gdf, edges_gdf = fix_fake_self_loops(nodes_gdf, edges_gdf, edgeID_column)  
    
    if dead_ends: 
        nodes_gdf, edges_gdf = fix_dead_ends(nodes_gdf, edges_gdf)
    if remove_islands:
        nodes_gdf, edges_gdf = remove_disconnected_islands(nodes_gdf, edges_gdf, nodeID_column)
    if fix_topology: 
        nodes_gdf, edges_gdf = fix_network_topology(nodes_gdf, edges_gdf, edgeID_column)
    
    cycle = 0
    while ((not _are_edges_simplified(edges_gdf, preserve_direction) and same_vertexes_edges) |
           (not _are_nodes_simplified(nodes_gdf, edges_gdf, nodes_to_keep_regardless)) |
            (cycle == 0)):

        edges_gdf['length'] = edges_gdf['geometry'].length # recomputing length, to account for small changes
        cycle += 1
            
        nodes_gdf, edges_gdf = clean_duplicate_nodes(nodes_gdf, edges_gdf, nodeID_column)
        #eliminate loops
        if self_loops: 
            edges_gdf = edges_gdf[edges_gdf['u'] != edges_gdf['v']] 
        if dead_ends: 
            nodes_gdf, edges_gdf = fix_dead_ends(nodes_gdf, edges_gdf)
        
        nodes_gdf, edges_gdf = clean_duplicate_edges(nodes_gdf, edges_gdf, nodeID_column, preserve_direction) 
        
        # edges with different geometries but same u-v nodes pairs
        if same_vertexes_edges:
            nodes_gdf, edges_gdf = clean_same_vertexes_edges(nodes_gdf, edges_gdf,nodeID_column, edgeID_column, preserve_direction)
  
        # simplify the graph                           
        nodes_gdf, edges_gdf = simplify_graph(nodes_gdf, edges_gdf, nodeID_column, edgeID_column, nodes_to_keep_regardless)
        
        # repreat eliminate loops
        if self_loops: 
            edges_gdf = edges_gdf[edges_gdf['u'] != edges_gdf['v']] 
        if dead_ends: 
            nodes_gdf, edges_gdf = fix_dead_ends(nodes_gdf, edges_gdf)
    
    if remove_islands:
        nodes_gdf, edges_gdf = remove_disconnected_islands(nodes_gdf, edges_gdf, nodeID_column)
    
    nodes_gdf['x'], nodes_gdf['y'] = list(zip(*[(r.coords[0][0], r.coords[0][1]) for r in nodes_gdf.geometry], strict=False))
    edges_gdf = correct_edge_geometries(nodes_gdf, edges_gdf) # correct edges coordinates
    return _finalize_dataframes(nodes_gdf, edges_gdf, crs, nodeID_column, edgeID_column)

def _prepare_dataframes(nodes_gdf, edges_gdf, nodeID_column = 'nodeID', edgeID_column = 'edgeID_column'):
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
    nodeID_column : str, optional
        Column name for node unique identifiers in `nodes_gdf`. Default is 'nodeID'.
    edgeID_column : str, optional
        Column name for edge unique identifiers in `edges_gdf`. Default is 'edgeID'.  
    
    Returns:
    ----------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The cleaned junctions and street segments GeoDataFrames.
    """

    nodes_gdf = nodes_gdf.copy().set_index(nodeID_column, drop=False)
    edges_gdf = edges_gdf.copy().set_index(edgeID_column, drop=False)
    
    nodes_gdf.index.name, edges_gdf.index.name = None, None
    nodes_gdf['x'], nodes_gdf['y'] = nodes_gdf.geometry.x, nodes_gdf.geometry.y
    edges_gdf.sort_index(inplace = True)  
    
    if 'highway' in edges_gdf.columns:
        edges_gdf = edges_gdf[edges_gdf['highway'] != 'elevator']
    
    return nodes_gdf, edges_gdf 
 
def _finalize_dataframes(nodes_gdf, edges_gdf, crs, nodeID_column = 'nodeID', edgeID_column = 'edgeID_column'):
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
    nodeID_column : str, optional
        Column name for node unique identifiers in `nodes_gdf`. Default is 'nodeID'.
    edgeID_column : str, optional
        Column name for edge unique identifiers in `edges_gdf`. Default is 'edgeID'.  
    
    Returns:
    ----------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The cleaned junctions and street segments GeoDataFrames.
    """    
        
    nodes_gdf.drop(['wkt'], axis = 1, inplace = True, errors = 'ignore') # remove temporary columns
    edges_gdf.drop(['coords', 'tmp', 'code', 'wkt', 'fixing', 'to_fix'], axis = 1, inplace = True, errors = 'ignore') # remove temporary columns
    edges_gdf['length'] = edges_gdf['geometry'].length
    edges_gdf.set_index(edgeID_column, drop = False, inplace = True, append = False)
    nodes_gdf.set_index(nodeID_column, drop = False, inplace = True, append = False)
    nodes_gdf.index.name = None
    edges_gdf.index.name = None
    nodes_gdf = convert_numeric_columns(nodes_gdf)
    edges_gdf = convert_numeric_columns(edges_gdf)
    nodes_gdf.set_crs(crs, inplace = True)
    edges_gdf.set_crs(crs, inplace = True)
    return nodes_gdf, edges_gdf
 
def _are_nodes_simplified(nodes_gdf, edges_gdf, nodes_to_keep_regardless = None):
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
    
    return len(to_edit) == 0

def _are_edges_simplified(edges_gdf, preserve_direction):
    """

    The function checks the presence of possible duplicate geometries in the edges_gdf GeoDataFrame.
     
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
   
    Returns
    -------
    simplified: bool
        Whether the edges of the network are simplified or not.
    """
    
    edges_gdf = edges_gdf.copy()
    if not preserve_direction:
        edges_gdf['code'] = np.where(
            edges_gdf['v'] >= edges_gdf['u'],
            edges_gdf['u'].astype(str) + "-" + edges_gdf['v'].astype(str),
            edges_gdf['v'].astype(str) + "-" + edges_gdf['u'].astype(str),
        )
    else:
        edges_gdf['code'] = edges_gdf['u'].astype(str) + "-" + edges_gdf['v'].astype(str)

    duplicates = edges_gdf.duplicated('code')
    return not duplicates.any()
        
def clean_duplicate_nodes(nodes_gdf, edges_gdf, nodeID_column = 'nodeID'):
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
    nodeID_column : str, optional
        Name of the column containing unique node identifiers in `nodes_gdf`. Default is 'nodeID'.

    Returns
    -------
    nodes_gdf : GeoDataFrame
        Cleaned nodes GeoDataFrame with duplicates removed.
    edges_gdf : GeoDataFrame
        Edges GeoDataFrame with references to duplicate node IDs updated.
    """
    
    nodes_gdf = nodes_gdf.copy().set_index(nodeID_column, drop=False)
    nodes_gdf.index.name = None
    
    # detecting duplicate geometries
    nodes_gdf['wkt'] = nodes_gdf['geometry'].apply(lambda geom: geom.wkt)
    # Detect duplicates
    subset_cols = ['wkt', 'z'] if 'z' in nodes_gdf.columns else ['wkt']
    new_nodes = nodes_gdf.drop_duplicates(subset=subset_cols).copy()

    # assign univocal nodeID to edges which have 'u' or 'v' referring to duplicate nodes
    # Identify duplicate nodes
    to_edit = set(nodes_gdf.index) - set(new_nodes.index)
    
    if not to_edit:
        return nodes_gdf.drop(columns='wkt'), edges_gdf  # No changes needed
    
    # Map duplicates to their new nodeIDs
    node_mapping = {
        old_node: new_nodes[new_nodes['geometry'] == nodes_gdf.loc[old_node, 'geometry']].index[0]
        for old_node in to_edit
    }

    # readjusting edges' nodes too, accordingly
    edges_gdf[['u', 'v']] = edges_gdf[['u', 'v']].replace(node_mapping)

    return new_nodes.drop(columns='wkt'), edges_gdf

def simplify_graph(nodes_gdf, edges_gdf, nodeID_column = 'nodeID', edgeID_column = 'edgeID', nodes_to_keep_regardless = None):
    """

    The function identify pseudo-nodes, namely nodes that represent intersection between only 2 segments.
    The segments geometries are merged and the node is removed from the nodes_gdf GeoDataFrame.
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    nodeID_column : str, optional
        Column name for node unique identifiers in `nodes_gdf`. Default is 'nodeID'.
    edgeID_column : str, optional
        Column name for edge unique identifiers in `edges_gdf`. Default is 'edgeID'.      
    nodes_to_keep_regardless: list
        List of nodeIDs representing nodes to keep, even when pseudo-nodes (e.g. stations, when modelling transport networks).    
    
    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The cleaned junctions and street segments GeoDataFrames.
    """
    if nodes_to_keep_regardless is None:
        nodes_to_keep_regardless = []

    nodes_gdf =  nodes_gdf.copy()
    edges_gdf = edges_gdf.copy()
    to_edit = list(set(n for n, d in nodes_degree(edges_gdf).items() if d == 2))
    
    if len(to_edit) == 0: 
        return(nodes_gdf, edges_gdf)
    
    if nodes_to_keep_regardless: 
        to_edit_list = list(to_edit)
        tmp_nodes = nodes_gdf[(nodes_gdf[nodeID_column].isin(to_edit_list)) & (~nodes_gdf[nodeID_column].isin(nodes_to_keep_regardless))].copy()
        to_edit = list(tmp_nodes[nodeID_column])

    def _merge_pair_pseudo_edges(first_edge, second_edge, nodes_gdf, edges_gdf, nodeID):
        """
        Merge pseudo-edges by updating node and edge information in the corresponding GeoDataFrames.

        """
        index_first, index_second = first_edge[edgeID_column], second_edge[edgeID_column] 
        first_coords = first_edge['geometry'].coords
        second_coords = second_edge['geometry'].coords
        
        # meeting at u
        if (first_edge['u'] == second_edge['u']):  
            edges_gdf.at[index_first,'u'] = first_edge['v']
            edges_gdf.at[index_first,'v'] = second_edge['v']
            line_coordsA, line_coordsB = list(first_coords), list(second_coords)    
            line_coordsA.reverse()
        # meeting at u and v
        elif (first_edge['u'] == second_edge['v']): 
            edges_gdf.at[index_first,'u'] = second_edge['u']
            line_coordsA, line_coordsB = list(second_coords), list(first_coords)                    
        # meeting at v and u
        elif (first_edge['v'] == second_edge['u']): 
            edges_gdf.at[index_first,'v'] = second_edge['v']
            line_coordsA, line_coordsB = list(first_coords), list(second_coords)  
        # meeting at v and v
        else: # (first_edge['v'] == second_edge['v']) 
            edges_gdf.at[index_first,'v'] = second_edge['u']
            line_coordsA, line_coordsB = list(first_coords), list(second_coords)
            line_coordsB.reverse()

        # checking that no edges with node_u == node_v has been created, if yes: drop it
        if edges_gdf.loc[index_first].u == edges_gdf.loc[index_first].v: 
            edges_gdf = edges_gdf.drop([index_first, index_second], axis = 0)
            nodes_gdf = nodes_gdf.drop(nodeID, axis = 0)
            return nodes_gdf, edges_gdf

        # Obtain coordinates in consistent order and merge them.
        # If both lines meet at the pseudo-node, avoid storing that coordinate twice.
        def _coord_key(coord, ndigits=10):
            return tuple(round(float(value), ndigits) for value in coord[:2])

        if _coord_key(line_coordsA[-1]) == _coord_key(line_coordsB[0]):
            merged_line = line_coordsA + line_coordsB[1:]
        else:
            merged_line = line_coordsA + line_coordsB

        edges_gdf.at[index_first, 'geometry'] = LineString([coor for coor in merged_line]) 
            
        # dropping the second segment, as the new geometry was assigned to the first edge
        edges_gdf = edges_gdf.drop(index_second, axis = 0)
        nodes_gdf = nodes_gdf.drop(nodeID, axis = 0)
        return nodes_gdf, edges_gdf
    
    for nodeID in to_edit:
        tmp = edges_gdf[(edges_gdf.u == nodeID) | (edges_gdf.v == nodeID)].copy()
        
        if len(tmp) == 0: 
            nodes_gdf.drop(nodeID, axis = 0, inplace = True)
            continue
        if len(tmp) == 1: 
            continue # possible dead end
        
        # pseudo junction identified
        first_edge, second_edge = tmp.iloc[0], tmp.iloc[1]
        nodes_gdf, edges_gdf = _merge_pair_pseudo_edges(first_edge, second_edge, nodes_gdf, edges_gdf, nodeID)
        
    edges_gdf = edges_gdf[edges_gdf['u'] != edges_gdf['v']] #eliminate node-lines

    return nodes_gdf, edges_gdf
    
def fix_dead_ends(nodes_gdf, edges_gdf):
    """

    The function removes dead-ends. In other words, it eliminates nodes from where only one segment originates, and the relative segment.
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame
   
    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The cleaned junctions and street segments GeoDataFrames.
    """
    
    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()

    # Find dead-end nodes
    degree = nodes_degree(edges_gdf)
    dead_end_nodes = [node for node, deg in degree.items() if deg == 1]

    if not dead_end_nodes:
        return nodes_gdf, edges_gdf

    # Drop dead-end nodes and their edges
    nodes_gdf = nodes_gdf.drop(dead_end_nodes)
    edges_gdf = edges_gdf[~edges_gdf['u'].isin(dead_end_nodes) & ~edges_gdf['v'].isin(dead_end_nodes)]
    
    return nodes_gdf, edges_gdf

def clean_same_vertexes_edges(nodes_gdf, edges_gdf, nodeID_column = 'nodeID', edgeID_column = 'edgeID', preserve_direction = False):
    """
    Removes duplicate edges with the same start and end nodes (same vertexes) in a network GeoDataFrame.

    For each pair of edges with the same node pair ('u', 'v'), the function:
      - Keeps only the longest edge if one is at least 10% longer than the other(s).
      - If their lengths are similar, generates a center line geometry to represent both and assigns it to one edge.
      - Drops all other duplicate edges.
      - Updates the node GeoDataFrame to retain only nodes still referenced by any edge.

    If `preserve_direction` is False, treats edges as undirected (edges (u,v) and (v,u) are considered duplicates).
    If True, edges in opposite directions are not treated as duplicates.

    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        GeoDataFrame of nodes (junctions), must include unique node IDs.
    edges_gdf : GeoDataFrame
        GeoDataFrame of street segments (edges), must include 'u', 'v', 'geometry', and 'length' columns.
    nodeID_column : str, optional
        Name of the column for unique node IDs in `nodes_gdf`. Default is 'nodeID'.
    edgeID_column : str, optional
        Name of the column for unique edge IDs in `edges_gdf`. Default is 'edgeID'.
    preserve_direction : bool
        Whether to preserve edge direction (see above).    

    Returns
    -------
    nodes_gdf : GeoDataFrame
        Filtered nodes, only those referenced by remaining edges.
    edges_gdf : GeoDataFrame
        Deduplicated edges with updated geometry where applicable.
    """
    to_drop = set()
   
    if not preserve_direction:
        edges_gdf["code"] = np.where(edges_gdf['v'] >= edges_gdf['u'], edges_gdf.u.astype(str)+"-"+edges_gdf.v.astype(str), edges_gdf.v.astype(str)+"-"+edges_gdf.u.astype(str))
    else:
        edges_gdf["code"] = edges_gdf.u.astype(str)+"-"+edges_gdf.v.astype(str)
    if not edges_gdf.duplicated('code').any():
        return nodes_gdf, edges_gdf    
        
    groups = edges_gdf.groupby("code").filter(lambda x: len(x) > 1)[['code','length', edgeID_column]].sort_values(by=['code','length'])
    max_lengths = edges_gdf.groupby("code").agg({'length': 'max'}).to_dict()['length']
    
    for code, g in edges_gdf.groupby("code"):
        if g[g.length < max_lengths[code] * 0.9].shape[0]>0:
            to_drop.update(list(g[g.length < max_lengths[code] * 0.9]['edgeID']))
    
    groups = groups.drop(list(to_drop), axis = 0)
    groups_filtered = groups.groupby('code').filter(lambda x: len(x) > 1)[['code','length', edgeID_column]].sort_values(by=['code','length'])
    first_indexes = list(groups_filtered.groupby("code")[[edgeID_column]].first()[edgeID_column])
    others = set(groups_filtered[edgeID_column].to_list())- set(first_indexes)
    to_drop.update(others)

     # Update the geometry of the first edge in each group to the center line of the edge to update
    for index in first_indexes:
        code = edges_gdf.loc[index]['code']
        geometryA = edges_gdf.loc[index].geometry
        geometryB = edges_gdf.query("code == @code").iloc[1].geometry
        cl = center_line([geometryA, geometryB])
        edges_gdf.at[index, 'geometry'] = cl
             
    edges_gdf = edges_gdf.drop(list(to_drop), axis = 0)
    
    # only keep nodes which are actually used by the edges in the GeoDataFrame
    to_keep = list(set(list(edges_gdf['u'].unique()) + list(edges_gdf['v'].unique())))
    nodes_gdf = nodes_gdf[nodes_gdf[nodeID_column].isin(to_keep)]
    return nodes_gdf, edges_gdf

def clean_duplicate_edges(nodes_gdf, edges_gdf, nodeID_column = 'nodeID', preserve_direction=False, ):
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
    nodeID_column : str, optional
        Name of the column for unique node IDs in `nodes_gdf`. Default is 'nodeID'.
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
    if not preserve_direction:
        edges_gdf["code"] = np.where(edges_gdf['v'] >= edges_gdf['u'], edges_gdf.u.astype(str)+"-"+edges_gdf.v.astype(str), edges_gdf.v.astype(str)+"-"+edges_gdf.u.astype(str))
    else:
        edges_gdf["code"] = edges_gdf.u.astype(str)+"-"+edges_gdf.v.astype(str)

    #eliminate node-lines
    edges_gdf = edges_gdf[edges_gdf['u'] != edges_gdf['v']] 
                        
    # dropping duplicate-geometries edges
    geometries = edges_gdf['geometry'].apply(lambda geom: geom.wkb)
    edges_gdf = edges_gdf.loc[geometries.drop_duplicates().index]
    
    # dropping edges with same geometry but with coords in different orders (depending on their directions)  
    # Reordering coordinates to allow for comparison between edges        
    edges_gdf['coords'] = [list(c.coords) for c in edges_gdf.geometry]
    if not preserve_direction:
        condition = ((edges_gdf.u.astype(str)+"-"+edges_gdf.v.astype(str)) != edges_gdf.code)
        edges_gdf.loc[condition, 'coords'] = pd.Series([x[::-1] for x in edges_gdf.loc[condition]['coords']], index = edges_gdf.loc[condition].index)                                                                               
    
    edges_gdf['tmp'] = edges_gdf['coords'].apply(tuple)  
    edges_gdf.drop_duplicates(['tmp'], keep = 'first', inplace = True)
    
    # only keep nodes which are actually used by the edges in the GeoDataFrame
    to_keep = list(set(list(edges_gdf['u'].unique()) + list(edges_gdf['v'].unique())))
    nodes_gdf = nodes_gdf[nodes_gdf[nodeID_column].isin(to_keep)]
    
    return nodes_gdf, edges_gdf
 
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
        line_coords[0] = (nodes_gdf.loc[u]['x'], nodes_gdf.loc[u]['y'])
        line_coords[-1] = (nodes_gdf.loc[v]['x'], nodes_gdf.loc[v]['y'])
        new_line_geometry = LineString([coor for coor in line_coords])
        return new_line_geometry
        
    edges_gdf['geometry'] = edges_gdf.apply(lambda row: _update_line_geometry_coords(row['u'], row['v'], nodes_gdf, row['geometry']), axis=1)                            
    return edges_gdf


# -----------------------------------------------------------------------------
# Node/edge consolidation
# -----------------------------------------------------------------------------
def consolidate_nodes(nodes_gdf, edges_gdf, nodeID_column = 'nodeID', consolidate_edges_too = False, tolerance = 20):
    """
    Consolidates nodes in a spatial network that are within a given distance (tolerance), preserving topology and unclustered nodes.

    Nodes within `tolerance` distance are clustered together and represented by a single consolidated node at the cluster centroid.
    For clusters containing disconnected components, each connected component is further split into its own consolidated node.
    Optionally, edges can be updated to reference the new consolidated node IDs and geometries.

    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        GeoDataFrame of nodes, must include columns 'nodeID' and 'geometry'. If present, 'z' is averaged for clusters.
    edges_gdf : GeoDataFrame
        GeoDataFrame of edges for checking network connectivity.
    nodeID_column : str, optional
        Column name for node unique identifiers in `nodes_gdf`. Default is 'nodeID'.
    consolidate_edges_too : bool, optional
        If True, also returns the updated edges GeoDataFrame (default: False).
    tolerance : float, optional
        Distance threshold for clustering nodes (in CRS units). Nodes within this distance are merged (default: 20).

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
    
    nodes_gdf = nodes_gdf.copy().set_index(nodeID_column, drop=False)
    nodes_gdf.index.name = None
    nodes_gdf.drop(columns=["x", "y"], inplace=True, errors="ignore")
    graph = graph_fromGDF(nodes_gdf, edges_gdf, nodeID_column = nodeID_column)

    # Step 1: Cluster nodes within tolerance
    clusters = nodes_gdf.buffer(tolerance).union_all()
    clusters = clusters.geoms if hasattr(clusters, "geoms") else [clusters]
    clusters = gpd.GeoDataFrame(geometry=gpd.GeoSeries(clusters, crs=nodes_gdf.crs))
    clusters["x"] = clusters.geometry.centroid.x
    clusters["y"] = clusters.geometry.centroid.y

    # Step 2: Assign nodes to clusters
    gdf = gpd.sjoin(nodes_gdf, clusters, how="left", predicate="within").drop(columns="geometry")
    gdf.rename(columns={"index_right": "new_"+nodeID_column}, inplace=True)
    new_nodeID = gdf.new_nodeID.max() + 1
    
    # Step 3: Split non-connected components in clusters
    for _cluster_label, nodes_subset in gdf.groupby("new_nodeID"):
        if len(nodes_subset) > 1:  # Skip unclustered nodes
            wccs = list(nx.connected_components(graph.subgraph(nodes_subset.index)))
            if len(wccs) > 1:
                for wcc in wccs:
                    idx = list(wcc)
                    subcluster_centroid = nodes_gdf.loc[idx].geometry.unary_union.centroid
                    gdf.loc[idx, ["x", "y"]] = subcluster_centroid.x, subcluster_centroid.y
                    gdf.loc[idx, "new_nodeID"] = new_nodeID
                    new_nodeID += 1

    # Step 4: Consolidate nodes, but preserve unclustered ones
    consolidated_nodes = []
    has_z = 'z' in nodes_gdf.columns
    oldIDs_column = "old_"+nodeID_column
    
    for new_nodeID, nodes_subset in gdf.groupby("new_nodeID"):
        old_nodeIDs = nodes_subset[nodeID_column].to_list()
        cluster_x, cluster_y = nodes_subset.iloc[0][["x", "y"]]

        new_node = {
            oldIDs_column: old_nodeIDs,
            "x": cluster_x,
            "y": cluster_y,
            nodeID_column: new_nodeID,
        }

        if has_z:
            new_node["z"] = nodes_gdf.loc[old_nodeIDs, "z"].mean() if len(old_nodeIDs) > 1 else nodes_gdf.loc[old_nodeIDs[0], "z"]
        
        consolidated_nodes.append(new_node)

    # Convert list of dicts to DataFrame
    consolidated_nodes_df = pd.DataFrame(consolidated_nodes)

    # Create final GeoDataFrame
    consolidated_nodes_gdf = gpd.GeoDataFrame(
        consolidated_nodes_df,
        geometry=gpd.points_from_xy(
            consolidated_nodes_df["x"],
            consolidated_nodes_df["y"],
            consolidated_nodes_df["z"] if "z" in consolidated_nodes_df.columns else None
        ),
        crs=nodes_gdf.crs
    )

    if consolidate_edges_too:
        return consolidated_nodes_gdf, consolidate_edges(edges_gdf, consolidated_nodes_gdf, nodeID_column)
        
    return consolidated_nodes_gdf
    
def consolidate_edges(edges_gdf, consolidated_nodes_gdf, nodeID_column):
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
    
    oldIDs_column = "old_"+nodeID_column
    # Create a mapping from old_nodeIDs to their corresponding nodeID and geometry
    nodes_mapping = (
        consolidated_nodes_gdf.explode(oldIDs_column)[[oldIDs_column, "geometry", nodeID_column]]
        .set_index(oldIDs_column)
    )

    def _update_edge(row):

        old_u, old_v, geom = row["u"], row["v"], row["geometry"]
        
        # Map old_u and old_v to their corresponding new nodeIDs
        new_u_id = nodes_mapping.loc[old_u, nodeID_column]
        new_v_id = nodes_mapping.loc[old_v, nodeID_column]

        # Get the new geometries for u and v
        new_u_geom = nodes_mapping.loc[old_u, "geometry"]
        new_v_geom = nodes_mapping.loc[old_v, "geometry"]

        # Update the geometry (replace first and last coordinates)
        if isinstance(geom, LineString):
            new_coords = [new_u_geom.coords[0]] + list(geom.coords[1:-1]) + [new_v_geom.coords[0]]
            geom = LineString(new_coords)

        return pd.Series({"u": new_u_id, "v": new_v_id, "geometry": geom})

    # Apply updates to the edges
    consolidated_edges = edges_gdf.copy()
    consolidated_edges[["u", "v", "geometry"]] = consolidated_edges.apply(_update_edge, axis=1)
    consolidated_edges = consolidated_edges[consolidated_edges.u != consolidated_edges.v]
    consolidated_edges.index = consolidated_edges['edgeID']
    consolidated_edges.index.name = None
    
    return consolidated_edges
