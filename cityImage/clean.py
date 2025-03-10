import networkx as nx
import pandas as pd
import numpy as np
import geopandas as gpd

from shapely.geometry import Point, LineString, MultiPoint, MultiLineString
from shapely.ops import unary_union, linemerge
pd.set_option("display.precision", 3)

from .graph import graph_fromGDF, nodes_degree
from .load import obtain_nodes_gdf, join_nodes_edges_by_coordinates, reset_index_graph_gdfs
from .utilities import center_line, split_line_at_MultiPoint

def duplicate_nodes(nodes_gdf, edges_gdf, nodeID = 'nodeID'):
    """
    The function checks the existencce of duplicate nodes through the network, on the basis of geometry.
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
   
    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The cleaned junctions and street segments GeoDataFrames.
    """
    
    nodes_gdf = nodes_gdf.copy().set_index('nodeID', drop=False)
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

def is_nodes_simplified(nodes_gdf, edges_gdf, nodes_to_keep_regardless = []):
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
    
    degree = nodes_degree(edges_gdf)
    to_edit = [node for node, deg in degree.items() if deg == 2]
    
    # Exclude nodes to keep regardless
    if nodes_to_keep_regardless:
        to_edit = [node for node in to_edit if node not in nodes_to_keep_regardless]
    
    return len(to_edit) == 0

def is_edges_simplified(edges_gdf, preserve_direction):
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

def simplify_graph(nodes_gdf, edges_gdf, nodes_to_keep_regardless = []):
    """
    The function identify pseudo-nodes, namely nodes that represent intersection between only 2 segments.
    The segments geometries are merged and the node is removed from the nodes_gdf GeoDataFrame.
     
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
   
    nodes_gdf =  nodes_gdf.copy()
    edges_gdf = edges_gdf.copy()
    to_edit = list(set(n for n, d in nodes_degree(edges_gdf).items() if d == 2))
    
    if len(to_edit) == 0: 
        return(nodes_gdf, edges_gdf)
    
    if nodes_to_keep_regardless: 
        to_edit_list = list(to_edit.keys())
        tmp_nodes = nodes_gdf[(nodes_gdf.nodeID.isin(to_edit_list)) & (~nodes_gdf.nodeID.isin(nodes_to_keep_regardless))].copy()
        to_edit = list(tmp_nodes.nodeID)
      
    for nodeID in to_edit:
        tmp = edges_gdf[(edges_gdf.u == nodeID) | (edges_gdf.v == nodeID)].copy()
        
        if len(tmp) == 0: 
            nodes_gdf.drop(nodeID, axis = 0, inplace = True)
            continue
        if len(tmp) == 1: 
            continue # possible dead end
        
        # pseudo junction identified
        first_edge, second_edge = tmp.iloc[0], tmp.iloc[1]
        nodes_gdf, edges_gdf = merge_pseudo_edges(first_edge, second_edge, nodeID, nodes_gdf, edges_gdf)
        
    edges_gdf = edges_gdf[edges_gdf['u'] != edges_gdf['v']] #eliminate node-lines

    return nodes_gdf, edges_gdf

def merge_pseudo_edges(first_edge, second_edge, nodeID, nodes_gdf, edges_gdf):
    """
    Merge pseudo-edges by updating node and edge information in the corresponding GeoDataFrames.

    Parameters
    ----------
    first_edge: Series
        The first pseudo-edge to be merged.
    second_edge: Series
        The second pseudo-edge to be merged.
    nodeID: int
        The nodeID of the node where the pseudo-edges meet.
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.

    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The updated junctions and street segments GeoDataFrames.
    """
    index_first, index_second = first_edge.edgeID, second_edge.edgeID 
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

    # obtaining coordinates-list in consistent order and merging
    merged_line = line_coordsA + line_coordsB
    edges_gdf.at[index_first, 'geometry'] = LineString([coor for coor in merged_line]) 
        
    # dropping the second segment, as the new geometry was assigned to the first edge
    edges_gdf = edges_gdf.drop(index_second, axis = 0)
    nodes_gdf = nodes_gdf.drop(nodeID, axis = 0)
    return nodes_gdf, edges_gdf

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
    
    Returns:
    ----------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The cleaned junctions and street segments GeoDataFrames.
    """

    nodes_gdf = nodes_gdf.copy().set_index('nodeID', drop=False)
    edges_gdf = edges_gdf.copy().set_index('edgeID', drop=False)
    
    nodes_gdf.index.name, edges_gdf.index.name = None, None
    nodes_gdf['x'], nodes_gdf['y'] = nodes_gdf.geometry.x, nodes_gdf.geometry.y
    edges_gdf.sort_index(inplace = True)  
    
    if 'highway' in edges_gdf.columns:
        edges_gdf = edges_gdf[edges_gdf['highway'] != 'elevator']
    
    return nodes_gdf, edges_gdf    
    
def simplify_same_vertexes_edges(nodes_gdf, edges_gdf, preserve_direction):
    """
    This function is used to simplify edges that have the same start and end point (i.e. 'u' and 'v' values) 
    in the edges_gdf GeoDataFrame. It removes duplicate edges that have similar geometry, keeping only the one 
    with the longest length if one of them is 10% longer of the other. Otherwise generates a center line
    
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    
    Returns
    -------
    edges_gdf: LineString GeoDataFrame
        The clean street segments GeoDataFrames.
    """
    to_drop = set()
   
    if not preserve_direction:
        edges_gdf["code"] = np.where(edges_gdf['v'] >= edges_gdf['u'], edges_gdf.u.astype(str)+"-"+edges_gdf.v.astype(str), edges_gdf.v.astype(str)+"-"+edges_gdf.u.astype(str))
    else:
        edges_gdf["code"] = edges_gdf.u.astype(str)+"-"+edges_gdf.v.astype(str)
    if not edges_gdf.duplicated('code').any():
        return nodes_gdf, edges_gdf    
        
    groups = edges_gdf.groupby("code").filter(lambda x: len(x) > 1)[['code','length', 'edgeID']].sort_values(by=['code','length'])
    max_lengths = edges_gdf.groupby("code").agg({'length': 'max'}).to_dict()['length']
    
    for code, g in edges_gdf.groupby("code"):
        if g[g.length < max_lengths[code] * 0.9].shape[0]>0:
            to_drop.update(list(g[g.length < max_lengths[code] * 0.9]['edgeID']))
    
    groups = groups.drop(list(to_drop), axis = 0)
    groups_filtered = groups.groupby('code').filter(lambda x: len(x) > 1)[['code','length','edgeID']].sort_values(by=['code','length'])
    first_indexes = list(groups_filtered.groupby("code")[['edgeID']].first().edgeID)
    others = set(groups_filtered.edgeID.to_list())- set(first_indexes)
    to_drop.update(others)

     # Update the geometry of the first edge in each group to the center line of the edge to update
    for index in first_indexes:
        code = edges_gdf.loc[index]['code']
        geometryA = edges_gdf.loc[index].geometry
        geometryB = edges_gdf.query("code == @code").iloc[1].geometry
        cl = center_line([geometryA, geometryB])
        edges_gdf.at[index, 'geometry'] = cl
        sub_group = edges_gdf.query("code == @code").copy()
             
    edges_gdf = edges_gdf.drop(list(to_drop), axis = 0)
    
    # only keep nodes which are actually used by the edges in the GeoDataFrame
    to_keep = list(set(list(edges_gdf['u'].unique()) + list(edges_gdf['v'].unique())))
    nodes_gdf = nodes_gdf[nodes_gdf['nodeID'].isin(to_keep)]
        
    return nodes_gdf, edges_gdf

def clean_edges(nodes_gdf, edges_gdf, preserve_direction = False):

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
    nodes_gdf = nodes_gdf[nodes_gdf['nodeID'].isin(to_keep)]
    
    return nodes_gdf, edges_gdf

def clean_network(nodes_gdf, edges_gdf, dead_ends = False, remove_islands = True, same_vertexes_edges = True, self_loops = False, fix_topology = False, 
                  preserve_direction = False, nodes_to_keep_regardless = []):
    """
    It calls a series of functions to clean nodes and edges GeoDataFrames.
    It handles:
        - pseudo-nodes;
        - duplicate-geometries (nodes and edges);
        - disconnected islands - optional;
        - edges with different geometry but same nodes - optional;
        - dead-ends - optional;
        - self-loops - optional;
        - toplogy issues - optional.
           
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    dead_ends: bool
        When true remove dead ends.
    remove_islands: bool
        When true checks the existence of disconnected islands within the network and deletes corresponding graph components.
    same_vertexes_edges: bool
        When true, it considers as duplicates couple of edges with same vertexes but different geometry. If one of the edges is 1% longer than the other, 
        the longer is deleted; otherwise a center line is built to replace the same vertexes edges.
    self_loops: bool
        When true, removes genuine self-loops.
    fix_topology: bool
        When true, it breaks lines at intersections with other lines in the streets GeoDataFrame.
    preserve_direction: bool
        When true, it does not consider segments with same coordinates list, but different directions, as identical. When false, it does and therefore
        considers them as duplicated geometries.
    nodes_to_keep_regardless: list
        List of nodeIDs representing nodes to keep, even when pseudo-nodes (e.g. stations, when modelling transport networks).
    
    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The cleaned junctions and street segments GeoDataFrames.
    """
    nodes_gdf, edges_gdf = _prepare_dataframes(nodes_gdf, edges_gdf)  
    # removes fake self-loops wrongly coded by the data source
    nodes_gdf, edges_gdf = fix_self_loops(nodes_gdf, edges_gdf)  
    
    if dead_ends: 
        nodes_gdf, edges_gdf = fix_dead_ends(nodes_gdf, edges_gdf)
    if remove_islands:
        nodes_gdf, edges_gdf = remove_disconnected_islands(nodes_gdf, edges_gdf, 'nodeID')
    if fix_topology: 
        nodes_gdf, edges_gdf = fix_network_topology(nodes_gdf, edges_gdf)
    
    cycle = 0
    while ((not is_edges_simplified(edges_gdf, preserve_direction) and same_vertexes_edges) |
            (not is_nodes_simplified(nodes_gdf, edges_gdf, nodes_to_keep_regardless)) |
            (cycle == 0)):

        edges_gdf['length'] = edges_gdf['geometry'].length # recomputing length, to account for small changes
        cycle += 1
            
        nodes_gdf, edges_gdf = duplicate_nodes(nodes_gdf, edges_gdf)
        #eliminate loops
        if self_loops: 
            edges_gdf = edges_gdf[edges_gdf['u'] != edges_gdf['v']] 
        if dead_ends: 
            nodes_gdf, edges_gdf = fix_dead_ends(nodes_gdf, edges_gdf)
        
        nodes_gdf, edges_gdf = clean_edges(nodes_gdf, edges_gdf, preserve_direction) 
        
        # edges with different geometries but same u-v nodes pairs
        if same_vertexes_edges:
            nodes_gdf, edges_gdf = simplify_same_vertexes_edges(nodes_gdf, edges_gdf, preserve_direction)
  
        # simplify the graph                           
        nodes_gdf, edges_gdf = simplify_graph(nodes_gdf, edges_gdf, nodes_to_keep_regardless) 
        
        #eliminate loops
        if self_loops: 
            edges_gdf = edges_gdf[edges_gdf['u'] != edges_gdf['v']] 
        if dead_ends: 
            nodes_gdf, edges_gdf = fix_dead_ends(nodes_gdf, edges_gdf)
    
    if remove_islands:
        nodes_gdf, edges_gdf = remove_disconnected_islands(nodes_gdf, edges_gdf, 'nodeID')
    
    nodes_gdf['x'], nodes_gdf['y'] = list(zip(*[(r.coords[0][0], r.coords[0][1]) for r in nodes_gdf.geometry]))
    
    nodes_gdf['nodeID'] = nodes_gdf.nodeID.astype(int)
    edges_gdf = correct_edges(nodes_gdf, edges_gdf) # correct edges coordinates
    
    nodes_gdf.drop(['wkt'], axis = 1, inplace = True, errors = 'ignore') # remove temporary columns
    edges_gdf.drop(['coords', 'tmp', 'code', 'wkt', 'fixing', 'to_fix'], axis = 1, inplace = True, errors = 'ignore') # remove temporary columns
    edges_gdf['length'] = edges_gdf['geometry'].length
    edges_gdf.set_index('edgeID', drop = False, inplace = True, append = False)
    
    nodes_gdf.index.name = None
    edges_gdf.index.name = None
    
    return nodes_gdf, edges_gdf

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
    new_geometries = to_fix_gdf.apply(lambda row: split_line_at_MultiPoint(row.geometry, 
                                                                            [Point(coord) for coord in row.to_fix]), axis=1)
    new_geometries = pd.DataFrame(new_geometries, columns = ['lines'])
    
    def append_new_geometries(row):
        for n, line in enumerate(row): # assigning the resulting geometries
            ix = row.name
            if n == 0: 
                index = ix
            else: 
                index = max(edges_gdf.index)+1

            # copy attributes
            row = to_fix_gdf.loc[ix].copy()
            # and assign geometry an new edgeID 
            row['edgeID'] = index 
            row['geometry'] = line 
            dfs.append(row.to_frame().T)

    new_geometries.apply(lambda row: append_new_geometries(row), axis = 1)
    rows = pd.concat(dfs, ignore_index = True)
    rows = rows.explode(column = 'geometry')
    
    # concatenate the dataframes and assign to edges_gdf
    edges_gdf = pd.concat([edges_gdf, rows], ignore_index=True)
    edges_gdf.drop(['u', 'v', 'to_fix', 'fixing', 'coords'], inplace=True, axis=1)
    edges_gdf['length'] = edges_gdf.geometry.length
    edges_gdf['edgeID'] = edges_gdf.index
    nodes_gdf = obtain_nodes_gdf(edges_gdf, edges_gdf.crs)
    nodes_gdf, edges_gdf = join_nodes_edges_by_coordinates(nodes_gdf, edges_gdf)
    
    return nodes_gdf, edges_gdf
   
def fix_network_topology(nodes_gdf, edges_gdf):
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
    
    Returns
    -------
    LineString GeoDataFrame
        The updated edges GeoDataFrame.
    """
    edges_gdf.copy()
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
        union = tmp.unary_union
        
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
        
        # from and to vertices of the given line
        segment_vertices = [coords[0], coords[-1]]
        # obtaining all the intersecting Points
        intersection_points = [intersection for intersection in intersections if intersection.geom_type == 'Point']
        
        # keeping intersections that are in the coordinate list of the given line, without actually coinciding with the from and to vertices
        for point in intersection_points: 
            if (point.coords[0] not in coords):
                pass
            if (point.coords[0] in segment_vertices): 
                pass 
            else: 
                actual_intersections.append(point)
        return actual_intersections
    
    # verify which street segment needs to be fixed
    edges_gdf['to_fix'] = edges_gdf.apply(lambda row: find_intersections(row.name, row.geometry, row.coords), axis=1)
    # verify which street segment needs to be fixed
    edges_gdf['fixing'] = [True if len(to_fix) > 0 else False for to_fix in edges_gdf['to_fix']]
    
    to_fix = edges_gdf[edges_gdf['fixing'] == True].copy()
    edges_gdf = edges_gdf[edges_gdf['fixing'] == False]   
    if len(to_fix) == 0:
        return edges_gdf    
    return _add_fixed_edges(edges_gdf, to_fix)
    
def fix_self_loops(nodes_gdf, edges_gdf):
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
    edges_gdf['coords'] = [list(geometry.coords) for geometry in edges_gdf.geometry]
    # all the coordinates but the from and to vertices' ones.
    edges_gdf['coords'] = [coords[1:-1] for coords in edges_gdf.coords]
    
    # convert nodes_gdf['x'] and nodes_gdf['y'] to numpy arrays for faster computation
    x = list(nodes_gdf['x'])
    y = list(nodes_gdf['y'])
    # create a set of all coordinates in nodes. This essentially correspond to the from and to nodes of the edges currently in the edges_gdf
    nodes_set = set(zip(x, y))

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
    edges_gdf['fixing'] = [True if len(to_fix) > 0 else False for to_fix in edges_gdf['to_fix']]
    to_fix = edges_gdf[edges_gdf['fixing'] == True].copy()
    edges_gdf = edges_gdf[edges_gdf['fixing'] == False]
    if len(to_fix) == 0:
        return nodes_gdf, edges_gdf
    return _add_fixed_edges(edges_gdf, to_fix)    
    
def remove_disconnected_islands(nodes_gdf, edges_gdf, nodeID):
    """
    Remove disconnected islands from a graph.

    Parameters:
    -----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    nodeID: str
        The name of the field containing the nodeIDs.

    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        The updated junctions and street segments GeoDataFrame.
    """
    Ng = graph_fromGDF(nodes_gdf, edges_gdf, nodeID)
    if not nx.is_connected(Ng):  
        largest_component = max(nx.connected_components(Ng), key=len)
        # Create a subgraph of Ng consisting only of this component:
        G = Ng.subgraph(largest_component)
        to_keep = list(G.nodes())
        nodes_gdf = nodes_gdf[nodes_gdf[nodeID].isin(to_keep)]
        edges_gdf = edges_gdf[(edges_gdf.u.isin(nodes_gdf[nodeID])) & (edges_gdf.v.isin(nodes_gdf[nodeID]))]
        
    return nodes_gdf, edges_gdf

def correct_edges(nodes_gdf, edges_gdf):
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
    edges_gdf['geometry'] = edges_gdf.apply(lambda row: _update_line_geometry_coords(row['u'], row['v'], nodes_gdf, row['geometry']), axis=1)                            
    return edges_gdf

def _update_line_geometry_coords(u, v, nodes_gdf, line_geometry):
    """
    It supports the correct_edges function checks that the edges coordinates are consistent with their relative u and v nodes'coordinates.
    It can be necessary to run the function after having cleaned the network.
    
    Parameters
    ----------
    u: int
        The nodeID of the from node of the geometry.
    v: int
        The nodeID of the to node of the geometry.
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame .
    line_geometry: LineString
        A street segment geometry.
        
    Returns
    -------
    new_line_geometry: LineString
        The readjusted LineString, on the basis of the given u and v nodes.
    """
    line_coords = list(line_geometry.coords)
    line_coords[0] = (nodes_gdf.loc[u]['x'], nodes_gdf.loc[u]['y'])
    line_coords[-1] = (nodes_gdf.loc[v]['x'], nodes_gdf.loc[v]['y'])
    new_line_geometry = LineString([coor for coor in line_coords])
    return new_line_geometry

def consolidate_nodes(nodes_gdf, edges_gdf, consolidate_edges_too = False, tolerance=20):
    """
    Consolidates nearby nodes within a given tolerance, preserving unclustered nodes.

    Parameters:
    - nodes_gdf (GeoDataFrame): Nodes with 'nodeID' and 'geometry'.
    - edges_gdf (GeoDataFrame): Edges for connectivity checks.
    - tolerance (float): Distance threshold for clustering.

    Returns:
    - GeoDataFrame: Consolidated nodes with preserved unclustered nodes.
    """
    
    nodes_gdf = nodes_gdf.copy().set_index('nodeID', drop=False)
    nodes_gdf.index.name = None
    nodes_gdf.drop(columns=["x", "y"], inplace=True, errors="ignore")

    graph = graph_fromGDF(nodes_gdf, edges_gdf, nodeID="nodeID")

    # Step 1: Cluster nodes within tolerance
    clusters = nodes_gdf.buffer(tolerance).unary_union
    clusters = clusters.geoms if hasattr(clusters, "geoms") else [clusters]
    clusters = gpd.GeoDataFrame(geometry=gpd.GeoSeries(clusters, crs=nodes_gdf.crs))
    clusters["x"] = clusters.geometry.centroid.x
    clusters["y"] = clusters.geometry.centroid.y

    # Step 2: Assign nodes to clusters
    gdf = gpd.sjoin(nodes_gdf, clusters, how="left", predicate="within").drop(columns="geometry")
    gdf.rename(columns={"index_right": "new_nodeID"}, inplace=True)
    new_nodeID = gdf.new_nodeID.max() + 1
    
    # Step 3: Split non-connected components in clusters
    for cluster_label, nodes_subset in gdf.groupby("new_nodeID"):
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

    for new_nodeID, nodes_subset in gdf.groupby("new_nodeID"):
        old_nodeIDs = nodes_subset.nodeID.to_list()
        cluster_x, cluster_y = nodes_subset.iloc[0][["x", "y"]]

        new_node = {
            "old_nodeIDs": old_nodeIDs,
            "x": cluster_x,
            "y": cluster_y,
            "nodeID": new_nodeID,
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
        return consolidated_nodes_gdf, consolidate_edges(edges_gdf, consolidated_nodes_gdf), 
        
    return consolidated_nodes_gdf
    
def consolidate_edges(edges_gdf, consolidated_nodes_gdf):
    """
    Updates 'u' and 'v' in edges based on the consolidated nodes.
    Replaces old_nodeIDs in 'u' and 'v' with the corresponding new nodeID from nodes_consolidated.
    Also updates the geometry of edges by replacing the first and last coordinates with new node positions.
    """
    # Create a mapping from old_nodeIDs to their corresponding nodeID and geometry
    nodes_mapping = (
        consolidated_nodes_gdf.explode("old_nodeIDs")[["old_nodeIDs", "geometry", "nodeID"]]
        .set_index("old_nodeIDs")
    )

    def update_edge(row):
        """Update edge properties based on consolidated nodes."""
        old_u, old_v, geom = row["u"], row["v"], row["geometry"]
        
        # Map old_u and old_v to their corresponding new nodeIDs
        new_u_id = nodes_mapping.loc[old_u, "nodeID"]
        new_v_id = nodes_mapping.loc[old_v, "nodeID"]

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
    consolidated_edges[["u", "v", "geometry"]] = consolidated_edges.apply(update_edge, axis=1)
    consolidated_edges = consolidated_edges[consolidated_edges.u != consolidated_edges.v]
    consolidated_edges.index = consolidated_edges['edgeID']
    consolidated_edges.index.name = None
    
    return consolidated_edges
    
    