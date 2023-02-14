import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import networkx as nx
import pandas as pd
import numpy as np
import geopandas as gpd

from shapely.geometry import Point, LineString, MultiPoint, MultiLineString
from shapely.ops import split, unary_union
pd.set_option("display.precision", 3)

from .graph import graph_fromGDF, nodes_degree
from .load import obtain_nodes_gdf, join_nodes_edges_by_coordinates
from .utilities import center_line

"""
This set of functions is designed for cleaning street network's GeoDataFrame (nodes, edges), by taking care of dead_ends, duplicate geometries, same vertexes edges and so on.
"""

def duplicate_nodes(nodes_gdf, edges_gdf):
    """
    The function checks the existencce of duplicate nodes through the network, on the basis of geometry.
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
   
    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        the cleaned junctions and street segments GeoDataFrame
    """
    
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf.index = nodes_gdf.nodeID

    # detecting duplicate geometries
    G = nodes_gdf["geometry"].apply(lambda geom: geom.wkb)
    new_nodes = nodes_gdf.loc[G.drop_duplicates().index]

    # assign univocal nodeID to edges which have 'u' or 'v' referring to duplicate nodes
    to_edit = list(set(nodes_gdf.index.values.tolist()) - set((new_nodes.index.values.tolist())))
    
    if not to_edit: 
        return(nodes_gdf, edges_gdf) 
    
    # readjusting edges' nodes too, accordingly
    for node in to_edit:
        geo = nodes_gdf.loc[node].geometry
        tmp = new_nodes[new_nodes.geometry == geo]
        index = tmp.iloc[0].nodeID
        
        # assigning the unique index to edges
        edges_gdf.loc[edges_gdf.u == node,'u'] = index
        edges_gdf.loc[edges_gdf.v == node,'v'] = index
    
    nodes_gdf = new_nodes.copy()
    return nodes_gdf, edges_gdf

def fix_dead_ends(nodes_gdf, edges_gdf):
    """
    The function removes dead-ends. In other words, it eliminates nodes from where only one segment originates, and the relative segment.
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
   
    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        the cleaned junctions and street segments GeoDataFrame
    """
    
    nodes_gdf =  nodes_gdf.copy()
    edges_gdf = edges_gdf.copy()

    # find the nodes that are dead-ends
    dead_end_nodes = [node for node, degree in nodes_degree(edges_gdf).items() if degree == 1]

    if not dead_end_nodes:  # if there are no dead-end nodes, return the original GeoDataFrames
        return nodes_gdf, edges_gdf

    # drop the dead-end nodes and edges from the GeoDataFrames
    nodes_gdf = nodes_gdf[~nodes_gdf.index.isin(dead_end_nodes)]
    edges_gdf = edges_gdf[~edges_gdf['u'].isin(dead_end_nodes) & ~edges_gdf['v'].isin(dead_end_nodes)]

    return nodes_gdf, edges_gdf

def is_nodes_simplified(nodes_gdf, edges_gdf):
    """
    The function checks the presence of pseudo-junctions, by using the edges_gdf GeoDataFrame.
     
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        street segments
   
    Returns
    -------
    simplified: boolean
        whether the nodes of the network are simplified or not
    """
    
    simplified = True
    to_edit = {k: v for k, v in nodes_degree(edges_gdf).items() if v == 2}
    if 'stationID' in nodes_gdf.columns: # for transport networks
        to_edit_list = list(to_edit.keys())
        tmp_nodes = nodes_gdf[(nodes_gdf.nodeID.isin(to_edit_list)) & (nodes_gdf.stationID == 999999)].copy()
        to_edit = list(tmp_nodes.nodeID)
    
    if len(to_edit) == 0: 
        return simplified
            
    return False

def is_edges_simplified(edges_gdf):
    """
    The function checks the presence of possible duplicate geometries in the edges_gdf GeoDataFrame.
     
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        street segments
   
    Returns
    -------
    simplified: boolean
        whether the edges of the network are simplified or not
    """

    if edges_gdf.duplicated('code').any():
        max_lengths = edges_gdf.groupby("code").agg({'length': 'max'}).to_dict()['length']
        for code, group in edges_gdf.groupby("code"):
            if group[group.length < max_lengths[code] * 0.9].shape[0]>0:
                return False
    return True

""

def simplify_graph(nodes_gdf, edges_gdf):
    """
    The function identify pseudo-nodes, namely nodes that represent intersection between only 2 segments.
    The segments geometries are merged and the node is removed from the nodes_gdf GeoDataFrame.
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
   
    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        the cleaned junctions and street segments GeoDataFrame
    """
    to_edit = set(n for n, d in nodes_degree(edges_gdf).items() if d == 2)
    if len(to_edit) == 0: 
        return(nodes_gdf, edges_gdf)
       
    if 'stationID' in nodes_gdf.columns:
        to_edit = set(nodes_gdf.query("nodeID in @to_edit and stationID == 999999").nodeID)
        if len(to_edit) == 0: 
            return(nodes_gdf, edges_gdf)
    
    sindex = edges_gdf.sindex
   
    for nodeID in to_edit:
        node_geometry = nodes_gdf.loc[nodeID].geometry
        possible_matches_index = list(sindex.intersection(node_geometry.bounds))
        
        if len(possible_matches_index) == 0: 
            nodes_gdf.drop(nodeID, axis = 0, inplace = True)
            continue
        possible_matches = edges_gdf.iloc[possible_matches_index]
        tmp = possible_matches[possible_matches.intersects(node_geometry)]
        if len(tmp) == 1: 
            continue # possible dead end
        
        # pseudo junction identified
        first, second = tmp.iloc[0], tmp.iloc[1]
        index_first, index_second = first.edgeID, second.edgeID # first segment index

        # Identifying the relationship between the two segments.
        # New node_u and node_v are assigned accordingly. A list of ordered coordinates is obtained for 
        # merging the geometries. 4 conditions:
        if (first['u'] == second['u']):  
            edges_gdf.at[index_first,'u'] = first['v']
            edges_gdf.at[index_first,'v'] = second['v']
            line_coordsA, line_coordsB = list(first.coords), list(second.coords)    
            line_coordsA.reverse()
        
        elif (first['u'] == second['v']): 
            edges_gdf.at[index_first,'u'] = second['u']
            line_coordsA, line_coordsB = list(second.coords), list(first.coords)               
        
        elif (first['v'] == second['u']): 
            edges_gdf.at[index_first,'v'] = second['v']
            line_coordsA, line_coordsB = list(first.coords), list(second.coords)  
        
        else: # (first['v'] == second['v']) 
            edges_gdf.at[index_first,'v'] =  second['u']
            line_coordsA, line_coordsB = list(first.coords), list(second.coords)
            line_coordsB.reverse()

        # obtaining coordinates-list in consistent order and merging
        new_line = line_coordsA + line_coordsB
        merged_line = LineString([coor for coor in new_line]) 
        edges_gdf.at[index_first, 'geometry'] = merged_line
        
        if 'pedestrian' in edges_gdf.columns: #type of street 
            if second.pedestrian: 
                edges_gdf.at[index_first, 'pedestrian'] = 1        
        
        # dropping the second segment, as the new geometry was assigned to the first edge
        edges_gdf = edges_gdf.drop(index_second, axis = 0)
        nodes_gdf = nodes_gdf.drop(nodeID, axis = 0)
    
    # checking that none edges with node_u == node_v have been created, if yes: drop them
    edges_gdf.drop(['u', 'v'], inplace = True, axis =1)
    nodes_gdf = obtain_nodes_gdf(edges_gdf, edges_gdf.crs)
    nodes_gdf, edges_gdf = join_nodes_edges_by_coordinates(nodes_gdf, edges_gdf)
    edges_gdf = edges_gdf[~((edges_gdf['u'] == edges_gdf['v']) & (edges_gdf['geometry'].length < 1.00))] #eliminate node-lines

    return nodes_gdf, edges_gdf

def prepare_dataframes(nodes_gdf, edges_gdf):
    """
    Prepare nodes and edges dataframes for further analysis by extracting the x,y coordinates of the nodes
    and adding new columns to the edges dataframe.
    
    Parameters:
    - nodes_gdf (GeoDataFrame): GeoDataFrame containing the nodes of the network.
    - edges_gdf (GeoDataFrame): GeoDataFrame containing the edges of the network.
    
    Returns:
    - nodes_gdf (GeoDataFrame): Updated GeoDataFrame with extracted x,y coordinates of the nodes
    - edges_gdf (GeoDataFrame): Updated GeoDataFrame with added new columns.
    """

    nodes_gdf = nodes_gdf.copy().set_index('nodeID', drop=False)
    edges_gdf = edges_gdf.copy().set_index('edgeID', drop=False)
    nodes_gdf.index.name, edges_gdf.index.name = None, None
    nodes_gdf['x'], nodes_gdf['y'] = list(zip(*[(r.coords[0][0], r.coords[0][1]) for r in nodes_gdf.geometry]))
    edges_gdf.sort_index(inplace = True)  
    edges_gdf["code"] = np.where(edges_gdf['v'] >= edges_gdf['u'], edges_gdf.u.astype(str)+"-"+edges_gdf.v.astype(str), edges_gdf.v.astype(str)+"-"+edges_gdf.u.astype(str))
    
    if 'highway' in edges_gdf.columns:
        edges_gdf['pedestrian'] = 0
        to_remove = ['elevator']
        edges_gdf = edges_gdf[~edges_gdf.highway.isin(to_remove)]
        edges_gdf.loc[edges_gdf.highway.isin(['footway', 'pedestrian', 'living_street', 'path']), 'pedestrian'] = 1
    
    return nodes_gdf, edges_gdf    
    
def simplify_same_vertexes_edges(edges_gdf):
    """
    This function is used to simplify edges that have the same start and end point (i.e. 'u' and 'v' values) 
    in the edges_gdf GeoDataFrame. It removes duplicate edges that have similar geometry, keeping only the one 
    with the longest length if one of them is 10% longer of the other. Otherwise generates a center line
    
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
    
    Returns
    -------
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
    """
    to_drop = set()
    if not edges_gdf.duplicated('code').any():
        return edges_gdf
        
    groups = edges_gdf.groupby("code").filter(lambda x: len(x) > 1)[['code','length', 'edgeID']].sort_values(by=['code','length'])
    max_lengths = edges_gdf.groupby("code").agg({'length': 'max'}).to_dict()['length']
    
    for code, g in edges_gdf.groupby("code"):
        if g[g.length < max_lengths[code] * 0.9].shape[0]>0:
            to_drop.update(list(g[g.length < max_lengths[code] * 0.9]['edgeID']))

    groups_filtered = groups.drop(list(to_drop), axis = 0).groupby('code').filter(lambda x: len(x) > 1)[['code','length','edgeID']].sort_values(by=['code','length'])
    first_indexes = list(groups_filtered.groupby("code")[['edgeID']].first().edgeID)
    others = set(groups_filtered.edgeID.to_list())- set(first_indexes)
    to_drop.update(others)

     # Update the geometry of the first edge in each group to the center line of the edge to update
    for index in first_indexes:
        code = edges_gdf.loc[index]['code']
        geometryA = edges_gdf.loc[index].geometry
        geometryB = edges_gdf.query("code == @code").iloc[1].geometry
        cl = center_line(geometryA, geometryB)
        edges_gdf.at[index, 'geometry'] = cl
        sub_group = edges_gdf.query("code == @code").copy()
        if 'pedestrian' in edges_gdf.columns:
            if sub_group['pedestrian'].any():
                edges_gdf.at[index, 'pedestrian'] = 1 
                    
    edges_gdf = edges_gdf.drop(list(to_drop), axis = 0)
    
    return edges_gdf
                   
def clean_network(nodes_gdf, edges_gdf, dead_ends = False, remove_islands = True, same_vertexes_edges = True, self_loops = False, fix_topology = False):
    """
    It calls a series of functions to clean nodes and edges GeoDataFrames.
    It handles:
        - pseudo-nodes;
        - duplicate-geometries (nodes and edges);
        - disconnected islands - optional;
        - edges with different geometry but same nodes - optional;
        - dead-ends - optional;
        - self-loops - optional;
        - toplogy issues - optional;
           
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
    dead_ends: boolean
        if true remove dead ends
    remove_disconnected_islands: boolean
        if true checks the existence of disconnected islands within the network and deletes corresponding graph components
    same_vertexes_edges: boolean
        if true, it considers as duplicates couple of edges with same vertexes but different geometry. If one of the edges is 1% longer than the other, the longer is deleted; 
        otherwise a center line is built to replace the same vertexes edges.
    fix_topology: boolean
        if true, it breaks lines at intersections with other lines in the streets GeoDataFrame, apart from segments categorised as bridges or tunnels in OSM.
   
    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        the cleaned junctions and street segments GeoDataFrame
    """
    

    nodes_gdf, edges_gdf = prepare_dataframes(nodes_gdf, edges_gdf)  
    if dead_ends: 
        nodes_gdf, edges_gdf = fix_dead_ends(nodes_gdf, edges_gdf)
    if remove_islands:
        nodes_gdf, edges_gdf = remove_disconnected_islands(nodes_gdf, edges_gdf, 'nodeID')
    if fix_topology: 
        nodes_gdf, edges_gdf = fix_network_topology(nodes_gdf, edges_gdf)
        
    cycle = 0
    while ((not is_edges_simplified(edges_gdf) & same_vertexes_edges) | (not is_nodes_simplified(nodes_gdf, edges_gdf)) | (cycle == 0)):

        edges_gdf['length'] = edges_gdf['geometry'].length # recomputing length, to account for small changes
        cycle += 1
            
        nodes_gdf, edges_gdf = duplicate_nodes(nodes_gdf, edges_gdf)
        if self_loops: 
            edges_gdf = edges_gdf[edges_gdf['u'] != edges_gdf['v']] #eliminate loops
        
        edges_gdf = edges_gdf[~((edges_gdf['u'] == edges_gdf['v']) & (edges_gdf['geometry'].length < 1.00))] #eliminate node-lines
                            
        # dropping duplicate-geometries edges
        geometries = edges_gdf['geometry'].apply(lambda geom: geom.wkb)
        edges_gdf = edges_gdf.loc[geometries.drop_duplicates().index]
        
        # dropping edges with same geometry but with coords in different orders (depending on their directions)  
        # Reordering coordinates to allow for comparison between edges        
        edges_gdf['coords'] = [list(c.coords) for c in edges_gdf.geometry]
        edges_gdf.loc[(edges_gdf.u.astype(str)+"-"+edges_gdf.v.astype(str)) != edges_gdf.code]['coords'] = [x[::-1] for x in edges_gdf[(edges_gdf.u.astype(str)+"-"+edges_gdf.v.astype(str)) != edges_gdf.code].coords]        
        edges_gdf['tmp'] = edges_gdf['coords'].apply(tuple, 1)  
        edges_gdf.drop_duplicates(['tmp'], keep = 'first', inplace = True)
        
        # edges with different geometries but same u-v nodes pairs
        if same_vertexes_edges:
            edges_gdf = simplify_same_vertexes_edges(edges_gdf)
  
        if dead_ends: 
            nodes_gdf, edges_gdf = fix_dead_ends(nodes_gdf, edges_gdf)
        
        # only keep nodes which are actually used by the edges in the GeoDataFrame
        to_keep = list(set(list(edges_gdf['u'].unique()) + list(edges_gdf['v'].unique())))
        nodes_gdf = nodes_gdf[nodes_gdf['nodeID'].isin(to_keep)]
        
        # simplify the graph                           
        nodes_gdf, edges_gdf = simplify_graph(nodes_gdf, edges_gdf) 
    
    if remove_islands:
        nodes_gdf, edges_gdf = remove_disconnected_islands(nodes_gdf, edges_gdf, 'nodeID')
    
    nodes_gdf['x'], nodes_gdf['y'] = list(zip(*[(r.coords[0][0], r.coords[0][1]) for r in nodes_gdf.geometry]))
    edges_gdf.drop(['coords', 'tmp'], axis = 1, inplace = True, errors = 'ignore') # remove temporary columns
    nodes_gdf['nodeID'] = nodes_gdf.nodeID.astype(int)
    edges_gdf = correct_edges(nodes_gdf, edges_gdf) # correct edges coordinates
    edges_gdf['length'] = edges_gdf['geometry'].length
    edges_gdf.set_index('edgeID', drop = False, inplace = True, append = False)
    edges_gdf.index.name = None
    
    return nodes_gdf, edges_gdf

def fix_network_topology(nodes_gdf, edges_gdf, use_z = False):
    """
    It breaks lines at intersections with other lines in the streets GeoDataFrame, apart from segments categorised as bridges or tunnels in OSM (if such attribut is provided)
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
       
    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        the cleaned junctions and street segments GeoDataFrame
    """
    
    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()

    to_preserve = []
    columns = ['bridge', 'tunnel']
    bridge_tunnels = edges_gdf.empty().copy()
    for column in columns:
        if column in edges_gdf.columns:
            edges_gdf[column].fillna(0, inplace=True)
            bridge_tunnels = pd.concat([bridge_tunnels, edges_gdf[edges_gdf[column] != 0]], ignore_index=True)
    if len(bridge_tunnels) > 0:
        to_preserve = bridge_tunnels.edgeID.unique()
        
    old_edges_gdf = edges_gdf.copy()
    sindex = old_edges.sindex 
    
    for row in old_edges_gdf.itertuples():
        if old_edges_gdf.loc[row.Index].edgeID in to_preserve:
            continue
        
        line_geometry = old_edges_gdf.loc[row.Index].geometry
        possible_matches_index = list(sindex.intersection(line_geometry.buffer(5).bounds))
        possible_matches = old_edges_gdf.iloc[possible_matches_index].copy()
        tmp = possible_matches[possible_matches.intersects(line_geometry)]
        
        tmp = old_edges_gdf[old_edges_gdf.geometry.intersects(line_geometry)].copy()
        tmp = tmp.drop(row.Index, axis = 0)
        
        union = tmp.unary_union
        intersections = line_geometry.intersection(union)
        if intersections.geom_type == 'Point': 
            intersections = [intersections]
        points = [intersection for intersection in intersections if intersection.geom_type == 'Point']
        new_collection = []

        for point in points:
            if (point.coords[0] == line_geometry.coords[0]) | (point.coords[0] == line_geometry.coords[-1]): 
                pass # disregarding the ones which lie on the line's u-v nodes
            else: 
                new_collection.append(point) # only checking the others
        
        if len(new_collection) == 0: 
            continue 
       
        geometry_collection = MultiPoint([point.coords[0] for point in new_collection])  
        # including the intersecting geometries in the coordinates sequence of the line and split
        new_line_geometries = _split_line_at_MultiPoint(line_geometry, geometry_collection) 

        for n, line in enumerate(new_line_geometries): # assigning the resulting geometries
            if n == 0: 
                index = row.Index
            else: 
                index = max(edges_gdf.index)+1
            # copy attributes
            edges_gdf.loc[index] = edges_gdf.loc[row.Index]  
            # and assign geometry an new edgeID 
            edges_gdf.at[index, 'geometry'] = line
            edges_gdf.at[index, 'edgeID'] = index 
            
    edges_gdf.drop(['u', 'v'], inplace = True, axis =1)
    nodes_gdf = obtain_nodes_gdf(edges_gdf, edges_gdf.crs)
    nodes_gdf, edges_gdf = join_nodes_edges_by_coordinates(nodes_gdf, edges_gdf)
    
    return nodes_gdf, edges_gdf
          
def _split_line_at_MultiPoint(line_geometry, intersection):   
    """
    The function checks whether the coordinates of Point(s) in a Point Collections coordinate are part of the sequence of coordinates of a LineString.
    When this has been ascerted or fixed, the LineString line_geometry is split at each of the intersecting points in the collection.
    
    The input intersection, must be an actual intersection.
               
    Parameters
    ----------
    line_geometry: LineString
        the LineString which has to be split
    intersection: MultiPoint
        the intersecting points
        
    Returns
    -------
    lines: MultiLineString
        the resulting segments composing the original line_geometry
    """
    for point in intersection:
        new_line_coords = list(line_geometry.coords)
        for n, v in enumerate(new_line_coords):
            if n == 0: 
                continue
            line = LineString([Point(new_line_coords[n-1]), Point(v)])
            if ((point.intersects(line)) | (line.distance(point) < 1e-8)):
                new_line_coords.insert(n, point.coords[0])
                break
        line_geometry = LineString([coor for coor in new_line_coords])
                     
    lines = split(line_geometry, intersection)   
    return lines

def remove_disconnected_islands(nodes_gdf, edges_gdf, nodeID):
    """
    Remove disconnected islands from a graph.

    Parameters:
    -----------
    nodes_gdf : GeoDataFrame
        A GeoDataFrame with the nodes information
    edges_gdf : GeoDataFrame
        A GeoDataFrame with the edges information
    nodeID : str
        name of the field containing the nodeIDs

    Returns
    -------
    nodes_gdf, edges_gdf: tuple of GeoDataFrames
        the simplified junctions and street segments GeoDataFrame
    """
    Ng = graph_fromGDF(nodes_gdf, edges_gdf, nodeID)
    if not nx.is_connected(Ng):  
        largest_component = max(nx.connected_components(Ng), key=len)
        
        # Create a subgraph of Ng consisting only of this component:
        G = Ng.subgraph(largest_component)
        list_nodes = list(nodes_gdf.nodeID)
        list_nodesGraph = list(G.nodes())
        to_drop = [item for item in list_nodes if item not in list_nodesGraph]
        nodes_gdf.drop(to_drop, axis = 0 , inplace = True)
        edges_gdf = edges_gdf[(edges_gdf.u.isin(nodes_gdf.nodeID)) & (edges_gdf.v.isin(nodes_gdf.nodeID))]
        
    return nodes_gdf, edges_gdf

def correct_edges(nodes_gdf, edges_gdf):
    """
    The function adjusts the edges LineString coordinates consistently with their relative u and v nodes' coordinates.
    It might be necessary to run the function after having cleaned the network.
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
   
    Returns
    -------
    edges_gdf: LineString GeoDataFrame
        the updated street segments GeoDataFrame
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
        the nodeID of the from node of the geometry
    v: int
        the nodeID of the to node of the geometry
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame 
    line_geometry: LineString
        a street segment geometry
        
    Returns
    -------
    new_line_geometry: LineString
        the readjusted LineString, on the basis of the given u and v nodes
    """
    
    line_coords = list(line_geometry.coords)
    line_coords[0] = (nodes_gdf.loc[u]['x'], nodes_gdf.loc[u]['y'])
    line_coords[-1] = (nodes_gdf.loc[v]['x'], nodes_gdf.loc[v]['y'])
    new_line_geometry = LineString([coor for coor in line_coords])
    
    return new_line_geometry
