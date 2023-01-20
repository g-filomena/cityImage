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
    new_nodes  = nodes_gdf.drop_duplicates(subset=nodes_gdf["geometry"].apply(lambda geom: geom.wkb))

    # assign univocal nodeID to edges which have 'u' or 'v' referring to duplicate nodes
    to_edit = list(set(nodes_gdf.index.values.tolist()) - set((new_nodes.index.values.tolist())))
    
    if to_edit.empty: 
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
    
    simplified = True 
    
    edges_gdf['code'] = None
    edges_gdf['code'][edges_gdf['v'] >= edges_gdf['u']] = edges_gdf.u.astype(str)+"-"+edges_gdf.v.astype(str)
    edges_gdf['code'][edges_gdf['v'] < edges_gdf['u']] = edges_gdf.v.astype(str)+"-"+edges_gdf.u.astype(str)
    dd = dict(edges_gdf['code'].value_counts())
    dd = {k: v for k, v in dd.items() if v > 1}
    
    if len(dd) > 0: 
        simplified = False
    
    return simplified

""

def simplify_graph(nodes_gdf, edges_gdf):
    """
    The function identify pseudo-nodes, namely nodes that represent intersection between only 2 segments.
    The segments are merged and the node is removed from the nodes_gdf GeoDataFrame.
     
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
    to_edit = set(nx.degree(nx.from_pandas_edgelist(edges_gdf, 'u', 'v')).keys()) & set(nodes_gdf.nodeID)
    if len(to_edit) == 0 or len(edges_gdf)==0: 
        return(nodes_gdf, edges_gdf)
    
    if 'stationID' in nodes_gdf.columns:
        to_edit = set(nodes_gdf.query("nodeID in @to_edit and stationID == 999999").nodeID)
        if len(to_edit) == 0: 
            return(nodes_gdf, edges_gdf)
    
    nodes_gdf = nodes_gdf[nodes_gdf.apply(lambda x: x['nodeID'] not in to_edit, axis=1)]
    edges_gdf = edges_gdf[edges_gdf.apply(lambda x: x['u'] not in to_edit and x['v'] not in to_edit, axis=1)]
    
    for nodeID in to_edit:
        edges = edges_gdf.query("u == @nodeID or v == @nodeID")
        if len(edges) == 0: 
            continue
        if len(edges) == 1: 
            continue # possible dead end
        index_first, index_second = edges.iloc[0].edgeID, edges.iloc[1].edgeID
        u, v = edges.iloc[1]['u' if edges.iloc[0]['u'] == edges.iloc[1]['v'] else 'v'], edges.iloc[1]['v' if edges.iloc[0]['u'] == edges.iloc[1]['v'] else 'u']
        edges_gdf.loc[index_first, ['u', 'v']] = u, v
        line_coordsA = edges.iloc[0]['geometry'].coords.tolist()
        line_coordsB = edges.iloc[1]['geometry'].coords.tolist()
        
        if edges.iloc[0]['u'] == edges.iloc[1]['v']:
            line_coordsB.reverse()
        if u == v:
            edges_gdf = edges_gdf[edges_gdf.edgeID != index_first]
            continue
        
        new_line = line_coordsA + line_coordsB
        merged_line = LineString([coor for coor in new_line])
        edges_gdf.loc[index_first, 'geometry'] = merged_line
        pedestrian = edges_gdf.loc[index_second]['pedestrian'] or edges_gdf.loc[index_first]['pedestrian']
        edges_gdf.loc[index_first, 'pedestrian'] = pedestrian
        edges_gdf = edges_gdf[edges_gdf.edgeID != index_second]

    return nodes_gdf, edges_gdf

def prepare_dataframes(nodes_gdf, edges_gdf)


    nodes_gdf = nodes_gdf.copy().set_index('nodeID', drop=False)
    nodes_gdf.index.name = None
    edges_gdf = edges_gdf.copy().set_index('edgeID', drop=False)
    edges_gdf.index.name = None
    nodes_gdf['x'], nodes_gdf['y'] = list(zip(*[(r.coords[0][0], r.coords[0][1]) for r in nodes_gdf.geometry]))
    edges_gdf.sort_index(inplace = True)  
    edges_gdf['code'], edges_gdf['coords'] = None, None
    if 'highway' in edges_gdf.columns:
        edges_gdf['pedestrian'] = 0
        to_remove = ['elevator']
        edges_gdf = edges_gdf[~edges_gdf.highway.isin(to_remove)]
        edges_gdf.loc[edges_gdf.highway.isin(['footway', 'pedestrian', 'living_street', 'path']), 'pedestrian'] = 1
    
    return edges_gdf    
    
def simplify_same_uv_edges(edges_gdf):
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

    edges_gdf["code"] = np.where(edges_gdf['v'] >= edges_gdf['u'], edges_gdf.u.astype(str)+"-"+edges_gdf.v.astype(str), edges_gdf.v.astype(str)+"-"+edges_gdf.u.astype(str))
    edges_gdf = edges_gdf.sort_values(by=['code','length'])
    to_drop = []
    
    # Get the index of the first edge in each "code" group
    first_edge_ix = edges_gdf.groupby("code").first().index
    
    # Get the edges that are not the first in their group and have a length within 10% of the first edge in their group
    to_update = edges_gdf[~edges_gdf.index.isin(first_edge_ix) & (edges_gdf["length"] <= edges_gdf.loc[first_edge_ix, "length"].mul(1.1))]
    
    # Update the geometry of the first edge in each group to the center line of the edge to update
    for code, group in to_update.groupby("code"):
        cl = center_line(edges_gdf.loc[first_edge_ix[code], "geometry"], group["geometry"])
        edges_gdf.at[first_edge_ix[code], 'geometry'] = cl
        
        if 'highway' in edges_gdf.columns:
            if group['pedestrian'].any():
                edges_gdf.at[first_edge_ix[code], 'pedestrian'] = 1 
                
        # Append the index of edges that are not the first in their group to the list of edges to drop
        to_drop.extend(group.index)
    edges_gdf = edges_gdf.drop(to_drop)
    
    return edges_gdf


                    
def clean_network(nodes_gdf, edges_gdf, dead_ends = False, remove_islands = True, same_uv_edges = True, self_loops = False, fix_topology = False):
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
    same_uv_edges: boolean
        if true, it considers as duplicates couple of edges with same vertexes but different geometry. If one of the edges is 1% longer than the other, the longer is deleted; 
        otherwise a center line is built to replace the same_uv_edges.
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
        nodes_gdf, edges_gdf = remove_islands(nodes_gdf, edges_gdf, nodeID)
        
    cycle = 0
    while (((not is_edges_simplified(edges_gdf)) & same_uv_edges) | (not is_nodes_simplified(nodes_gdf, edges_gdf)) | (cycle == 0)):

        edges_gdf['length'] = edges_gdf['geometry'].length # recomputing length, to account for small changes
        cycle += 1
            
        nodes_gdf, edges_gdf = duplicate_nodes(nodes_gdf, edges_gdf)
        if self_loops: 
            edges_gdf = edges_gdf[edges_gdf['u'] != edges_gdf['v']] #eliminate loops
        
        edges_gdf = edges_gdf[~((edges_gdf['u'] == edges_gdf['v']) & (edges_gdf['geometry'].length < 1.00))] #eliminate node-lines
                
        # Reordering coordinates to allow for comparison between edges
        edges_gdf['coords'] = [list(c.coords) for c in edges_gdf.geometry]
        edges_gdf['coords'][(edges_gdf.u.astype(str)+"-"+edges_gdf.v.astype(str)) != edges_gdf.code] = [list(x.coords)[::-1] for x in edges_gdf.geometry]
        
        # dropping duplicate-geometries edges
        geometries = edges_gdf['geometry'].apply(lambda geom: geom.wkb)
        edges_gdf = edges_gdf.loc[geometries.drop_duplicates().index]
        
        # dropping edges with same geometry but with coords in different orders (depending on their directions)    
        edges_gdf['tmp'] = edges_gdf['coords'].apply(tuple, 1)  
        edges_gdf.drop_duplicates(['tmp'], keep = 'first', inplace = True)
        
        # edges with different geometries but same u-v nodes pairs
        edges_gdf = simplify_same_uv_edges(edges_gdf)
  
        if dead_ends: 
            nodes_gdf, edges_gdf = fix_dead_ends(nodes_gdf, edges_gdf)
        
        # only keep nodes which are actually used by the edges in the GeoDataFrame
        to_keep = list(set(list(edges_gdf['u'].unique()) + list(edges_gdf['v'].unique())))
        nodes_gdf = nodes_gdf[nodes_gdf['nodeID'].isin(to_keep)]
        if fix_topology: 
            nodes_gdf, edges_gdf = fix_network_topology(nodes_gdf, edges_gdf)
        
        # simplify the graph                           
        nodes_gdf, edges_gdf = simplify_graph(nodes_gdf, edges_gdf) 
    
    nodes_gdf['x'], nodes_gdf['y'] = list(zip(*[(r.coords[0][0], r.coords[0][1]) for r in nodes_gdf.geometry]))
    edges_gdf.drop(['coords', 'tmp'], axis = 1, inplace = True, errors = 'ignore') # remove temporary columns
    nodes_gdf['nodeID'] = nodes_gdf.nodeID.astype(int)
    edges_gdf = correct_edges(nodes_gdf, edges_gdf) # correct edges coordinates
    edges_gdf['length'] = edges_gdf['geometry'].length
    edges_gdf.set_index('edgeID', drop = False, inplace = True, append = False)
    edges_gdf.index.name = None
    
    return nodes_gdf, edges_gdf


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
    

def fix_network_topology(nodes_gdf, edges_gdf):

    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()
    edges_gdf.drop(['u', v'], inplace = True, axis =1)
    columns = ['bridge', 'tunnel']
    bridge_tunnels = edges_gdf.empty().copy()
    for column in columns:
        if column in edges_gdf.columns:
            edges_gdf[column].fillna(0, inplace=True)
            bridge_tunnels = bridge_tunnels.append(edges_gdf[edges_gdf[column] != 0])
    
    # Create a Graph object
    G = nx.Graph()
    
    # Add nodes to the graph
    for idx, row in edges_gdf.iterrows():
        G.add_node(idx, geometry=row['geometry'])
    
    # Add edges to the graph
    for idx, row in edges_gdf.iterrows():
        for nbr in row['geometry'].coords:
            G.add_edge(idx, nbr, attr_dict=row.to_dict())
    
    # Create spatial index for streets
    index = edges_gdf.sindex
    # Create prepared geometries for tunnels and bridges
    bridge_tunnels_prep = prep(bridge_tunnels.geometry.unary_union)

    # Split segments that intersect with other segments
    for idx, row in edges_gdf.iterrows():
        # Find all streets that intersect with the current street
        possible_matches_index = list(index.intersection(row.geometry.bounds))
        possible_matches = edges_gdf.iloc[possible_matches_index]
        for i, match in possible_matches.iterrows():
            # Check if the current street intersects with any other street
            if row.geometry.intersects(match.geometry) and not row.geometry.equals(match.geometry):
                new_line = row.geometry.intersection(match.geometry)
                # Check if the intersection point is not in a tunnel/bridge
                if not bridge_tunnels_prep.contains(new_line):
                    #split the edge
                    G.add_node(new_line, geometry=new_line)
                    G.add_edge(idx, new_line, attr_dict=row.to_dict())
                    G.add_edge(new_line, nbr, attr_dict=row.to_dict())
                    G.remove_edge(idx, nbr)
                    
    # Create a new GeoDataFrame with the fixed topology
    new_edges_gdf = nx.to_pandas_edgelist(G)
    new_edges_gdf.rename(columns: {'source': 'u', 'target' : 'v'}, inplace = True) 
    new_edges_gdf['geometry'] = new_edges_gdf.apply(lambda row: LineString([row['u'], row['v']]), axis=1)
    new_edges_gdf = GeoDataFrame(new_edges_gdf, geometry='geometry', crs = edges_gdf.crs)
    
    new_nodes_gdf = pandas.DataFrame.from_dict(G.nodes, orient='index')
    new_nodes_gdf['geometry'] = new_nodes_gdf.index.map(lambda row: nx.get_node_attributes(G, 'geometry')[row])
    new_nodes_gdf['nodeID'] = new_nodes_gdf.index
    new_nodes_gdf = GeoDataFrame(new_nodes_gdf, geometry='geometry', crs = edges_gdf.crs)

    return new_nodes_gdf, new_edges_gdf
    
    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()
    edges_gdf.drop(['u', v'], inplace = True, axis =1)
    columns = ['bridge', 'tunnel']
    bridge_tunnels = edges_gdf.empty().copy()
    for column in columns:
        if column in edges_gdf.columns:
            edges_gdf[column].fillna(0, inplace=True)
            bridge_tunnels = bridge_tunnels.append(edges_gdf[edges_gdf[column] != 0])


    old_edges_gdf = edges_gdf.copy()
    
    for row in old_edges_gdf.itertuples():
        
        line_geometry = old_edges_gdf.loc[row.Index].geometry
        tmp = old_edges_gdf[old_edges_gdf.geometry.intersects(line_geometry)].copy()
        tmp.drop(row.Index, axis = 0, inplace = True)
        
        union = tmp.unary_union
        intersections = line_geometry.intersection(union)
        if intersections.geom_type == 'Point': 
            intersections = [intersections]
        points = [i for i in intersections if i.geom_type == 'Point']
        new_collection = []

        for p in points:
            if (p.coords[0] == line_geometry.coords[0]) | (p.coords[0] == line_geometry.coords[-1]): 
                pass # disregarding the ones which lie on the line's u-v nodes
            else: 
                new_collection.append(p) # only checking the others
        
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
    return 

    return nodes_gdf, edges_gdf






















def remove_islands(nodes_gdf, edges_gdf, nodeID):
    Ng = ci.graph_fromGDF(nodes_gdf, edges_gdf, nodeID)
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
