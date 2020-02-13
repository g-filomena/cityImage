import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import networkx as nx, matplotlib.cm as cm, pandas as pd, numpy as np, geopandas as gpd
from shapely.geometry import Point, LineString

pd.set_option('precision', 10)

from .graph import*
from .utilities import *

"""
This set of functions is designed for cleaning street network's GeoDataFrame (nodes, edges), by taking care of dead_ends, duplicate geometries, same vertexes edges and so on.

"""

def duplicate_nodes(nodes_gdf, edges_gdf):

    """
    The function checks the existencce of duplicate nodes through the network, on the basis of geometry
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
   
    Returns
    -------
    tuple of GeoDataFrames
    """
    
    # the index of nodes_gdf has to be nodeID
    if list(nodes_gdf.index.values) != list(nodes_gdf.nodeID.values): nodes_gdf.index = nodes_gdf.nodeID
    nodes_gdf, edges_gdf =  nodes_gdf.copy(), edges_gdf.copy()
    
    # detecting duplicate geometries
    G = nodes_gdf["geometry"].apply(lambda geom: geom.wkb)
    new_nodes = nodes_gdf.loc[G.drop_duplicates().index]
    
    # assign univocal nodeID to edges which have 'u' or 'v' referring to duplicate nodes
    to_edit = list(set(nodes_gdf.index.values.tolist()) - set((new_nodes.index.values.tolist())))
    
    if len(to_edit) == 0: return(nodes_gdf, edges_gdf) 
    else:
        # readjusting edges' nodes too, accordingly
        for node in to_edit:
            geo = nodes_gdf.loc[node].geometry
            tmp = new_nodes[new_nodes.geometry == geo]
            index = tmp.iloc[0].nodeID
            
            # assigning the unique index to edges
            edges_gdf.loc[edges_gdf.u == node,'u'] = index
            edges_gdf.loc[edges_gdf.v == node,'v'] = index
        
    return new_nodes, edges_gdf
    

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
    tuple of GeoDataFrames
    """
    
    nodes_gdf =  nodes_gdf.copy()
    edges_gdf = edges_gdf.copy()
    dd_u = dict(edges_gdf['u'].value_counts())
    dd_v = dict(edges_gdf['v'].value_counts())
    
    dd = {k: dd_u.get(k, 0) + dd_v.get(k, 0) for k in set(dd_u) | set(dd_v)}
    to_delete = {k: v for k, v in dd.items() if v == 1}
    if len(to_delete) == 0: return(nodes_gdf, edges_gdf)
    
    # removing edges and nodes
    to_delete_list = list(to_delete.keys())
    nodes_gdf.drop(to_delete_list, axis = 0 , inplace = True)
    edges_gdf = edges_gdf[(~edges_gdf['u'].isin(to_delete_list)) & (~edges_gdf['v'].isin(to_delete_list))]

    return nodes_gdf, edges_gdf

def is_nodes_simplified(edges_gdf):

    """
    The function checks the presence of pseudo-junctions, by using the edges_gdf geodataframe.
     
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        street segments
   
    Returns
    -------
    boolean
    """
    
    simplified = True
    dd_u = dict(edges_gdf['u'].value_counts())
    dd_v = dict(edges_gdf['v'].value_counts())
    dd = {k: dd_u.get(k, 0) + dd_v.get(k, 0) for k in set(dd_u) | set(dd_v)}
    to_edit = {k: v for k, v in dd.items() if v == 2}
    if len(to_edit) == 0: return simplified
    simplified = False
            
    return simplified

def is_edges_simplified(edges_gdf):

    """
    The function checks the presence of possible duplicate geometries in the edges_gdf geodataframe.
     
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        street segments
   
    Returns
    -------
    boolean
    """
    
    simplified = True 
    
    edges_gdf['code'] = None
    edges_gdf['code'][edges_gdf['v'] >= edges_gdf['u']] = edges_gdf.u.astype(str)+"-"+edges_gdf.v.astype(str)
    edges_gdf['code'][edges_gdf['v'] < edges_gdf['u']] = edges_gdf.v.astype(str)+"-"+edges_gdf.u.astype(str)
    dd = dict(edges_gdf['code'].value_counts())
    dd = {k: v for k, v in dd.items() if v > 1}
    if len(dd) > 0: simplified = False
    
    return simplified

def simplify_graph(nodes_gdf, edges_gdf):

    """
    The function identify pseudo-nodes, namely nodes that represent intersection between only 2 segments.
    The segments are merged and the node is removed from the nodes_gdf geodataframe.
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
   
    Returns
    -------
    tuple of GeoDataFrames
    """
    
    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()
    
    # keeping only one item per node and counting its "appearances"
    dd_u, dd_v = dict(edges_gdf['u'].value_counts()), dict(edges_gdf['v'].value_counts())
    dd = {k: dd_u.get(k, 0) + dd_v.get(k, 0) for k in set(dd_u) | set(dd_v)}
    
    # editing the ones which only connect two edges
    to_edit = {k: v for k, v in dd.items() if v == 2}
    if len(to_edit) == 0: return(nodes_gdf, edges_gdf)
    to_edit_list = list(to_edit.keys())
    
    for nodeID in to_edit_list:

        tmp = edges_gdf[(edges_gdf['u'] == nodeID) | (edges_gdf['v'] == nodeID)].copy()    
        if len(tmp) == 0: 
            nodes_gdf.drop(nodeID, axis = 0, inplace = True)
            continue
        if len(tmp) == 1: continue # possible dead end
        
        # dead end identified
        index_first, index_second = tmp.iloc[0].edgeID, tmp.iloc[1].edgeID # first segment index
        
        # Identifying the relationship between the two segments.
        # New node_u and node_v are assigned accordingly. A list of ordered coordinates is obtained for 
        # merging the geometries. 4 conditions:
        if (tmp.iloc[0]['u'] == tmp.iloc[1]['u']):  
            edges_gdf.at[index_first,'u'] = edges_gdf.loc[index_first]['v']
            edges_gdf.at[index_first,'v'] = edges_gdf.loc[index_second]['v']
            line_coordsA, line_coordsB = list(tmp.iloc[0]['geometry'].coords), list(tmp.iloc[1]['geometry'].coords)    
            line_coordsA.reverse()
        
        elif (tmp.iloc[0]['u'] == tmp.iloc[1]['v']): 
            edges_gdf.at[index_first,'u'] = edges_gdf.loc[index_second]['u']
            line_coordsA, line_coordsB = list(tmp.iloc[1]['geometry'].coords), list(tmp.iloc[0]['geometry'].coords)               
        
        elif (tmp.iloc[0]['v'] == tmp.iloc[1]['u']): 
            edges_gdf.at[index_first,'v'] = edges_gdf.loc[index_second]['v']
            line_coordsA, line_coordsB = list(tmp.iloc[0]['geometry'].coords), list(tmp.iloc[1]['geometry'].coords)  
        
        else: # (tmp.iloc[0]['v'] == tmp.iloc[1]['v']) 
            edges_gdf.at[index_first,'v'] = edges_gdf.loc[index_second]['u']
            line_coordsA, line_coordsB = list(tmp.iloc[0]['geometry'].coords), list(tmp.iloc[1]['geometry'].coords)
            line_coordsB.reverse()

        # checking that none edges with node_u == node_v have been created, if yes: drop them
        if edges_gdf.loc[index_first].u == edges_gdf.loc[index_first].v: 
            edges_gdf.drop([index_first, index_second], axis = 0, inplace = True)
            nodes_gdf.drop(nodeID, axis = 0, inplace = True)
            continue
        
        # obtaining coordinates-list in consistent order and merging
        new_line = line_coordsA + line_coordsB
        merged_line = LineString([coor for coor in new_line]) 
        edges_gdf.at[index_first, 'geometry'] = merged_line
        if 'highway' in edges_gdf.columns: 
            if edges_gdf.loc[index_second]['pedestrian'] == True: edges_gdf.at[index_first, 'pedestrian'] = 1        
        # dropping the second segment, as the new geometry was assigned to the first edge
        edges_gdf.drop(index_second, axis = 0, inplace = True)
        nodes_gdf.drop(nodeID, axis = 0, inplace = True)
    
    return nodes_gdf, edges_gdf


def clean_network(nodes_gdf, edges_gdf, dead_ends = False, remove_disconnected_islands = True, same_uv_edges = True, self_loops = False):

    """
    It calls a series of functions (see above) to clean and remove dubplicate geometries or possible parallel short edges.
    It handles:
        - pseudo-nodes;
        - duplicate-geometries (nodes and edges);
        - disconnected islands (optional)
        - edges with different geometry (optional);
        - dead-ends (optional);
           
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
        if true, it considers as duplicates couple of edges with same vertexes but different geometry. If one of the edges is 20% longer than the other, the longer is deleted; 
        otherwise a center line is built to replace the same_uv_edges.
   
    Returns
    -------
    tuple of GeoDataFrames
    """
    
    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()       
    nodes_gdf.set_index('nodeID', drop = False, inplace = True, append = False)
    nodes_gdf.index.name = None
    edges_gdf.set_index('edgeID', drop = False, inplace = True, append = False)
    edges_gdf.index.name = None
    

    nodes_gdf['x'], nodes_gdf['y'] = list(zip(*[(r.coords[0][0], r.coords[0][1]) for r in nodes_gdf.geometry]))
    edges_gdf.sort_index(inplace = True)  
    edges_gdf['code'], edges_gdf['coords'] = None, None
    ix_geo = edges_gdf.columns.get_loc("geometry")+1
    
    if 'highway' in edges_gdf.columns:
        edges_gdf['pedestrian'] = 0
        to_remove = ['elevator']  
        edges_gdf = edges_gdf[~edges_gdf.highway.isin(to_remove)]
        pedestrian = ['footway', 'pedestrian', 'living_street', 'path']
        edges_gdf['pedestrian'][edges_gdf.highway.isin(pedestrian)] = 1
    
    if dead_ends: nodes_gdf, edges_gdf = fix_dead_ends(nodes_gdf, edges_gdf)
    cycle = 0
    
    while ((not is_edges_simplified(edges_gdf)) | (not is_nodes_simplified(edges_gdf)) | (cycle == 0)):

        processed = []
        edges_gdf['length'] = edges_gdf['geometry'].length # recomputing length, to account for small changes
        cycle += 1
            
        nodes_gdf, edges_gdf = duplicate_nodes(nodes_gdf, edges_gdf)
        if self_loops: edges_gdf = edges_gdf[edges_gdf['u'] != edges_gdf['v']] #eliminate loops
        edges_gdf = edges_gdf[~((edges_gdf['u'] == edges_gdf['v']) & (edges_gdf['geometry'].length < 1.00))] #eliminate node-lines
        
        # Assigning codes based on the edge's nodes. 
        # The string is formulated putting the node with lower ID first, regardless it being 'u' or 'v'
        edges_gdf["code"] = np.where(edges_gdf['v'] >= edges_gdf['u'], edges_gdf.u.astype(str)+"-"+edges_gdf.v.astype(str), 
                              edges_gdf.v.astype(str)+"-"+edges_gdf.u.astype(str))
        
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
        if same_uv_edges:
            dd = dict(edges_gdf['code'].value_counts())
            dd = {k: v for k, v in dd.items() if v > 1} # keeping u-v combinations that appear more than once
            # iterate through possible duplicate edges for each specific combination of possible duplicates
            for key,_ in dd.items():
                tmp = edges_gdf[edges_gdf.code == key].copy()
                # sorting the temporary GDF by length, the shortest is then used as a term of comparison
                tmp.sort_values(['length'], ascending = True, inplace = True)
                line_geometry, ix_line = tmp.iloc[0]['geometry'], tmp.iloc[0].edgeID
                
                # iterate through all the other edges with same u-v nodes                                
                for connector in tmp.itertuples():
                    if connector.Index == ix_line: continue
                    line_geometry_connector, ix_line_connector = connector[ix_geo], connector.Index 
                    
                    # if this edge is x% longer than the edge identified in the outer loop, leave it
                    if (line_geometry_connector.length > (line_geometry.length * 1.20)): pass
                    # else draw a center-line, replace the geometry of the outer-loop segment with the CL, drop the segment of the inner-loop
                    else:
                        cl = center_line(line_geometry, line_geometry_connector)
                        edges_gdf.at[ix_line,'geometry'] = cl
                        if 'highway' in edges_gdf.columns:
                            if edges_gdf.loc[ix_line_connector]['pedestrian'] == 1: edges_gdf.at[ix_line,'pedestrian'] = 1 
                    edges_gdf.drop(ix_line_connector, axis = 0, inplace = True)
                        
        
        # only keep nodes which are actually used by the edges in the geodataframe
        to_keep = list(set(list(edges_gdf['u'].unique()) + list(edges_gdf['v'].unique())))
        nodes_gdf = nodes_gdf[nodes_gdf['nodeID'].isin(to_keep)]
        
        # simplify the graph                           
        nodes_gdf, edges_gdf = simplify_graph(nodes_gdf, edges_gdf)  
    
    nodes_gdf['x'], nodes_gdf['y'] = list(zip(*[(r.coords[0][0], r.coords[0][1]) for r in nodes_gdf.geometry]))
    edges_gdf.drop(['code', 'coords', 'tmp'], axis = 1, inplace = True, errors = 'ignore') # remove temporary columns
    nodes_gdf['nodeID'] = nodes_gdf.nodeID.astype(int)
    edges_gdf = correct_edges(nodes_gdf, edges_gdf) # correct edges coordinates
    edges_gdf['length'] = edges_gdf['geometry'].length
    
    # check if there are disconnected islands and remove nodes and edges belongings to these islands.
    if remove_disconnected_islands:
        Ng = graph_fromGDF(nodes_gdf, edges_gdf, 'nodeID')
        if not nx.is_connected(Ng):  
            largest_component = max(nx.connected_components(Ng), key=len)
            # Create a subgraph of Ng consisting only of this component:
            G = Ng.subgraph(largest_component)

            to_drop = [item for item in list(nodes_gdf.nodeID) if item not in list(G.nodes())]
            nodes_gdf.drop(to_drop, axis = 0 , inplace = True)
            edges_gdf = edges_gdf[(edges_gdf.u.isin(nodes_gdf.nodeID)) & (edges_gdf.v.isin(nodes_gdf.nodeID))]   

    edges_gdf.set_index('edgeID', drop = False, inplace = True, append = False)
    edges_gdf.index.name = None
    print("Done after ", cycle, " cleaning cycles")  
    
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
    GeoDataFrame
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
    LineString
    """
    
    line_coords = list(line_geometry.coords)
    line_coords[0] = (nodes_gdf.loc[u]['x'], nodes_gdf.loc[u]['y'])
    line_coords[-1] = (nodes_gdf.loc[v]['x'], nodes_gdf.loc[v]['y'])
    new_line_geometry = (LineString([coor for coor in line_coords]))
    
    return new_line_geometry

class Error(Exception):
    """Base class for other exceptions"""
    pass
class columnError(Error):
    """Raised when a column name is not provided"""
    pass

