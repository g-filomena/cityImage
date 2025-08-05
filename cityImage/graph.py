import osmnx as ox
import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx
import math
import ast
import functools

from math import sqrt
from collections import Counter
from shapely.geometry import Point, LineString
import warnings

from .angles import angle_line_geometries

pd.set_option("display.precision", 3)
pd.options.mode.chained_assignment = None
warnings.simplefilter(action="ignore", category=FutureWarning)


def graph_fromGDF(nodes_gdf, edges_gdf, nodeID_column = "nodeID"):
    """
    From two GeoDataFrames (nodes and edges), it creates a NetworkX undirected Graph.
       
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    nodeID_column : str, optional
        Column name for node unique identifiers in `nodes_gdf`. Default is 'nodeID'.
        
    Returns
    -------
    G: NetworkX.Graph
        The undirected street network graph.
    """
    nodes_gdf.set_index(nodeID_column, drop = False, inplace = True, append = False)
    nodes_gdf.index.name = None
    G = nx.Graph()   
    G.add_nodes_from(nodes_gdf.index)
    attributes = nodes_gdf.to_dict()
    
    # ignore fields containing values of type list
    for attribute_name in nodes_gdf.columns:
        if nodes_gdf[attribute_name].apply(lambda x: type(x) == list).any(): 
            continue    
        # only add this attribute to nodes which have a non-null value for it
        else: 
            attribute_values = {k: v for k, v in attributes[attribute_name].items() if pd.notnull(v)}
        nx.set_node_attributes(G, name=attribute_name, values=attribute_values)

    # add the edges and attributes that are not u, v (as they're added separately) or null
    for _, row in edges_gdf.iterrows():
        attrs = {}
        for label, value in row.items():
            if (label not in ['u', 'v']) and (isinstance(value, list) or pd.notnull(value)):  
                attrs[label] = value
        G.add_edge(row['u'], row['v'], **attrs)
    
    return G


def multiGraph_fromGDF(nodes_gdf, edges_gdf, nodeID_column):
    """
    From two GeoDataFrames (nodes and edges), it creates a NetworkX.MultiGraph.
    
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    nodeID_column : str, optional
        Column name for node unique identifiers in `nodes_gdf`. Default is 'nodeID'.
    
    Returns
    -------
    G: NetworkX.MultiGraph
        The street network graph.
    """
    nodes_gdf.set_index(nodeID_column, drop = False, inplace = True, append = False)
    nodes_gdf.index.name = None
    
    Mg = nx.MultiGraph()   
    Mg.add_nodes_from(nodes_gdf.index)
    attributes = nodes_gdf.to_dict()
      
    for attribute_name in nodes_gdf.columns:
        if nodes_gdf[attribute_name].apply(lambda x: type(x) == list).any(): 
            continue 
        # only add this attribute to nodes which have a non-null value for it
        attribute_values = {k:v for k, v in attributes[attribute_name].items() if pd.notnull(v)}
        nx.set_node_attributes(Mg, name=attribute_name, values=attribute_values)

    # add the edges and attributes that are not u, v, key (as they're added separately) or null
    for row in edges_gdf.itertuples():
        attrs = {label: value for label, value in row._asdict().items() if (label not in ['u', 'v', 'key']) and (isinstance(value, list) or pd.notnull(value))}
        Mg.add_edge(row.u, row.v, key=row.key, **attrs)
      
    return Mg
    
def dual_gdf(nodes_gdf, edges_gdf, crs, oneway = False, angle = None):
    """
    It creates two dataframes that are later exploited to generate the dual graph of a street network. The nodes_dual gdf contains edges 
    centroids; the edges_dual gdf, instead, contains links between the street segment centroids. Those dual edges link real street segments 
    that share a junction. The centroids are stored with the original edge edgeID, while the dual edges are associated with several
    attributes computed on the original street segments (distance between centroids, deflection angle).
    
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        The nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    crs : str, or pyproj.CRS
        Coordinate Reference System for the output GeoDataFrames. Can be a string (e.g. 'EPSG:32633'),  or a pyproj.CRS object.
    oneway: boolean
        When true, the function takes into account the direction and therefore it may ignore certain links whereby vehichular movement is not allowed in 
        a certain direction.
    angle: string {'degree', 'radians'}
        It indicates how to express the angle of deflection.
        
    Returns
    -------
    nodes_dual, edges_dual: tuple of GeoDataFrames
        The dual nodes and edges GeoDataFrames.
    """
    nodes_gdf.set_index("nodeID", drop = False, inplace = True, append = False)
    nodes_gdf.index.name = None
    
    edges_gdf.set_index("edgeID", drop = False, inplace = True, append = False)
    edges_gdf.index.name = None
    
    # computing centroids                                       
    centroids_gdf = edges_gdf.copy()
    centroids_gdf['centroid'] = centroids_gdf['geometry'].centroid

     # find_intersecting segments and storing them in the centroids gdf
    # Create a new column 'intersecting' with a list of indexes of rows that have matching 'u' or 'v' values
    centroids_gdf['intersecting'] = centroids_gdf.apply(lambda row: list(centroids_gdf[(centroids_gdf['u'] == row['u'])|(centroids_gdf['u'] == row['v'])|(centroids_gdf['v'] == row['v'])|
                        (centroids_gdf['v'] == row['u'])].index), axis=1)


    # find_intersecting segments and storing them in the centroids gdf
    centroids_gdf['intersecting'] = centroids_gdf.apply(lambda row: list(centroids_gdf.loc[(centroids_gdf['u'] == row['u'])|(centroids_gdf['u'] == row['v'])|
                                                    (centroids_gdf['v'] == row['v'])|(centroids_gdf['v'] == row['u'])].index), axis=1)
    if oneway:
        centroids_gdf['intersecting'] = centroids_gdf.apply(lambda row: list(centroids_gdf.loc[(centroids_gdf['u'] == row['v']) | ((centroids_gdf['v'] == row['v']) & (centroids_gdf['oneway'] == 0))].index) 
                                                if row['oneway'] == 1 else list(centroids_gdf.loc[(centroids_gdf['u'] == row['v']) | ((centroids_gdf['v'] == row['v']) & (centroids_gdf['oneway'] == 0)) | 
                                                (centroids_gdf['u'] == row['u']) | ((centroids_gdf['v'] == row['u']) & (centroids_gdf['oneway'] == 0))].index), axis = 1)
    
    # creating vertexes representing street segments (centroids)
    centroids_data = centroids_gdf.drop(['geometry', 'centroid'], axis = 1)
    nodes_dual = gpd.GeoDataFrame(centroids_data, crs=crs, geometry=centroids_gdf['centroid'])
    nodes_dual['x'], nodes_dual['y'] = [x.coords.xy[0][0] for x in centroids_gdf['centroid']],[y.coords.xy[1][0] for y in centroids_gdf['centroid']]
    nodes_dual.index = nodes_dual.edgeID
    nodes_dual.index.name = None
        
    new_edges = []
    processed = set()

    for row in nodes_dual.itertuples():
        for intersecting in getattr(row, 'intersecting'):
            if (
                row.Index == intersecting or
                (row.Index, intersecting) in processed or
                (intersecting, row.Index) in processed
            ):
                continue
            length_intersecting = getattr(nodes_dual.loc[intersecting], 'length')
            distance = (getattr(row, 'length') + length_intersecting) / 2
            ls = LineString([
                getattr(row, 'geometry'),
                getattr(nodes_dual.loc[intersecting], 'geometry')
            ])
            new_edges.append({
                'u': row.Index,
                'v': intersecting,
                'geometry': ls,
                'length': distance
            })
            processed.add((row.Index, intersecting))

    edges_dual = pd.DataFrame(new_edges, columns=['u', 'v', 'geometry', 'length'])

    edges_dual = edges_dual.sort_index(axis=0)
    edges_dual = gpd.GeoDataFrame(edges_dual[['u', 'v', 'length']], crs=crs, geometry=edges_dual['geometry'])
    
    # setting angle values in degrees and radians
    if angle != 'radians':
        edges_dual['deg'] = edges_dual.apply(lambda row: angle_line_geometries(edges_gdf.loc[row['u']].geometry, edges_gdf.loc[row['v']].geometry, degree = True, 
                                            calculation_type = 'deflection'), axis = 1)
    else: 
        edges_dual['rad'] = edges_dual.apply(lambda row: angle_line_geometries(edges_gdf.loc[row['u']].geometry, edges_gdf.loc[row['v']].geometry, degree = False, 
                                            calculation_type = 'deflection'), axis = 1)
    return nodes_dual, edges_dual

def dual_graph_fromGDF(nodes_dual, edges_dual):
    """
    The function generates a NetworkX.Graph from dual-nodes and -edges GeoDataFrames.
            
    Parameters
    ----------
    nodes_dual: Point GeoDataFrame
        The GeoDataFrame of the dual nodes, namely the street segments' centroids.
    edges_dual: LineString GeoDataFrame
        The GeoDataFrame of the dual edges, namely the links between street segments' centroids.
        
    Returns
    -------
    Dg: NetworkX.Graph
        The dual graph of the street network.
    """
    nodes_dual.set_index('edgeID', drop = False, inplace = True, append = False)
    nodes_dual.index.name = None
    edges_dual.u, edges_dual.v = edges_dual.u.astype(int), edges_dual.v.astype(int)
    
    Dg = nx.Graph()   
    Dg.add_nodes_from(nodes_dual.index)
    attributes = nodes_dual.to_dict()
       
    for attribute_name in nodes_dual.columns:
        # only add this attribute to nodes which have a non-null value for it
        if nodes_dual[attribute_name].apply(lambda x: type(x) == list).any(): 
            continue
        attribute_values = {k:v for k, v in attributes[attribute_name].items() if pd.notnull(v)}
        nx.set_node_attributes(Dg, name=attribute_name, values=attribute_values)

    # add the edges and attributes that are not u, v, key (as they're added separately) or null
    for _, row in edges_dual.iterrows():
        attrs = {label:value for label, value in row.items() if (label not in ['u', 'v']) and (isinstance(value, list) or pd.notnull(value))}
        Dg.add_edge(row['u'], row['v'], **attrs)

    return Dg

def dual_id_dict(dict_values, graph, node_attribute):
    """
    It can be used when one deals with a dual graph and wants to link analyses conducted on this representation to 
    the primal graph. For instance, it takes the dictionary containing the betweennes-centrality values of the
    nodes in the dual graph, and associates these variables to the corresponding edgeID.
    
    Parameters
    ----------
    dict_values: dictionary 
        It should be in the form {nodeID: value} where values is a measure that has been computed on the graph, for example.
    graph: networkx graph
        The graph that was used to compute or to assign values to nodes or edges.
    node_attribute: string
        The attribute of the node to link to the edges GeoDataFrame.
    
    Returns
    -------
    ed_dict: dictionary
        A dictionary where each item consists of a edgeID (key) and centrality values (for example) or other attributes (values).
    """
    ed_list = [(graph.nodes[node][node_attribute], value) for node, value in dict_values.items()]
    return dict(ed_list)

def nodes_degree(edges_gdf):
    """
    It returns a dictionary where keys are nodes identifier (e.g. "nodeID") and values their degree.
    
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    
    Returns
    -------
    dd: dictionary
        A dictionary where each item consists of a nodeID (key) and degree values (values).
    """
    dd = edges_gdf[['u','v']].stack().value_counts().to_dict()
    return dd 

def from_nx_to_gdf(graph, crs):
    """
    Converts a NetworkX graph with geometry attributes into two GeoDataFrames: one for nodes and one for edges.

    Each node and edge in the input graph must have a 'geometry' attribute containing a valid shapely geometry
    (e.g., Point for nodes, LineString for edges).

    Parameters
    ----------
    graph : networkx.Graph or networkx.DiGraph
        The input NetworkX graph. Each node and edge must have a 'geometry' key in its attribute dictionary.
    crs : str, or pyproj.CRS
        Coordinate Reference System for the output GeoDataFrames. Can be a string (e.g. 'EPSG:32633'), or a pyproj.CRS object.

    Returns
    -------
    nodes_gdf : GeoDataFrame
        GeoDataFrame of graph nodes
    edges_gdf : GeoDataFrame
        GeoDataFrame of graph edges
    """
    # Nodes GeoDataFrame
    nodes_gdf = gpd.GeoDataFrame(
        [
            {**data, "nodeID": node, "geometry": data["geometry"]} 
            for node, data in graph.nodes(data=True)
        ],
        crs=crs
    )
    
    # Edges GeoDataFrame
    edges_gdf = gpd.GeoDataFrame(
        [
            {**data, "u": u, "v": v, "geometry": data["geometry"]} 
            for u, v, data in graph.edges(data=True)
        ],
        crs=crs
    )
    
    return nodes_gdf, edges_gdf

