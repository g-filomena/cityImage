import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import osmnx as ox
import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx

import math
from math import sqrt
import ast
import functools

from collections import Counter
from shapely.geometry import Point, LineString
pd.set_option("display.precision", 3)
pd.options.mode.chained_assignment = None

from .utilities import *
from .angles import angle_line_geometries

## Obtaining graphs ###############

def graph_fromGDF(nodes_gdf, edges_gdf, nodeID = "nodeID"):
    """
    From two GeoDataFrames (nodes and edges), it creates a NetworkX undirected Graph.
       
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
    nodeID: str
        column name that indicates the node identifier column (if different from "nodeID")
        
    Returns
    -------
    G: NetworkX.Graph
        the undirected street network graph
    """

    nodes_gdf.set_index(nodeID, drop = False, inplace = True, append = False)
    nodes_gdf.index.name = None
    G = nx.Graph()   
    G.add_nodes_from(nodes_gdf.index)
    attributes = nodes_gdf.to_dict()
    
    # ignore fields containing values of type list
    to_ignore = nodes_gdf.select_dtypes(include=['list']).columns
    
    for attribute_name in nodes_gdf.columns:
        if attribute_name in to_ignore: 
            continue    
        # only add this attribute to nodes which have a non-null value for it
        else: 
            attribute_values = {k: v for k, v in attributes[attribute_name].items() if pd.notnull(v)}
        nx.set_node_attributes(G, name=attribute_name, values=attribute_values)

    # add the edges and attributes that are not u, v (as they're added separately) or null
    for _, row in edges_gdf.iterrows():
        attrs = {}
        for label, value in row.iteritems():
            if (label not in ['u', 'v']) and (isinstance(value, list) or pd.notnull(value)):  
                attrs[label] = value
        G.add_edge(row['u'], row['v'], **attrs)
    
    return G


def multiGraph_fromGDF(nodes_gdf, edges_gdf, nodeIDcolumn):
    """
    From two GeoDataFrames (nodes and edges), it creates a NetworkX.MultiGraph.
    
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
    nodeIDcolumn: string
        column name that indicates the node identifier column.
    
    Returns
    -------
    G: NetworkX.MultiGraph
        the street network graph
    """
    
    nodes_gdf.set_index(nodeIDcolumn, drop = False, inplace = True, append = False)
    nodes_gdf.index.name = None
    
    Mg = nx.MultiGraph()   
    Mg.add_nodes_from(nodes_gdf.index)
    attributes = nodes_gdf.to_dict()
    to_ignore = nodes_gdf.select_dtypes(include=['list']).columns
    
    for attribute_name in nodes_gdf.columns:
        if attribute_name in to_ignore: 
            continue 
        # only add this attribute to nodes which have a non-null value for it
        attribute_values = {k:v for k, v in attributes[attribute_name].items() if pd.notnull(v)}
        nx.set_node_attributes(Mg, name=attribute_name, values=attribute_values)

    # add the edges and attributes that are not u, v, key (as they're added separately) or null
    for row in edges_gdf.itertuples():
        attrs = {label: value for label, value in row._asdict().items() if (label not in ['u', 'v', 'key']) and (isinstance(value, list) or pd.notnull(value))}
        Mg.add_edge(row.u, row.v, key=row.key, **attrs)
      
    return Mg
    
## Building geo-dataframes for dual graph representation ###############

def dual_gdf(nodes_gdf, edges_gdf, epsg, oneway = False, angle = None):
    """
    It creates two dataframes that are later exploited to generate the dual graph of a street network. The nodes_dual gdf contains edges 
    centroids; the edges_dual gdf, instead, contains links between the street segment centroids. Those dual edges link real street segments 
    that share a junction. The centroids are stored with the original edge edgeID, while the dual edges are associated with several
    attributes computed on the original street segments (distance between centroids, deflection angle).
    
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
    epsg: int
        epsg of the area considered 
    oneway: boolean
        if true, the function takes into account the direction and therefore it may ignore certain links whereby vehichular movement is not allowed in a certain direction
    angle: string {'degree', 'radians'}
        it indicates how to express the angle of deflection
        
    Returns
    -------
    nodes_dual, edges_dual: tuple of GeoDataFrames
        the dual nodes and edges GeoDataFrames
    """
    

    edges_gdf.index = edges_gdf.edgeID
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
    
    if epsg is not None: 
        centroids_gdf = centroids_gdf.to_crs({'init': 'epsg:' + str(epsg)})

    centroids_gdf = centroids_gdf.assign(x = [point.x for point in centroids_gdf['centroid']], y = [point.y for point in centroids_gdf['centroid']])
    centroids_gdf.index = centroids_gdf.edgeID
    centroids_gdf.index.name = None
        
    # creating fictious links between centroids
    edges_dual = pd.DataFrame(columns=['u','v', 'geometry', 'length'])

    # connecting nodes which represent street segments share a linked in the actual street network   
    processed = set()
    for row in nodes_dual.itertuples():                                           
        # intersecting segments:  # i is the edgeID                                      
        for intersecting in getattr(row, 'intersecting'):
            if ((row.Index == intersecting) | ((row.Index, intersecting) in processed) | ((intersecting, row.Index) in processed)): 
                    continue
            length_intersecting =  getattr(nodes_dual.loc[intersecting], 'length')
            distance = (getattr(row, 'length') + length_intersecting) / 2
            # from the first centroid to the centroid intersecting segment 
            ls = LineString([getattr(row, 'geometry'), getattr(nodes_dual.loc[intersecting], 'geometry')])
            edges_dual = edges_dual.append({'u': row.Index, 'v': intersecting, 'geometry': ls, 'length': distance}, ignore_index=True)
            processed.add((row.Index, intersecting))
            
    edges_dual = edges_dual.sort_index(axis=0)
    edges_dual = gpd.GeoDataFrame(edges_dual[['u', 'v', 'length']], crs=crs, geometry=edges_dual['geometry'])
    
    # setting angle values in degrees and radians
    if angle != 'radians':
        edges_dual['deg'] = edges_dual.apply(lambda row: angle_line_geometries(edges_gdf.loc[row['u']].geometry, edges_gdf.loc[row['v']].geometry, degree = True, measure = 'deflection'), axis = 1)
    else: 
        edges_dual['rad'] = edges_dual.apply(lambda row: angle_line_geometries(edges_gdf.loc[row['u']].geometry, edges_gdf.loc[row['v']].geometry, degree = False, measure = 'deflection'), axis = 1)
        
    return nodes_dual, edges_dual

def dual_graph_fromGDF(nodes_dual, edges_dual):
    """
    The function generates a NetworkX.Graph from dual-nodes and -edges GeoDataFrames.
            
    Parameters
    ----------
    nodes_dual: Point GeoDataFrame
        the GeoDataFrame of the dual nodes, namely the street segments' centroids
    edges_dual: LineString GeoDataFrame
        the GeoDataFrame of the dual edges, namely the links between street segments' centroids 
        
    Returns
    -------
    Dg: NetworkX.Graph
        the dual graph of the street network
    """
   
    nodes_dual.set_index('edgeID', drop = False, inplace = True, append = False)
    nodes_dual.index.name = None
    edges_dual.u, edges_dual.v  = edges_dual.u.astype(int), edges_dual.v.astype(int)
    
    Dg = nx.Graph()   
    Dg.add_nodes_from(nodes_dual.index)
    attributes = nodes_dual.to_dict()
    
    to_ignore = nodes_dual.select_dtypes(include=['list']).columns
    
    for attribute_name in nodes_dual.columns:
        # only add this attribute to nodes which have a non-null value for it
        if attribute_name in to_ignore: 
            continue
        attribute_values = {k:v for k, v in attributes[attribute_name].items() if pd.notnull(v)}
        nx.set_node_attributes(Dg, name=attribute_name, values=attribute_values)

    # add the edges and attributes that are not u, v, key (as they're added
    # separately) or null
    for _, row in edges_dual.iterrows():
        attrs = {label:value for label, value in row.iteritems() if (label not in ['u', 'v']) and (isinstance(value, list) or pd.notnull(value))}
        Dg.add_edge(row['u'], row['v'], **attrs)

    return Dg

def dual_id_dict(dict_values, G, node_attribute):
    """
    It can be used when one deals with a dual graph and wants to link analyses conducted on this representation to 
    the primal graph. For instance, it takes the dictionary containing the betweennes-centrality values of the
    nodes in the dual graph, and associates these variables to the corresponding edgeID.
    
    Parameters
    ----------
    dict_values: dictionary 
        it should be in the form {nodeID: value} where values is a measure that has been computed on the graph, for example
    G: networkx graph
        the graph that was used to compute or to assign values to nodes or edges
    node_attribute: string
        the attribute of the node to link to the edges GeoDataFrame
    
    Returns
    -------
    ed_dict: dictionary
        a dictionary where each item consists of a edgeID (key) and centrality values (for example) or other attributes (values)
    """
    
    ed_list = [(G.nodes[node][node_attribute], value) for node, value in dict_values.items()]
    return dict(ed_list)

def nodes_degree(edges_gdf):
    """
    It returns a dictionary where keys are nodes identifier (e.g. "nodeID") and values their degree.
    
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
    
    Returns
    -------
    dd: dictionary
        a dictionary where each item consists of a nodeID (key) and degree values (values)
    """

    dd_u = Counter(edges_gdf['u'])
    dd_v = Counter(edges_gdf['v'])
    dd = dict(dd_u + dd_v)
    return dd 