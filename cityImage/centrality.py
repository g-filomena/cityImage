import networkx as nx
import pandas as pd
import numpy as np
import geopandas as gpd
import functools
import math

from math import sqrt
from shapely.geometry import Point 
pd.set_option("precision", 10)

from .utilities import scaling_columnDF, dict_to_df
   
## Centrality functions ###############

def nodes_dict(G):
    """
    it creates a dictionary where keys represent the node ID, and items the coordinate tuples
    
    Parameters
    ----------
    G: NetworkX graph
    
    Returns
    -------
    dictionary
    """

    nodes_list = G.nodes()
    nodes_dict = {}

    for i, item in enumerate(nodes_list):
        cod = item
        x = nodes_list[item]["x"]
        y = nodes_list[item]["y"]
        nodes_dict[cod] = (x,y)
    
    return nodes_dict
    
def straightness_centrality(G, weight, normalized = True):
    """
    Straightness centrality compares the length of the path between two nodes with the straight line that links them capturing a 
    centrality that refers to ‘being more directly reachable’. (Porta, S., Crucitti, P. & Latora, V., 2006b. The Network Analysis Of Urban
    Streets: A Primal Approach. Environment and Planning B: Planning and Design, 33(5), pp.705–725.)
    
    Function readapted from: https://github.com/jcaillet/mca/blob/master/mca/centrality/overridden_nx_straightness.py

    Parameters
    ----------
    G: NetworkX graph
    weight: string, 
        edges weight
    normalized: boolean
    
    Returns
    -------
    dictionary
    """
    
    path_length = functools.partial(nx.single_source_dijkstra_path_length, weight = weight)
    nodes = G.nodes()
    straightness_centrality = {}

    # Initialize dictionary containing all the node id and coordinates
    coord_nodes = nodes_dict(G)

    for n in nodes:
        straightness = 0
        sp = path_length(G,n)

        if len(sp) > 0 and len(G) > 1:
            # start computing the sum of euclidean distances
            for target in sp:
                if n != target and target in coord_nodes:
                    network_dist = sp[target]
                    euclidean_dist = _euclidean_distance(*coord_nodes[n]+coord_nodes[target])
                    straightness = straightness + (euclidean_dist/network_dist)

            straightness_centrality[n] = straightness * (1.0/(len(G)-1.0))
            if normalized: 
                if len(sp)> 1:
                    s = (len(G) - 1.0) / (len(sp) - 1.0)
                    straightness_centrality[n] *= s
                else: straightness_centrality[n] = 0.0
        else:
            straightness_centrality[n] = 0.0

    return straightness_centrality
    
def _euclidean_distance(xs, ys, xt, yt):
    """ xs stands for x source and xt for x target """
    return sqrt((xs - xt)**2 + (ys - yt)**2)

def weight_nodes(nodes_gdf, services_gdf, G, field_name, radius = 400):
    """
    Given a nodes- and a services/points-geodataframes, the function assigns an attribute to nodes in the graph G (prevously derived from 
    nodes_gdf) based indeed on the amount of features in the services_gdf in a buffer around each node. 
    
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    services_gdf: Point GeoDataFrame
    G: NetworkX graph
    field_name: string
        attribute name
    radius: float
        distance around the node within looking for point features (services)
    
    Returns
    -------
    NetworkX graph
    """
    
    nodes_gdf[field_name] = None
    sindex = services_gdf.sindex
    
    nodes_gdf[field_name] = nodes_gdf.apply(lambda row: _services_around_node(row["geometry"], services_gdf, sindex, radius = radius), axis=1)
    for n in G.nodes(): 
        G.nodes[n][field_name] = nodes_gdf[field_name].loc[n]
    
    return G
    
def _services_around_node(node_geometry, services_gdf, services_gdf_sindex, radius):
    """
    It supports the weight_nodes function.
    
    Parameters
    ----------
    node_geometry: Point geometry
    services_gdf: Point GeoDataFrame
    services_gdf_sindex: Rtree Spatial Index
    radius: float
        distance around the node within looking for point features (services)
    
    Returns
    -------
    Integer value
    """

    buffer = node_geometry.buffer(radius)
    possible_matches_index = list(services_gdf_sindex.intersection(buffer.bounds))
    possible_matches = services_gdf.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(buffer)]
    weight = len(precise_matches)
        
    return weight


def reach_centrality(G, weight, radius, attribute):
    """
    The measure contemplates the assignment of attributes (e.g. number of activities, population, employees in an area) to nodes and
    accounts for opportunities that are reachable along the actual street network as perceived by pedestrians’. The reach centrality of a
    node j, indicates the number of other nodes reachable from i, at the shortest path distance of r, where nodes are rewarded with a
    score (indicated by "attribute") which indicates their importance. The function is readapted from: Sevtsuk, A. & Mekonnen, M., 2012.
    Urban Network Analysis: A New Toolbox For ArcGIS. Revue internationale de géomatique, 2, pp.287–305.

    Parameters
    ----------
    G: NetworkX graph
    weight: string
        edges weight
    radius: float
        distance from node within looking for other reachable nodes
    attribute: string
        node attribute used to compute reach centralily. It indicates the importance of the node 
        (e.g. number of services in 50mt buffer)
    
    Returns
    -------
    dictionary
    """
    
    path_length = functools.partial(nx.single_source_dijkstra_path_length, weight = weight)

    nodes = G.nodes()
    reach_centrality = {}
    coord_nodes = nodes_dict(G)

    for n in nodes:
        reach = 0
        sp = path_length(G, n)
        sp_radium = dict((k, v) for k, v in sp.items() if v <= radius)
        
        if len(sp_radium) > 0 and len(G) > 1:
            
            for target in sp_radium:
                if (n != target) & (target in coord_nodes):
                    weight_target = G.nodes[target][attribute]
                    reach = reach + weight_target
            reach_centrality[n] = reach
        else: reach_centrality[n]=0.0

    return reach_centrality
    
    
def rescale_centrality(nodes_gdf, measure, radius = 400):
    """
    The measure rescales precomputed centrality values within a certain radius around each node.
    Indicate the value to rescale through the parameter "measure" (column name).

    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    measure: string
    radius: float
        distance from node within wich rescaling the betweenness centrality value
    
    Returns
    -------
    dictionary
    """
    
    nodes_gdf = nodes_gdf.copy()
    if measure not in nodes_gdf.columns: 
        raise columnError("The column name provided was not found in the nodes GeoDataFrame")
    spatial_index = nodes_gdf.sindex # spatial index
    nodes_gdf[measure+"_"+str(radius)] = nodes_gdf.apply(lambda row: _rescale_centrality(row.Index, nodes_gdf, sindex, radius = radius, measure = measure), axis=1)
        
    return nodes_gdf

def _rescale_centrality(nodeID, nodes_gdf, nodes_gdf_sindex, radius, measure):
    """
    It supports the rescale_centrality function.
    
    Parameters
    ----------
    nodeID: integer
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    nodes_gdf_sindex = Rtree Spatial Index
    radius: float
        distance around the node within looking for other nodes
    measure: string
    
    Returns
    -------
    float value
    """
    
    node_geometry = nodes_gdf.loc[nodeID].geometry
    buffer = node_geometry.buffer(radius)
    possible_matches_index = list(sindex.intersection(buffer.bounds))
    possible_matches = nodes_gdf.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(buffer)]
    scaling_columnDF(precise_matches, measure) 
        
    return precise_matches[measure+"_sc"].loc[nodeID]

def centrality(G, nodes_gdf, measure, weight, normalized = False):
    """"
    The function computes several node centrality measures.
      
    Parameters
    ----------
    G: Networkx graph
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    measure: string
    weight: string
        edges weight
    normalized: boolean
    
    Returns
    -------
    dictionary
    """     
    
    if measure == "betweenness_centrality": 
        c = nx.betweenness_centrality(G, weight = weight, normalized=normalized)
    elif measure == "straightness_centrality": 
        c = straightness_centrality(G, weight = weight, normalized=normalized)
    elif measure == "closeness_centrality": 
        c = nx.closeness_centrality(G, weight = weight, normalized=normalized)
    elif measure == "information_centrality": 
        c = nx.current_flow_betweenness_centrality(G, weight = weight, solver ="lu", normalized=normalized) 
    else:
        raise nameError("The name provided is not a valid centrality name associated with a function")
    
    return c
    
# TODO: optimise code above  
# def local_centrality(G, measure, weight, radius = 400, normalized = False):
    # """
    # The function computes betweenness centrality at the local level.
      
    # Parameters
    # ----------
    # G: Networkx graph
    # weight: string, edges weight
    # radius: float, distance from node, within local betweenness is computed  
    
    # Returns
    # -------
    # dictionary
    # """
           
    # path_length = functools.partial(nx.single_source_dijkstra_path_length, weight = weight)
    # nodes = G.nodes()
    # cm = {}
    # coord_nodes = nodes_dict(G)

    # for n in nodes:
        # G_small = nx.ego_graph(G, n, radius=radius, distance="weight")
        # if measure == "betweenness_centrality": c = nx.betweenness_centrality(G_small, weight = weight, normalized=normalized)[n]
        # elif measure == "straightness_centrality": c = straightness_centrality(G_small, weight = weight, normalized=normalized)[n]
        # elif measure == "closeness_centrality": c = nx.closeness_centrality(G_small, weight = weight, normalized=normalized)[n]
        # cm[n] = c
    
    # return cm
      
    
def append_edges_metrics(edges_gdf, G, dicts, column_names):
    """"
    The function attaches edges centrality value at the edges_gdf GeoDataFrame.
      
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
    G: Networkx graph
    dicts: list
        list of dictionaries resulting from centrality measures
    column_names: list
        list of strings with desired column names
    
    Returns
    -------
    dictionary
    """    
    
    edgesID = {}
    for i, e in G.edges():
        edgesID[(i,e)] = G[i][e]['edgeID']
    missing_values = [item for item in list(edges_gdf.index) if item not in list(edgesID.values())]
    
    dicts.append(edgesID)
    column_names.append("edgeID")
    
    tmp = dict_to_df(dicts, column_names)
    tmp.edgeID = tmp.edgeID.astype(int)
    edges_gdf = pd.merge(edges_gdf, tmp, on = 'edgeID', how = 'left')
    edges_gdf.index = edges_gdf.edgeID
    edges_gdf.index.name = None
    
    # handling possible missing values (happens with self-loops)
    for metric in column_names:
        if metric == "edgeID": 
            continue
        for i in missing_values: 
            edges_gdf.at[i, metric] = 0.0
    
    return edges_gdf
    

    
class Error(Exception):
    """Base class for other exceptions"""
class columnError(Error):
    """Raised when a column name is not provided"""

class nameError(Error):
    """Raised when a not supported or not existing centrality name is input"""

    
