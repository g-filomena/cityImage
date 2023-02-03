import networkx as nx
import pandas as pd
import numpy as np
import geopandas as gpd
import functools
import math

from math import sqrt
from shapely.geometry import Point 
pd.set_option("display.precision", 3)

from .utilities import dict_to_df

def nodes_dict(G: nx.Graph) -> dict:
    """
    Creates a dictionary where keys represent the node ID and items represent the coordinates of the node in the form of a tuple (x, y).
    
    Parameters
    ----------
    G: A NetworkX Graph object.
        the graph
    
    Returns
    ----------
        a dictionary where each key is a node ID and each value is a tuple of coordinates (x, y).
    """
   
    return {item: (G.nodes[item]["x"], G.nodes[item]["y"]) for item in G.nodes()}
    
def straightness_centrality(G, weight, normalized = True):
    """
    Straightness centrality compares the length of the path between two nodes with the straight line that links them capturing a 
    centrality that refers to ‘being more directly reachable’. (Porta, S., Crucitti, P. & Latora, V., 2006b. The Network Analysis Of Urban
    Streets: A Primal Approach. Environment and Planning B: Planning and Design, 33(5), pp.705–725.)
    
    Function readapted from: https://github.com/jcaillet/mca/blob/master/mca/centrality/overridden_nx_straightness.py.

    Parameters
    ----------
    G: NetworkX Graph
        the graph
    weight: string 
        edges weight
    normalized: boolean
    
    Returns
    -------
    straightness_centrality: dict
        a dictionary where each item consists of a node (key) and the centrality value (value)
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

    Given a nodes' and a services/points' GeoDataFrame, the function assigns an attribute to nodes in the graph G (prevously derived from 
    nodes_gdf) based on the amount of features in the services_gdf in a buffer around each node. 
    
    The measure contemplates the assignment of attributes (e.g. number of activities, population, employees in an area) to nodes and
    accounts for opportunities that are reachable along the actual street network as perceived by pedestrians’. The reach centrality of a
    node j, indicates the number of other nodes reachable from i, at the shortest path distance of r, where nodes are rewarded with a
    score (indicated by "attribute") which indicates their importance. The function is readapted from: Sevtsuk, A. & Mekonnen, M., 2012.
    Urban Network Analysis: A New Toolbox For ArcGIS. Revue internationale de géomatique, 2, pp.287–305.
    
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    services_gdf: Point GeoDataFrame
    G: NetworkX Graph
    field_name: string
        the name of the nodes' attribute
    radius: float
        distance around the node within looking for point features (services)
    
    Returns
    -------
    G:
    
    """
    
    # create a spatial index for services_gdf
    sindex = services_gdf.sindex

    for n, node in nodes_gdf.iterrows():
        # get a buffer around the node
        buffer = node["geometry"].buffer(radius)
        # get the possible matches from services_gdf using the spatial index
        possible_matches_index = list(sindex.intersection(buffer.bounds))
        possible_matches = services_gdf.iloc[possible_matches_index]
        # get the precise matches using the buffer
        precise_matches = possible_matches[possible_matches.intersects(buffer)]
        weight = len(precise_matches)
        nodes_gdf.at[n, field_name] = weight
        G.nodes[n][field_name] = weight

    return G

def reach_centrality(G, weight, radius, attribute):
    """
    Calculates the reach centrality of each node in the graph G based on the attribute of the reachable nodes
    within a given radius.

    Parameters
    ----------
    G: NetworkX.Graph
        the street network graph
    weight: string
        the street segments' weight (e.g. distance)
    radius: float
        distance from node within looking for other reachable nodes
    attribute: string
        node attribute used to compute reach centralily. It indicates the importance of the node 
        (e.g. number of services in 50mt buffer - name of a column in the nodes_gdf GeoDataFrame)
    
    Returns
    -------
    reach_centrality: dict
        a dictionary where each item consists of a node (key) and the centrality value (value)
    """
    
    coord_nodes = set(n for n, d in G.nodes(data=True) if 'x' in d and 'y' in d)
    reach_centrality = {}
    
    for n in G.nodes():
        sp = nx.single_source_dijkstra_path_length(G, n, cutoff=radius, weight=weight)
        reach = sum(G.nodes[target][attribute] for target in sp if target != n and target in coord_nodes)
        reach_centrality[n] = reach
    
    return reach_centrality
    
def centrality(G, nodes_gdf, measure, weight, normalized = False):
    """"
    The function computes several node centrality measures.
      
    Parameters
    ----------
    G: Networkx graph
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    measure: string {"betweenness_centrality", "straightness_centrality", "closeness_centrality","information_centrality"}
        the type of centrality to be computed 
    weight: string
        the street segments' weight (e.g. distance)
    normalized: boolean
    
    Returns
    -------
    centrality: dict
        a dictionary where each item consists of a node (key) and the centrality value (value)
    """    
    measure_mapping = {
        "betweenness_centrality": (nx.betweenness_centrality, {"weight": weight, "normalized": normalized}),
        "straightness_centrality": (straightness_centrality, {"weight": weight, "normalized": normalized}),
        "closeness_centrality": (nx.closeness_centrality, {"distance": weight}),
        "information_centrality": (nx.current_flow_betweenness_centrality, {"weight": weight, "solver": "lu", "normalized": normalized})
    }
    if measure in measure_mapping:
        func, kwargs = measure_mapping[measure]
        centrality = func(G, **kwargs)
    else:
        raise ValueError("Invalid centrality measure provided. Options are 'betweenness_centrality', 'straightness_centrality', 'closeness_centrality', 'information_centrality'.")
    
    return centrality
    
    
def append_edges_metrics(edges_gdf, G, dicts, column_names):
    """"
    The function attaches edges centrality values at the edges_gdf GeoDataFrame.
      
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
    G: Networkx graph
        the street network graph
    dicts: list
        list of dictionaries resulting from centrality measures
    column_names: list
        list of strings with the desired column names for the attributes to be attached
    
    Returns
    -------
    edges_gdf: LineString GeoDataFrame
        the updated street segments GeoDataFrame
    """    
    
    edgesID = {(i,e): G[i][e]['edgeID'] for i, e in G.edges()}
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

    
