import networkx as nx
import pandas as pd
import numpy as np
import geopandas as gpd
import functools
import math
import igraph as ig

from math import sqrt
from shapely.geometry import Point 
pd.set_option("display.precision", 3)

from .utilities import dict_to_df

def nodes_dict(ig_graph):
    """Create a dictionary of node indices and their coordinates from igraph graph.

    Parameters
    ----------
    ig_graph : igraph.Graph
        The igraph Graph object.

    Returns
    -------
    dict
        Dictionary with node indices as keys and tuples of (x, y) coordinates as values.
    """
    return {v.index: (v['x'], v['y']) for v in ig_graph.vs}
    
def _assign_attributes_from_nx_to_igraph(nx_graph, ig_graph):
    """
    Assign attributes from a NetworkX graph to an iGraph graph.

    This function transfers all node attributes from a NetworkX graph to the corresponding nodes in an iGraph graph. 
    It assumes that both graphs have the same set of nodes in the same order.

    Parameters
    ----------
    nx_graph : networkx.Graph
        The input NetworkX graph from which to copy attributes.
    ig_graph : igraph.Graph
        The target iGraph graph to which attributes will be copied.
    """
    original_ids = list(nx_graph.nodes)
    for idx, node in enumerate(ig_graph.vs):
        nx_node_attrs = nx_graph.nodes[original_ids[idx]]
        for attr, value in nx_node_attrs.items():
            node[attr] = value   
     
def _euclidean_distance(xs, ys, xt, yt):
    """ xs stands for x source and xt for x target """
    return sqrt((xs - xt)**2 + (ys - yt)**2)           
            
def calculate_centrality(nx_graph, measure='betweenness', weight='weight', radius=None, attribute=None):
    """
    Convert a NetworkX graph to an igraph graph and calculate centrality measures.

    Parameters
    ----------
    graph: networkx.Graph
        The input NetworkX graph.
    measure: str
        The type of centrality measure to calculate ('betweenness', 'closeness', 'straightness', 'reach').
    weight: str
        The edge attribute to be used as weights for weighted centrality measures.
    radius: float, optional
        The radius for reach centrality.
    attribute: str, optional
        The node attribute for reach centrality.

    Returns
    -------
    centrality_dict: dict
        A dictionary with the original NetworkX node IDs as keys and the centrality values as values.
    """
    # Create a mapping from original NetworkX node IDs to indices
    original_ids = list(nx_graph.nodes)
    id_to_index = {original_id: index for index, original_id in enumerate(original_ids)}
    
    # Convert NetworkX graph to igraph graph
    edges = [(id_to_index[u], id_to_index[v]) for u, v in nx_graph.edges()]
    weights = [nx_graph[u][v][weight] for u, v in nx_graph.edges()]

    ig_graph = ig.Graph(edges=edges)
    ig_graph.es['weight'] = weights

    # Assign attributes to igraph vertices
    _assign_attributes_from_nx_to_igraph(nx_graph, ig_graph)

    # Calculate the specified centrality measure
    if measure == 'betweenness':
        centrality_values = ig_graph.betweenness(weights='weight')
    elif measure == 'closeness':
        centrality_values = ig_graph.closeness(weights='weight')
    elif measure == 'straightness':
        centrality_values = straightness_centrality(ig_graph, weight='weight')
    elif measure == 'reach':
        if radius is None or attribute is None:
            raise ValueError("For reach centrality, both radius and attribute parameters must be provided")
        centrality_values = reach_centrality(ig_graph, weight='weight', radius=radius, attribute=attribute)
    else:
        raise ValueError(f"Unsupported centrality type: {measure}")

    # Map centrality values back to original NetworkX node IDs
    centrality_dict = {original_ids[i]: centrality_values[i] for i in range(len(original_ids))}

    return centrality_dict

def reach_centrality(ig_graph, weight, radius, attribute):
    """
    Calculates the reach centrality of each node in the graph G based on the attribute of the reachable nodes
    within a given radius.

    Parameters
    ----------
    ig_graph: igraph Graph
        The street network graph.
    weight: str
        The street segments' weight (e.g. distance).
    radius: float
        Distance from node within looking for other reachable nodes
    attribute: str
        Node attribute used to compute reach centralily. It indicates the importance of the node 
        (e.g. number of services in 50mt buffer - name of a column in the nodes_gdf GeoDataFrame).
    
    Returns
    -------
    reach_centrality: dict
        A dictionary where each item consists of a node (key) and the centrality value (value).
    """
    
    n_nodes = ig_graph.vcount()
    reach_centrality = {}

    # Compute all shortest paths from all nodes
    shortest_paths = ig_graph.distances(weights=weight)

    for node in range(n_nodes):
        reach = 0
        sp = shortest_paths[node]
        # Filter paths based on radius and compute reach centrality
        reach = sum(ig_graph.vs[target][attribute] for target, dist in enumerate(sp) if dist != float('inf') and dist <= radius and target != node)
        reach_centrality[node] = reach

    return reach_centrality

def straightness_centrality(ig_graph, weight, normalized=True):
    """
    Calculate the straightness centrality for each node in the graph.

    Parameters
    ----------
    ig_graph : igraph.Graph
        The igraph Graph object.
    weight : str
        The edge attribute to be used as weight.
    normalized : bool, optional
        Whether to normalize the straightness centrality values (default is True).

    Returns
    -------
    dict
        Dictionary of node indices and their straightness centrality values.
    """
    n_nodes = ig_graph.vcount()
    coord_nodes = nodes_dict(ig_graph)
    straightness_centrality = {}

    # Compute shortest paths for all pairs
    shortest_paths = ig_graph.distances(weights=weight)

    for node in range(n_nodes):
        straightness = 0
        sp = shortest_paths[node]

        if len(sp) > 0 and n_nodes > 1:
            # Compute the sum of euclidean distances
            for target, network_dist in enumerate(sp):
                if node != target and network_dist < float('inf'):
                    euclidean_dist = _euclidean_distance(*coord_nodes[node], *coord_nodes[target])
                    straightness += euclidean_dist / network_dist

            straightness_centrality[node] = straightness / (n_nodes - 1.0)
            if normalized:
                reachable_nodes = sum(1 for dist in sp if dist < float('inf')) - 1
                if reachable_nodes > 0:
                    s = (n_nodes - 1.0) / reachable_nodes
                    straightness_centrality[node] *= s
                else:
                    straightness_centrality[node] = 0.0
        else:
            straightness_centrality[node] = 0.0

    return straightness_centrality
    
def weight_nodes(nodes_gdf, services_gdf, nx_graph, field_name, radius = 400):
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
        nodes (junctions) GeoDataFrame.
    services_gdf: Point GeoDataFrame
        
    nx_graph: Networkx graph
        The street network graph.
    field_name: string
        The name of the nodes' attribute.
    radius: float
        Distance around the node within looking for point features (services).
    
    Returns
    -------
    nx_graph: Networkx graph
        The updated street network graph.
    
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
        nx_graph.nodes[n][field_name] = weight

    return nx_graph    
    
def append_edges_metrics(edges_gdf, G, dicts, column_names):
    """"
    The function attaches edges centrality values at the edges_gdf GeoDataFrame.
      
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        The street segments GeoDataFrame.
    G: Networkx graph
        The street network graph.
    dicts: list
        The list of dictionaries resulting from centrality measures.
    column_names: list
        The list of strings with the desired column names for the attributes to be attached.
    
    Returns
    -------
    edges_gdf: LineString GeoDataFrame
        The updated street segments GeoDataFrame.
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

    
