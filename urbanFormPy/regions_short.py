import networkx as nx, pandas as pd, numpy as np, geopandas as gpd
import functools
import community
import array
import numbers
import warnings


from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping, MultiLineString
from shapely.ops import cascaded_union, linemerge, nearest_points
pd.set_option("precision", 10)

from .graph import *
from .utilities import *

def identify_regions(dual_graph, edges_graph, weight = None, barrier_field = None):
    """
    Run the natural_roads function on an entire geodataframe of street segments.
    The geodataframes are supposed to be cleaned and can be obtained via the functions "get_fromOSM(place)" or "get_fromSHP(directory, 
    epsg)". The parameter tolerance indicates the maximux deflection allowed to consider two roads possible natural continuation.
    
    Parameters
    ----------
    dual_graph: GeoDataFrames
    weight = string
    
    Returns
    -------
    GeoDataFrames
    """

    subdvisions = []
    if weight == None: weight = "no_weight" #the function requires a string 
    # extraction of the best partitions
    partition = community.best_partition(dual_graph, weight=weight, barrier_field = barrier_field)
    dct = dual_id_dict(partition, dual_graph, "edgeID")
    subdvisions.append(dct)

    # saving the data in a geodataframe
    partitions_df = dict_to_df(subdvisions, ["p_"+weight])
    districts = pd.merge(edges_graph, partitions_df, left_on = "edgeID", right_index = True, how= "left")
    return districts

def assign_district_to_nodes(node_geometry, sidenx, edges_gdf):
    point = node_geometry
    n = point.buffer(20)
    possible_matches_index = list(sindex.intersection(n.bounds))
    pm = edges_gdf.iloc[possible_matches_index].copy()
    dist = distance_geometry_gdf(point, pm)
    return edges_graph.loc[dist[1]]['district']
    

 