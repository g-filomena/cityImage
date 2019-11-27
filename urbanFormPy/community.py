import networkx as nx, pandas as pd, numpy as np, geopandas as gpd
import functools
import community

from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping, MultiLineString
from shapely.ops import cascaded_union, linemerge, nearest_points
pd.set_option("precision", 10)

from .utilities import *

def identify_regions(dual_graph, weight = None):
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

    # extraction of the best partitions
    partition = community.best_partition(dual_graph, weight=weight)
    dct = dual_id_dict(partition, dual_graph, "streetID")
    subdvisions.append(dct)

    # saving the data in a geodataframe
    partitions_df = dict_to_df(subdvisions, ["p_len", "p_rad", "p_no"])
    districts = pd.merge(edges_graphB, partitions_df, left_on = "streetID", right_index = True, how= "left")
    return districts