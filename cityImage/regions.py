import networkx as nx
import pandas as pd
import numpy as np
import geopandas as gpd
import functools
import community
import array
import numbers
import warnings

from shapely.ops import polygonize_full, polygonize, unary_union
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping, MultiLineString
from shapely.ops import cascaded_union, linemerge, nearest_points
pd.set_option("precision", 10)

from .graph import *
from .utilities import *

def identify_regions(dual_graph, edges_gdf, weight = None):
    """
    It identifies regions in the street network, using the dual graph representation.
    The modularity optimisation technique is used to identify urban regions.
    
    Parameters
    ----------
    dual_graph: Networkx.Graph
        the dual graph of an urban area
    edges_gdf: LineString GeoDataFrame
        the (primal) street segments GeoDataFrame
    weight: string
        the edges' attribute to use when extracting the communities. If None is passed, only the topological relations influence the resulting communities.
    
    Returns
    -------
    regions: dict
        a dictionary where to each primal edgeID (key) corresponds a region code (value)
    """

    subdvisions = []
    if weight is None: 
        weight = 'topo' #the function requires a string 
    # extraction of the best partitions
    partition = community.best_partition(dual_graph, weight=weight)
    dct = dual_id_dict(partition, dual_graph, 'edgeID')
    subdvisions.append(dct)

    # saving the data in a GeoDataFrame
    partitions_df = dict_to_df(subdvisions, ['p_'+weight])
    regions = pd.merge(edges_gdf, partitions_df, left_on = 'edgeID', right_index = True, how= 'left')
    return regions
    
def identify_regions_primal(graph, nodes_graph, weight = None):
    """
    It identifies regions in the street network, using the primal graph representation.
    The modularity optimisation technique is used to identify urban regions.
    
    Parameters
    ----------
    graph: Networkx.Graph
        the primal graph of an urban area
    nodes_gdf: Point GeoDataFrame
        the nodes (junctions) GeoDataFrame

    Returns
    -------
    regions: dict
        a dictionary where to each nodeID (key) corresponds a region code (value)
    """

    subdvisions = []
    if weight is None: 
        weight = 'topo' #the function requires a string 
    # extraction of the best partitions
    partition = best_partition(graph, weight=weight)
    regions = nodes_graph.copy()
    regions['p_'+weight] = regions.nodeID.map(partition)
    return regions
  
                  
def polygonise_partition(edges_gdf, partition_field, method = None, buffer = 30):
    """
    Run the natural_roads function on an entire GeoDataFrame of street segments.
    The geodataframes are supposed to be cleaned and can be obtained via the functions "get_fromOSM(place)" or "get_fromSHP(directory, 
    epsg)". The parameter tolerance indicates the maximux deflection allowed to consider two roads possible natural continuation.
    
    Parameters
    ----------
    edges_gdf: GeoDataFrames
    partition_field = string
    method = None, 
    buffer = 30
    
    Returns
    -------
    GeoDataFrames
    """
    polygons = []
    partitionIDs = []
    d = {'geometry' : polygons, 'districtID' : partitionIDs}

    partitions = edges_gdf[partition_field].unique()
    for i in partitions:
        polygon =  polygonize_full(edges_gdf[edges_gdf[partition_field] == i].geometry.unary_union)
        polygon = unary_union(polygon).buffer(buffer)
        if method == 'convex_hull': 
            polygons.append(polygon.convex_hull)
        else: polygons.append(polygon)
        partitionIDs.append(i)

    df = pd.DataFrame(d)
    partitions_polygonised = gpd.GeoDataFrame(df, crs=edges_gdf.crs, geometry=df['geometry'])
    return partitions_polygonised
    
    
def districts_to_edges_from_nodes(edges_gdf, nodes_gdf, district_field):
    """
    Run the natural_roads function on an entire GeoDataFrame of street segments.
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
    ix_u = edges_gdf.columns.get_loc('u')+1  
    ix_v = edges_gdf.columns.get_loc('v')+1  
    
    edges_gdf = edges_gdf.copy()
    edges_gdf[district_field+'_uv'] = 999999
    edges_gdf[district_field+'_u'] = 999999
    edges_gdf[district_field+'_v'] = 999999
    
    edges_gdf[[district_field+'_uv', district_field+'_u', district_field+'_v']] = edges_gdf.apply(lambda row: _assign_district_to_edge(row['edgeID'], edges_gdf, 
                        nodes_gdf, district_field), axis = 1, result_type= 'expand')      

    return edges_gdf

def _assign_district_to_edge(edgeID, edges_gdf, nodes_gdf, district_field):
    """
    Run the natural_roads function on an entire GeoDataFrame of street segments.
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
    series = edges_gdf.loc[edgeID]
    district_uv = 999999
    district_u = nodes_gdf.loc[series.u][district_field]
    district_v = nodes_gdf.loc[series.v][district_field]
    if district_u == district_v: 
        district_uv = district_u
    return district_uv, district_u, district_v

   
def district_to_nodes_from_edges(node_geometry, edges_gdf, sindex):
    """
    Run the natural_roads function on an entire GeoDataFrame of street segments.
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
    point = node_geometry
    n = point.buffer(20)
    possible_matches_index = list(sindex.intersection(n.bounds))
    pm = edges_gdf.iloc[possible_matches_index].copy()
    dist = distance_geometry_gdf(point, pm)
    return edges_gdf.loc[dist[1]]['district']
    
def district_to_nodes_from_polygons(node_geometry, polygons_gdf):
    """
    Run the natural_roads function on an entire GeoDataFrame of street segments.
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
    point = node_geometry
    dist = distance_geometry_gdf(point, polygons_gdf)
    return polygons_gdf.loc[dist[1]]['districtID']
              
def find_gateways(nodeID, nodes_gdf, edges_gdf):
    """
    Run the natural_roads function on an entire GeoDataFrame of street segments.
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
    tmp = edges_gdf[(edges_gdf.u == nodeID) | (edges_gdf.v == nodeID)].copy()
    tmp_nodes = nodes_gdf[nodes_gdf.nodeID.isin(tmp.u) | nodes_gdf.nodeID.isin(tmp.v)].copy()
    if (len(tmp_nodes.district.unique()) > 1): 
        return 1
    return 0
    
def check_disconnected_districts(nodes_gdf, edges_gdf, min_size = 10):
    """
    Run the natural_roads function on an entire GeoDataFrame of street segments.
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
    nodes_gdf = nodes_gdf.copy()
    districts = nodes_gdf.district.unique()
    
    for district in districts:
        if district == 999999: 
            continue
        tmp_nodes = nodes_gdf[nodes_gdf.district == district].copy()
        tmp_edges = edges_gdf[edges_gdf.u.isin(tmp_nodes.nodeID) & edges_gdf.v.isin(tmp_nodes.nodeID)].copy()
        tmp_graph = graph_fromGDF(tmp_nodes, tmp_edges, 'nodeID')
        if len(tmp_nodes) < min_size: 
            nodes_gdf.loc[nodes_gdf.nodeID.isin(tmp_nodes.nodeID), 'district'] = 999999
            continue
        if not nx.is_connected(tmp_graph): 
            largest_component = max(nx.connected_components(tmp_graph), key=len)
            G = tmp_graph.subgraph(largest_component)
            to_check = [item for item in list(tmp_nodes.nodeID) if item not in list(G.nodes())]
            nodes_gdf.loc[nodes_gdf.nodeID.isin(to_check), 'district'] = 999999
    
    return nodes_gdf
    
  
def amend_node_membership(nodeID, nodes_gdf, edges_gdf):
    """
    Run the natural_roads function on an entire GeoDataFrame of street segments.
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
    
    if nodes_gdf.loc[nodeID]['district'] != 999999: 
        return nodes_gdf.loc[nodeID]['district']
    tmp_edges = edges_gdf[(edges_gdf.u == nodeID) | (edges_gdf.v == nodeID)].copy()
    unique =  list(np.unique(tmp_edges[['u', 'v']].values))
    unique.remove(nodeID)
    tmp_nodes = nodes_gdf[(nodes_gdf.nodeID.isin(unique)) & (nodes_gdf.district != 999999) ].copy()
    if len(tmp_nodes) == 0: 
        return 999999
    districts_sorted = tmp_nodes.district.value_counts(sort=True, ascending=False)
    
    if len(districts_sorted) == 1:  
        return districts_sorted.idxmax()
    if districts_sorted.iloc[0] > districts_sorted.iloc[1]: 
        return districts_sorted.idxmax()
    
    tmp_nodes = tmp_nodes[tmp_nodes.district.isin(list(districts_sorted[0:2].index))]
    tmp_edges = edges_gdf[((edges_gdf.u == nodeID) & (edges_gdf.v.isin(tmp_nodes.nodeID))) |
                          ((edges_gdf.v == nodeID) & (edges_gdf.u.isin(tmp_nodes.nodeID)))]
    closest = distance_geometry_gdf(nodes_gdf.loc[nodeID].geometry, tmp_nodes)[1]
    return tmp_nodes.loc[closest].district
