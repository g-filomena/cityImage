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
    if weight is None: 
        weight = 'topo' #the function requires a string 
    # extraction of the best partitions
    partition = community.best_partition(dual_graph, weight=weight)
    dct = dual_id_dict(partition, dual_graph, 'edgeID')
    subdvisions.append(dct)

    # saving the data in a geodataframe
    partitions_df = dict_to_df(subdvisions, ['p_'+weight])
    regions = pd.merge(edges_gdf, partitions_df, left_on = 'edgeID', right_index = True, how= 'left')
    return regions
    
def identify_regions_primal(graph, nodes_graph, weight = None, barrier_field = None):
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
    if weight is None: 
        weight = 'topo' #the function requires a string 
    # extraction of the best partitions
    partition = best_partition(graph, weight=weight, barrier_field = barrier_field)
    regions = nodes_graph.copy()
    regions['p_'+weight] = regions.nodeID.map(partition)
    return regions
  
                
def reset_index_dual_gdfsIG(nodesDual_gdf, edgesDual_gdf):
    '''
    The function simply reset the indexes of the two dataframes.
     
    Parameters
    ----------
    nodesDual_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
   
    Returns
    -------
    tuple of GeoDataFrames
    '''
    
    nodesDual_gdf, edgesDual_gdf = nodesDual_gdf.copy(), edgesDual_gdf.copy() 
    nodesDual_gdf = nodesDual_gdf.reset_index(drop = True)

    edgesDual_gdf['u'], edgesDual_gdf['v'] = edgesDual_gdf['u'].astype(int), edgesDual_gdf['v'].astype(int)
    nodesDual_gdf['IG_edgeID'] = nodesDual_gdf.index.values.astype(int)
    nodesDual_gdf['edgeID'] = nodesDual_gdf['edgeID'].astype(int)

    edgesDual_gdf = edgesDual_gdf.rename(columns = {'u':'old_u', 'v':'old_v'})
    edgesDual_gdf = pd.merge(edgesDual_gdf, nodesDual_gdf[['edgeID', 'IG_edgeID']], how='left', left_on='old_u', right_on='edgeID')
    edgesDual_gdf = edgesDual_gdf.rename(columns = {'IG_edgeID' : 'u'})
    edgesDual_gdf = pd.merge(edgesDual_gdf, nodesDual_gdf[['edgeID', 'IG_edgeID']], how='left', left_on='old_v', right_on='edgeID')
    edgesDual_gdf = edgesDual_gdf.rename(columns = {'IG_edgeID' : 'v'})
    edgesDual_gdf.drop(['edgeID_x', 'edgeID_y', 'old_u', 'old_v'], inplace = True, axis = 1)
    
    nodesDual_gdf.index = nodesDual_gdf['IG_edgeID']
    nodesDual_gdf.index.name = None
    
    return nodesDual_gdf, edgesDual_gdf
    
def reset_index_gdfsIG(nodes_gdf, edges_gdf):
    '''
    The function simply reset the indexes of the two dataframes.
     
    Parameters
    ----------
    nodesDual_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
   
    Returns
    -------
    tuple of GeoDataFrames
    '''
    
    nodes_gdf, edges_gdf = nodes_gdf.copy(), edges_gdf.copy()
    nodes_gdf = nodes_gdf.reset_index(drop = True)
  
    edges_gdf['u'], edges_gdf['v'] = edges_gdf['u'].astype(int), edges_gdf['v'].astype(int)
    nodes_gdf['IG_nodeID'] = nodes_gdf.index.values.astype(int)
    nodes_gdf['nodeID'] = nodes_gdf['nodeID'].astype(int)
    edges_gdf = edges_gdf.rename(columns = {'u':'old_u', 'v':'old_v'})
    edges_gdf = pd.merge(edges_gdf, nodes_gdf[['nodeID', 'IG_nodeID']], how='left', left_on='old_u', right_on=label_index)
    edges_gdf = edges_gdf.rename(columns = {'IG_nodeID' : 'u'})
    edges_gdf = pd.merge(edges_gdf, nodes_gdf[['nodeID','IG_nodeID']], how='left', left_on='old_v', right_on=label_index)
    edges_gdf = edges_gdf.rename(columns = {'IG_nodeID': 'v'})
    edges_gdf.drop(['nodeID_x', 'nodeID_y', 'old_u', 'old_v'], inplace = True, axis = 1)
    
    nodes_gdf.index = nodes_gdf['IG_nodeID']
    nodes_gdf.index.name = None
    return nodes_gdf, edges_gdf    
    

def dual_graphIG_fromGDF(nodes_dual, edges_dual):

    '''
    The function generates a NetworkX graph from dual-nodes and -edges GeoDataFrames.
            
    Parameters
    ----------
    nodes_dual: Point GeoDataFrame
        the GeoDataFrame of the dual nodes, namely the street segments' centroids
    edges_dual: LineString GeoDataFrame
        the GeoDataFrame of the dual edges, namely the links between street segments' centroids 
        
    Returns
    -------
    NetworkX Graph
    '''
   
    edges_dual.u = edges_dual.u.astype(int)
    edges_dual.v = edges_dual.v.astype(int)
    
    Dg = nx.Graph()   
    Dg.add_nodes_from(nodes_dual.index)
    attributes = nodes_dual.to_dict()
    
    a = (nodes_dual.applymap(type) == list).sum()
    if len(a[a>0]): 
        to_ignore = a[a>0].index[0]
    else: to_ignore = []
    
    for attribute_name in nodes_dual.columns:
        # only add this attribute to nodes which have a non-null value for it
        if attribute_name in to_ignore: 
            continue
        attribute_values = {k:v for k, v in attributes[attribute_name].items() if pd.notnull(v)}
        nx.set_node_attributes(Dg, name=attribute_name, values=attribute_values)

    # add the edges and attributes that are not u, v, key (as they're added
    # separately) or null
    for _, row in edges_dual.iterrows():
        attrs = {}
        for label, value in row.iteritems():
            if (label not in ['u', 'v']) and (isinstance(value, list) or pd.notnull(value)):
                attrs[label] = value
        Dg.add_edge(row['u'], row['v'], **attrs)

    return Dg
    ## polygonise

    
def polygonise_partition(edges_gdf, partition_field, method = None, buffer = 30):
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
    ix_u = edges_gdf.columns.get_loc('u')+1  
    ix_v = edges_gdf.columns.get_loc('v')+1  
    
    edges_gdf = edges_gdf.copy()
    edges_gdf[district_field+'_uv'] = 999999
    edges_gdf[district_field+'_u'] = 999999
    edges_gdf[district_field+'_v'] = 999999
    
    edges_gdf[[district_field+'_uv', district_field+'_u', district_field+'_v']] = edges_gdf.apply(lambda row: _assign_district_to_edge(row['edgeID'], edges_gdf, 
                        nodes_gdf, district_field), axis = 1, result_type= 'expand')      

    return edges_gdf

# def append_dual_edges_metrics(edges_gdf, dual_graph, dict_metric, name_metric): 
    
    # dictionary = dual_id_dict(dict_metric, dual_graph, 'edgeID')
    # tmp = dict_to_df([dictionary], [name_metric])
    # edges_gdf = pd.merge(edges_gdf, tmp, left_on = 'edgeID', right_index = True, how = 'left')
    # edges_gdf.index = edges_gdf.edgeID
    # edges_gdf.index.name = None
    
    # return edges_gdf

def _assign_district_to_edge(edgeID, edges_gdf, nodes_gdf, district_field):
    series = edges_gdf.loc[edgeID]
    district_uv = 999999
    district_u = nodes_gdf.loc[series.u][district_field]
    district_v = nodes_gdf.loc[series.v][district_field]
    if district_u == district_v: 
        district_uv = district_u
    return district_uv, district_u, district_v

   
def district_to_nodes_from_edges(node_geometry, edges_gdf, sindex):
    point = node_geometry
    n = point.buffer(20)
    possible_matches_index = list(sindex.intersection(n.bounds))
    pm = edges_gdf.iloc[possible_matches_index].copy()
    dist = distance_geometry_gdf(point, pm)
    return edges_gdf.loc[dist[1]]['district']
    
def district_to_nodes_from_polygons(node_geometry, polygons_gdf):
    point = node_geometry
    dist = distance_geometry_gdf(point, polygons_gdf)
    return polygons_gdf.loc[dist[1]]['districtID']
              
def find_gateways(nodeID, nodes_gdf, edges_gdf):
    tmp = edges_gdf[(edges_gdf.u == nodeID) | (edges_gdf.v == nodeID)].copy()
    tmp_nodes = nodes_gdf[nodes_gdf.nodeID.isin(tmp.u) | nodes_gdf.nodeID.isin(tmp.v)].copy()
    if (len(tmp_nodes.district.unique()) > 1): 
        return 1
    return 0
    
def check_disconnected_districts(nodes_gdf, edges_gdf, min_size = 10):
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