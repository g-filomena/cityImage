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

from .graph import graph_fromGDF, dual_id_dict
from .utilities import dict_to_df, distance_geometry_gdf

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
    
def identify_regions_primal(graph, nodes_gdf, weight = None):
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
    partition = community.best_partition(graph, weight=weight)
    regions = nodes_gdf.copy()
    regions['p_'+weight] = regions.nodeID.map(partition)
    return regions
                
def polygonise_partitions(edges_gdf, column, convex_hull = True, buffer = 30):
    """
    Given districts assign to street segments it create polygons represint districts, either by creating a convex_hull for each group of segments or 
    simply polygonising them.
    
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        the street segments GeoDataFrame
    column: string
        the name of the column containing the district identifier
    convex_hull: boolean
        if trues creates create convex hulls after having polygonised the cluster of segments
    buffer: float
        desired buffer around the polygonised segments, before possibly obtaining the convex hulls
        
    Returns
    -------
    polygonised_partitions: Polygon GeoDataFrame
        a GeoDataFrame containing the polygonised partitions
    """
    
    polygons = []
    partitionIDs = []
    d = {'geometry' : polygons, column : partitionIDs}

    partitions = edges_gdf[column].unique()
    for i in partitions:
        polygon =  polygonize_full(edges_gdf[edges_gdf[column] == i].geometry.unary_union)
        polygon = unary_union(polygon).buffer(buffer)
        if convex_hull:
            polygons.append(polygon.convex_hull)
        else: 
            polygons.append(polygon)
        partitionIDs.append(i)

    df = pd.DataFrame(d)
    polygonised_partitions = gpd.GeoDataFrame(df, crs=edges_gdf.crs, geometry=df['geometry'])
    return polygonised_partitions
   
def district_to_nodes_from_edges(nodes_gdf, edges_gdf, column):
    """
    It assigns districts' identifiers to the street junctions (nodes), when the districts are assigned to the street segments (edges),
    i.e. communities are identified on the dual graph. The attribution is based on Euclidean distance from each node to the closest street segment.
    
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        the nodes (junctions) GeoDataFrame   
    edges_gdf: LineString GeoDataFrame
        the street segments GeoDataFrame
    column: string
        the name of the column containing the district identifier
        
    Returns
    -------
    nodes_gdf: Point GeoDataFrame
        the updated street junctions GeoDataFrame
    """
    
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf[column] = 0
    sindex = edges_gdf.sindex # spatial index
    
    nodes_gdf[column] = nodes_gdf.apply(lambda row: _assign_district_to_node(row['geometry'], edges_gdf, sindex, column), axis = 1)
    nodes_gdf[column] = nodes_gdf[column].astype(int)
    return nodes_gdf
    
def _assign_district_to_node(node_geometry, edges_gdf, sindex, column):   
    """
    It assigns a district's identifier to a street junctions (node), when the districts are assigned to the street segments (edges), 
    i.e. communities are identified on the dual graph.
    
    Parameters
    ----------
    node_geometry: Point
        a node's geometry
    edges_gdf: LineString GeoDataFrame
        the street segments GeoDataFrame
    sindex: 
    
    column: string
        the name of the column containing the district identifier
        
    Returns
    -------
    district: int
        the district identifier
    """   
    point = node_geometry
    n = point.buffer(20)
    possible_matches_index = list(sindex.intersection(n.bounds))
    pm = edges_gdf.iloc[possible_matches_index].copy()
    dist = distance_geometry_gdf(point, pm)
    district = edges_gdf.loc[dist[1]][column]
    return district
    
def districts_to_edges_from_nodes(nodes_gdf, edges_gdf, column):
    """
    It assigns districts' identifiers to the street segments (edges), when the districts are assigned to the junctions(nodes), i.e. communities are identified on the 
    primal graph. The attribution is based on Euclidean distance from each node to the closest street segment.
    
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        the nodes (junctions) GeoDataFrame   
    edges_gdf: LineString GeoDataFrame
        the street segments GeoDataFrame
    column: string
        the name of the column containing the district identifier
        
    Returns
    -------
    edges_gdf: LineString GeoDataFrame
        the updated street segments GeoDataFrame
    """
    
    ix_u = edges_gdf.columns.get_loc('u')+1  
    ix_v = edges_gdf.columns.get_loc('v')+1  
    
    edges_gdf = edges_gdf.copy()
    edges_gdf[column+'_uv'] = 999999
    edges_gdf[column+'_u'] = 999999
    edges_gdf[column+'_v'] = 999999
    
    edges_gdf[[column+'_uv', column+'_u', column+'_v']] = edges_gdf.apply(lambda row: _assign_district_to_edge(row['edgeID'], nodes_gdf, edges_gdf, 
                        column), axis = 1, result_type= 'expand')      

    return edges_gdf

def _assign_district_to_edge(edgeID, nodes_gdf, edges_gdf, column):
    """
    It assigns a district's identifier to a street segments (edge), when the districts are assigned to the junctions (nodes), 
    i.e. communities are identified on the primal graph. The attribution is based on Euclidean distance from each node to the closest street segment.
    
    Parameters
    ----------
    edgeID: int
        the edgeID
    nodes_gdf: Point GeoDataFrame
        the nodes (junctions) GeoDataFrame   
    edges_gdf: LineString GeoDataFrame
        the street segments GeoDataFrame
    column: string
        the name of the column containing the district identifier
    
    Returns
    -------
    Tuple
    """
    series = edges_gdf.loc[edgeID]
    district_uv = 999999
    district_u = nodes_gdf.loc[series.u][column]
    district_v = nodes_gdf.loc[series.v][column]
    if district_u == district_v: 
        district_uv = district_u
    return district_uv, district_u, district_v
    
def district_to_nodes_from_polygons(nodes_gdf, partitions_gdf, column):
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
    nodes_gdf[column] = nodes_gdf.apply(lambda row: _assign_district_to_node_from_polygons(row['geometry'], partitions_gdf, column), axis = 1)
    nodes_gdf[column] = nodes_gdf[column].astype(int)

    return nodes_gdf
    
def _assign_district_to_node_from_polygons(node_geometry, partitions_gdf, column):
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
    dist = distance_geometry_gdf(point, partitions_gdf)
    district = partitions_gdf.loc[dist[1]][column]
    return district        
    
def _check_disconnected_districts(nodes_gdf, edges_gdf, column, min_size = 10):
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
    districts = nodes_gdf[column].unique()
    
    for district in districts:
        if district == 999999: 
            continue
        tmp_nodes = nodes_gdf[nodes_gdf[column] == district].copy()
        tmp_edges = edges_gdf[edges_gdf.u.isin(tmp_nodes.nodeID) & edges_gdf.v.isin(tmp_nodes.nodeID)].copy()
        tmp_graph = graph_fromGDF(tmp_nodes, tmp_edges, 'nodeID')
        
        if len(tmp_nodes) < min_size: 
            nodes_gdf.loc[nodes_gdf.nodeID.isin(tmp_nodes.nodeID), column] = 999999
            continue
        
        if not nx.is_connected(tmp_graph): 
            largest_component = max(nx.connected_components(tmp_graph), key=len)
            G = tmp_graph.subgraph(largest_component)
            to_check = [item for item in list(tmp_nodes.nodeID) if item not in list(G.nodes())]
            nodes_gdf.loc[nodes_gdf.nodeID.isin(to_check), column] = 999999
    
    return nodes_gdf
  
def amend_nodes_membership(nodes_gdf, edges_gdf, column, min_size_district = 10):
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
    nodes_gdf = _check_disconnected_districts(nodes_gdf, edges_gdf, column, min_size_district)
    while (999999 in nodes_gdf[column].unique()):
        nodes_gdf[column] = nodes_gdf.apply(lambda row: _amend_node_membership(row['nodeID'], nodes_gdf, edges_gdf, column), axis = 1)
        nodes_gdf = _check_disconnected_districts(nodes_gdf, edges_gdf, min_size_district)
    
    return nodes_gdf

def _amend_node_membership(nodeID, nodes_gdf, edges_gdf, column):
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
    if nodes_gdf.loc[nodeID][column] != 999999: 
        return nodes_gdf.loc[nodeID][column]
    tmp_edges = edges_gdf[(edges_gdf.u == nodeID) | (edges_gdf.v == nodeID)].copy()
    unique =  list(np.unique(tmp_edges[['u', 'v']].values))
    unique.remove(nodeID)
    tmp_nodes = nodes_gdf[(nodes_gdf.nodeID.isin(unique)) & (nodes_gdf[column] != 999999) ].copy()
    if len(tmp_nodes) == 0: 
        return 999999
    districts_sorted = tmp_nodes[column].value_counts(sort=True, ascending=False)
    
    if len(districts_sorted) == 1:  
        return districts_sorted.idxmax()
    if districts_sorted.iloc[0] > districts_sorted.iloc[1]: 
        return districts_sorted.idxmax()
    
    tmp_nodes = tmp_nodes[tmp_nodes[column].isin(list(districts_sorted[0:2].index))]
    tmp_edges = edges_gdf[((edges_gdf.u == nodeID) & (edges_gdf.v.isin(tmp_nodes.nodeID))) |
                          ((edges_gdf.v == nodeID) & (edges_gdf.u.isin(tmp_nodes.nodeID)))]
    closest = distance_geometry_gdf(nodes_gdf.loc[nodeID].geometry, tmp_nodes)[1]
    new_district = tmp_nodes.loc[closest][column]
    return new_district
    
def find_gateways(nodes_gdf, edges_gdf, column):
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
    
    # assign gateways
    nodes_gdf = nodes_gdf.copy()
    nodes_gdf['gateway'] = nodes_gdf.apply(lambda row: _gateway(row['nodeID'], nodes_gdf, edges_gdf, column), axis = 1)
    return nodes_gdf
    
def _gateway(nodeID, nodes_gdf, edges_gdf, column):
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
    gateway = 99999
    tmp = edges_gdf[(edges_gdf.u == nodeID) | (edges_gdf.v == nodeID)].copy()
    tmp_nodes = nodes_gdf[nodes_gdf.nodeID.isin(tmp.u) | nodes_gdf.nodeID.isin(tmp.v)].copy()
    gateway 
    if (len(tmp_nodes['column'].unique()) > 1): 
        gateway = 1
    else:
        gateway = 0
    return gateway