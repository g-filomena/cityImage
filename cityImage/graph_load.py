import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import osmnx as ox 
import pandas as pd
import numpy as np
import geopandas as gpd
import math
from math import sqrt
import ast
import functools

from shapely.geometry import Point, LineString, Polygon, MultiPoint
from shapely.ops import split, linemerge

from .utilities import fix_multiparts_LineString_gdf, downloader
pd.set_option("display.precision", 3)
pd.options.mode.chained_assignment = None
   
def get_network_fromOSM(OSMplace, download_method, network_type = "all", crs = None, distance = 500.0): 
    """
    The function downloads and creates a simplified OSMNx graph for a selected area's street network.
    Afterwards, GeoDataFrames for nodes and edges are created, assigning new nodeID and edgeID identifiers.
        
    Parameters
    ----------
    place: str, tuple
        Name of cities or areas in OSM: 
        - when using "distance_from_point" please provide a (lat, lon) tuple to create the bounding box around it; 
        - when using "distance_from_address" provide an existing OSM address; 
        - when using "OSMplace" provide an OSM place name.  
        - when using "polygon" please provide the name of a relation in OSM as an argument of place;
    download_method: str, {"distance_from_address", "distance_from_point", "OSMpolygon", "OSMplace", "polygon"}
        It indicates the method that should be used for downloading the data. When 'polygon' the shape to get network data within coordinates should be in
        unprojected latitude-longitude degrees (EPSG:4326).
    network_type: str {"walk", "bike", "drive", "drive_service", "all", "all_private", "none"}
        It indicates type of street or other network to extract - from OSMNx paramaters.
    crs : str, or pyproj.CRS
        Coordinate Reference System for the study area. Can be a string (e.g. 'EPSG:32633'), or a pyproj.CRS object.
    distance: float
        It is used only if download_method == "distance_from_address" or "distance_from_point".
        
    Returns
    -------
    nodes_gdf, edges_gdf: Tuple of GeoDataFrames
        the junction and street segments GeoDataFrames.
    """
   
    G = downloader(OSMplace = OSMplace, download_method = download_method, distance = distance, downloading_graph = True, network_type = network_type)
    
    # fix list of osmid assigned to same edges
    for i, item in enumerate(G.edges()):
        if isinstance(G[item[0]][item[1]][0]["osmid"], (list,)): 
            G[item[0]][item[1]][0]["osmid"] = G[item[0]][item[1]][0]["osmid"][0]
    
    edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False, node_geometry=True, fill_edge_geometry=False)
    
    nodes_gdf = nodes.drop(["highway", "ref"], axis=1, errors = "ignore")
    edges_gdf.reset_index(inplace = True)
    nodes_gdf['nodeID'] = nodes_gdf.index
    nodes_gdf, edges_gdf = reset_index_graph_gdfs(nodes_gdf, edges_gdf, nodeID = 'nodeID')    
               
    # columns to keep (u and v represent "from" and "to" node)
    nodes_gdf = nodes_gdf[["nodeID","x","y","geometry"]]
    to_keep = ["edgeID", "u", "v", "key", "geometry", "length", "name", "highway", "oneway"]
    
    for column in ["lanes", "bridge", "tunnel"]:
        if column in edges_gdf.columns:
            to_keep.append(column)
            if column == "oneway":
                edges_gdf["oneway"] *= 1
    
    edges_gdf = edges_gdf[to_keep]
    edges_gdf = _resolve_list_edges_gdf(edges_gdf)
    nodes_gdf, edges_gdf = _project_gdfs(nodes_gdf, edges_gdf, crs)
    nodes_gdf["x"], nodes_gdf["y"] = list(zip(*[(geometry.coords[0][0], geometry.coords[0][1]) for geometry in nodes_gdf.geometry]))
    
    if len(nodes_gdf.geometry.iloc[0].coords) > 2:
        nodes_gdf['z'] = [geometry.coords[0][2] for geometry in nodes_gdf.geometry]
    else: 
        nodes_gdf['z'] = 2.0
    
    nodes_gdf = nodes_gdf[nodes_gdf.nodeID.isin(np.unique(edges_gdf[['u', 'v']].values))]

    return nodes_gdf, edges_gdf
    
def get_pedestrian_network_fromOSM(OSMplace, download_method, crs = None, distance = 500.0):
    """
    The function downloads and creates a simplified OSMNx graph for a selected area's street network. Compare to get_network_fromOSM, 
    this is more precise in terms of pedestrian paths and it includes the column "lit", and anything the refers to "sidewalk".
    Afterwards, GeoDataFrames for nodes and edges are created, assigning new nodeID and edgeID identifiers.
        
    Parameters
    ----------
    OSMplace: str, tuple
        Name of cities or areas in OSM: 
        - when using "distance_from_point" please provide a (lat, lon) tuple to create the bounding box around it; 
        - when using "distance_from_address" provide an existing OSM address; 
        - when using "OSMplace" provide an OSM place name.  
        - when using "OSMpolygon" please provide the name of a relation in OSM as an argument of place;
    download_method: str, {"distance_from_address", "distance_from_point", "OSMpolygon", "OSMplace", "polygon"}
        It indicates the method that should be used for downloading the data. When 'polygon' the shape to get network data within coordinates should be in
        unprojected latitude-longitude degrees (EPSG:4326).
    crs : str, or pyproj.CRS
        Coordinate Reference System for the study area. Can be a string (e.g. 'EPSG:32633'), or a pyproj.CRS object.
    distance: float
        It is used only if download_method == "distance_from_address" or "distance_from_point".
        
    Returns
    -------
    nodes_gdf, edges_gdf: Tuple of GeoDataFrames
        the junction and street segments GeoDataFrames.
    """
    
    # Fetch all geometries with the "highway" tag
    tags = {"highway": True}
    
    # Retrieve geometries
    gdf = downloader(OSMplace = OSMplace, download_method = download_method, tags = tags, distance = distance)
       
    # Make sure columns exist (otherwise you'll get a KeyError)
    for col in ["area", "foot", "service", "sidewalk",
                "sidewalk:both", "sidewalk:left", "sidewalk:right"]:
        if col not in gdf.columns:
            gdf[col] = np.nan
    
    exclude_list = [
        "abandoned", "bus_guideway", "construction",
        "motor", "no", "planned", "platform", "proposed", 
        "raceway", "razed", "motorway_junction", "motorway", "trunk_link", 
        'emergency_bay', 'bus_bay', "trunk", 'motorway_link', "busway", "cycleway",
    ]
    
    exclude_sidewalk = ["no", "none"]
    exclude_footway = ['sidewalk','traffic_island']
    gdf = gdf[
        (gdf["area"] != "yes") &
        (~gdf["highway"].isin(exclude_list)) &
        (gdf["foot"] != "no") &
        (~gdf["footway"].isin(exclude_footway)) &
        (gdf["service"] != "private") &
        (
            (~gdf["sidewalk"].isin(exclude_sidewalk)) |
            (gdf["highway"] == "residential")
        )
    ]

    gdf = gdf[gdf.geometry.type == 'LineString']
    to_keep = ["geometry", "name", "highway", "lit", "foot"]
    sidewalk_columns = [col for col in gdf.columns if col.startswith("sidewalk")]
    to_keep += sidewalk_columns
    
    gdf = gdf[to_keep].copy()
    gdf.reset_index(inplace = True)
    gdf = gdf.drop(["element_type", "osmid"], axis = 1, errors = 'ignore')
    
    nodes_gdf, edges_gdf = get_network_fromGDF(gdf, epsg, other_columns = ["name", "highway", "lit"])
    edges_gdf = _resolve_list_edges_gdf(edges_gdf)
    
    return nodes_gdf, edges_gdf

def _resolve_list_edges_gdf(edges_gdf):

    """
    Resolves list-type values in specified columns of an edge GeoDataFrame.

    For the columns 'highway', 'name', and 'oneway', if a cell contains a list, only the first element is kept.
    For the columns 'lanes', 'bridge', and 'tunnel', if a cell contains a list, the maximum value is kept.
    For 'bridge' and 'tunnel', values are then binarized: 1 if present, 0 otherwise (or if NaN/False).

    Parameters
    ----------
    edges_gdf : GeoDataFrame
        Edge GeoDataFrame whose specified columns may contain list-type values.

    Returns
    -------
    edges_gdf : GeoDataFrame
        Modified GeoDataFrame with resolved columns.
    """
    
    for column in ["highway", "name", "oneway"]:
        if column in edges_gdf.columns:
            edges_gdf[column] = [x[0] if isinstance(x, list) else x for x in edges_gdf[column]]

    for column in ["lanes", "bridge", "tunnel"]:
        if column in edges_gdf.columns:
            edges_gdf[column] = [max(x) if isinstance(x, list) else x for x in edges_gdf[column]]
            if column in ["bridge", "tunnel"]:
                edges_gdf[column] = edges_gdf[column].apply(lambda x: 0 if pd.isna(x) or x is False else 1)
                
    return edges_gdf

def _project_gdfs(nodes_gdf, edges_gdf, crs):
    """
    Projects nodes and edges GeoDataFrames to the specified coordinate reference system (CRS).

    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        GeoDataFrame of network nodes.
    edges_gdf : GeoDataFrame
        GeoDataFrame of network edges.
    crs : str or pyproj.CRS. 
        Coordinate Reference System for the GeoDataFrame. Pass as a string (e.g., 'EPSG:32633') or a pyproj.CRS object.

    Returns
    -------
    nodes_gdf : GeoDataFrame
        Projected nodes GeoDataFrame.
    edges_gdf : GeoDataFrame
        Projected edges GeoDataFrame.
    """
    nodes_gdf, edges_gdf = nodes_gdf.to_crs(crs), edges_gdf.to_crs(crs)
    return nodes_gdf, edges_gdf
    
def get_network_fromGDF(edges_gdf, crs, dict_columns={}, other_columns=[]):
    """
    Constructs node and edge GeoDataFrames from a LineString GeoDataFrame representing a street network.

    The function processes a GeoDataFrame of vector line features (edges), along with the coordinate reference system (CRS).
    It builds two GeoDataFrames: one for street junctions (nodes) and one for street segments (edges).
    The network is assumed to be planar and undirected.

    Parameters
    ----------
    edges_gdf : GeoDataFrame
        GeoDataFrame containing the street segments as LineString geometries.
    crs : str or pyproj.CRS
        Coordinate Reference System for the output GeoDataFrames. Pass as a string (e.g., 'EPSG:32633') or a pyproj.CRS object.
    dict_columns : dict, optional
        Dictionary mapping standard network attribute names to column names in the input data.
        For example:
            {
                "highway": "roadType_field",
                "oneway": "direction_field",
                "lanes": "nr_lanes_field",
                "maxspeed": "speed_field",
                "name": "name_field"
            }
        Only include attributes that are relevant and available in your input.
    other_columns : list, optional
        List of additional column names in the input GeoDataFrame to preserve in the output `edges_gdf`.

    Returns
    -------
    nodes_gdf : GeoDataFrame
        GeoDataFrame of street junctions (nodes), with unique node IDs and geometries.
    edges_gdf : GeoDataFrame
        GeoDataFrame of street segments (edges), with all relevant and requested attributes.
    """

    edges_gdf = edges_gdf.to_crs(crs)       
    edges_gdf["key"] = 0
    
    # creating the dataframes
    new_columns = []
    
    if len(dict_columns) > 0:
        for key, value in dict_columns.items():
            if (value is not None): 
                edges_gdf[key] = edges_gdf[value]
                new_columns.append(key)
     
    standard_columns = ["geometry", "key"]
    edges_gdf = edges_gdf[standard_columns + new_columns + other_columns]
    
    edges_gdf = fix_multiparts_LineString_gdf(edges_gdf)    
    
    # assign indexes
    edges_gdf.reset_index(inplace=True, drop=True)
    edges_gdf["edgeID"] = edges_gdf.index.values.astype("int64")
    nodes_gdf = obtain_nodes_gdf(edges_gdf, crs)
    
    # linking on coordinates
    nodes_gdf["nodeID"] = nodes_gdf.index.values.astype("int64")
    nodes_gdf, edges_gdf = join_nodes_edges_by_coordinates(nodes_gdf, edges_gdf)
    edges_gdf["length"] = edges_gdf["geometry"].length # computing length
    
    if 'z' not in nodes_gdf.columns:
        nodes_gdf['z'] = 2.0
    
    nodes_gdf = nodes_gdf[nodes_gdf.nodeID.isin(np.unique(edges_gdf[['u', 'v']].values))]

    return nodes_gdf, edges_gdf      
    
def get_network_fromFile(input_path, crs, dict_columns = {}, other_columns = []):
    """
    The function loads a vector lines from a specified directory, along with the epsg coordinate code.
    It creates two GeoDataFrame, one for street junctions (nodes) and one for street segments (edges).
    The GeoDataFrames are built assuming a planar undirected graph. 
   
    Parameters
    ----------
    input_path: str
        The local path where the file is stored, including its extention (".shp", ..).
    crs : str or pyproj.CRS. 
        Coordinate Reference System for the GeoDataFrame. Pass as a string (e.g., 'EPSG:32633') or a pyproj.CRS object.
    dict_columns: dict
        It should be structured as: {"highway": "roadType_field",  "oneway": "direction_field", "lanes": "nr. lanes", "maxspeed": "speed_field", "name": "name_field"}.
        Replace the items with the field names in the input data (if the relative attributes are relevant and existing).
    other_columns: list
        Other columns to be preserved in the edges_gdf GeoDataFrame.
    
    Returns
    -------
    nodes_gdf, edges_gdf: tuple
        The junction and street segment GeoDataFrames.
    """
    edges_gdf = gpd.read_file(input_path)
    
    return get_network_fromGDF(edges_gdf, crs, dict_columns = dict_columns, other_columns = other_columns)
    
def obtain_nodes_gdf(edges_gdf, crs):
    """
    It obtains the nodes GeoDataFrame from the unique coordinates pairs in the edges_gdf GeoDataFrame.
        
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        Street segments GeoDataFrame.
    crs : str or pyproj.CRS. 
        Coordinate Reference System for the GeoDataFrame. Pass as a string (e.g., 'EPSG:32633') or a pyproj.CRS object.
    Returns
    -------
    nodes_gdf: Point GeoDataFrame
        The street junctions GeoDataFrame.
    """
    unique_nodes = pd.concat([edges_gdf.geometry.apply(lambda row: row.coords[0]), edges_gdf.geometry.apply(lambda row: row.coords[-1])]).unique()
    
    # z coordinates
    if len(edges_gdf.geometry.iloc[0].coords[0]) > 2:
        nodes_data = pd.DataFrame(list(unique_nodes), columns=["x", "y", "z"]).astype("float")
    else:
        nodes_data = pd.DataFrame(list(unique_nodes), columns=["x", "y"]).astype("float")
        
    geometry = [Point(xy) for xy in zip(nodes_data.x, nodes_data.y)]
    nodes_gdf = gpd.GeoDataFrame(nodes_data, crs=crs, geometry=geometry)
    nodes_gdf.reset_index(drop=True, inplace = True)
    nodes_gdf["nodeID"] = nodes_gdf.index.values.astype("int64")
    return nodes_gdf
    
def join_nodes_edges_by_coordinates(nodes_gdf, edges_gdf):
    """
    The function merge the u-v nodes information, from the nodes GeoDataFrame, with the edges_gdf GeoDataFrame.
    The process exploits coordinates pairs of the edges for finding the relative nodes in the nodes GeoDataFrame.
    
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        Nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        Street segments GeoDataFrame.
   
    Returns
    -------
    nodes_gdf, edges_gdf: tuple
        The junction and street segment GeoDataFrames.
    """
       
    if not "nodeID" in nodes_gdf.columns: 
        nodes_gdf["nodeID"] = nodes_gdf.index.values.astype("int64")
    nodes_gdf["coordinates"] = list(zip(nodes_gdf.x, nodes_gdf.y))
    edges_gdf["u"] = edges_gdf.geometry.apply(lambda row: row.coords[0]).map(nodes_gdf.set_index('coordinates').nodeID)
    edges_gdf["v"] = edges_gdf.geometry.apply(lambda row: row.coords[-1]).map(nodes_gdf.set_index('coordinates').nodeID)
    nodes_gdf = nodes_gdf.drop('coordinates', axis = 1)
    return nodes_gdf, edges_gdf
    
def reset_index_graph_gdfs(nodes_gdf, edges_gdf, nodeID = "nodeID", edgeID = "edgeID"):
    """
    The function simply resets the indexes of the two dataframes.
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        Nodes (junctions) GeoDataFrame.
    edges_gdf: LineString GeoDataFrame
        Street segments GeoDataFrame.
   
    Returns
    -------
    nodes_gdf, edges_gdf: tuple
        The junction and street segment GeoDataFrames.
    """

    edges_gdf['u'], edges_gdf['v'] = edges_gdf['u'].astype("int64"), edges_gdf['v'].astype("int64")
    edges_gdf = edges_gdf.rename(columns = {"u":"old_u", "v":"old_v"})    
    nodes_gdf["old_nodeID"] = nodes_gdf[nodeID].values.astype("int64")
    nodes_gdf = nodes_gdf.reset_index(drop = True)
    nodes_gdf[nodeID] = nodes_gdf.index.values.astype("int64")
    
    edges_gdf = pd.merge(edges_gdf, nodes_gdf[["old_nodeID", nodeID]], how="left", left_on="old_u", right_on="old_nodeID")
    edges_gdf = edges_gdf.rename(columns = {nodeID:"u"})
    edges_gdf = pd.merge(edges_gdf, nodes_gdf[["old_nodeID", nodeID]], how="left", left_on="old_v", right_on="old_nodeID")
    edges_gdf = edges_gdf.rename(columns = {nodeID:"v"})
    
    edges_gdf.drop(["old_u", "old_nodeID_x", "old_nodeID_y", "old_v"], axis = 1, inplace = True)
    nodes_gdf.drop(["old_nodeID", "index"], axis = 1, inplace = True, errors = "ignore")
    edges_gdf = edges_gdf.reset_index(drop=True)
    edges_gdf[edgeID] = edges_gdf.index.values.astype("int64")
        
    return nodes_gdf, edges_gdf
    
