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
   
def get_network_fromOSM(place, download_method, network_type = "all", epsg = None, distance = 500.0): 
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
        - when using "OSMpolygon" please provide the name of a relation in OSM as an argument of place;
    download_method: str, {"distance_from_address", "distance_from_point", "OSMpolygon", "OSMplace", "polygon"}
        It indicates the method that should be used for downloading the data. When 'polygon' the shape to get network data within coordinates should be in
        unprojected latitude-longitude degrees (EPSG:4326).
    network_type: str {"walk", "bike", "drive", "drive_service", "all", "all_private", "none"}
        It indicates type of street or other network to extract - from OSMNx paramaters.
    epsg: int
        Epsg of the area considered; if None OSMNx is used for the projection.
    distance: float
        It is used only if download_method == "distance_from_address" or "distance_from_point".
        
    Returns
    -------
    nodes_gdf, edges_gdf: Tuple of GeoDataFrames
        the junction and street segments GeoDataFrames.
    """
    if epsg is not None:
        crs = 'EPSG:' + str(epsg)
    
    G = downloader(place = place, download_method = download_method, distance = distance, downloading_graph = True, network_type = network_type)
    
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
    
    # resolving lists 
    for column in ["highway", "name", "oneway"]:
        edges_gdf[column] = [x[0] if isinstance(x, list) else x for x in edges_gdf[column]]
    for column in ["lanes", "bridge", "tunnel"]:
        if column not in edges_gdf.columns:
            continue
        edges_gdf[column] = [max(x) if isinstance(x, list) else x for x in edges_gdf[column]]
        if column in ["bridge", "tunnel"]:
            edges_gdf[column] = edges_gdf[column].apply(lambda x: 0 if pd.isna(x) or x is False else 1)

       
    # finalising geodataframes
    if epsg is None: 
        nodes_gdf, edges_gdf = ox.projection.project_gdf(nodes_gdf), ox.projection.project_gdf(edges_gdf)
    else: 
        nodes_gdf, edges_gdf = nodes_gdf.to_crs(crs), edges_gdf.to_crs(crs)
    
    nodes_gdf["x"], nodes_gdf["y"] = list(zip(*[(geometry.coords[0][0], geometry.coords[0][1]) for geometry in nodes_gdf.geometry]))
    
    if len(nodes_gdf.geometry.iloc[0].coords) > 2:
        nodes_gdf['z'] = [geometry.coords[0][2] for geometry in nodes_gdf.geometry]
    else: 
        nodes_gdf['z'] = 2.0
    
    return nodes_gdf, edges_gdf

def get_network_fromFile(path, epsg, dict_columns = {}, other_columns = []):
    """
    The function loads a vector lines from a specified directory, along with the epsg coordinate code.
    It creates two GeoDataFrame, one for street junctions (nodes) and one for street segments (edges).
    The GeoDataFrames are built assuming a planar undirected graph. 
   
    Parameters
    ----------
    path: str
        The local path where the file is stored, including its extention (".shp", ..).
    epsg: int
        Epsg of the area considered.
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
    # try reading street network from directory
    crs = 'EPSG:' + str(epsg)
    edges_gdf = gpd.read_file(path)
    
    return get_network_fromGDF(edges_gdf, epsg, dict_columns = dict_columns, other_columns = other_columns)
    
    
def get_network_fromGDF(edges_gdf, epsg, dict_columns = {}, other_columns = []):
    """
    The function loads a vector lines shapefile from a given LineString GeoDataFrame, along with the epsg coordinate code.
    It creates two GeoDataFrame, one for street junctions (nodes) and one for street segments (edges).
    The GeoDataFrames are built assuming a planar undirected graph. 
     
    Parameters
    ----------
    path: str
        the local path where the .shp file is stored.
    epsg: int
        Epsg of the area considered .
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
    # try reading street network from directory
    crs = 'EPSG:' + str(epsg)
    try:
        edges_gdf = edges_gdf.to_crs(crs)
    except:
        edges_gdf.crs = crs
       
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
    
    # edges_gdf["geometry"] = edges_gdf.apply(lambda row: LineString([coor for coor in [row["geometry"].coords[i][0:2] for i in range(0, len(row["geometry"].coords))]]), axis = 1)
    edges_gdf = fix_multiparts_LineString_gdf(edges_gdf)    
    
    # assign indexes
    edges_gdf.reset_index(inplace=True, drop=True)
    edges_gdf["edgeID"] = edges_gdf.index.values.astype(int) 
    nodes_gdf = obtain_nodes_gdf(edges_gdf, crs)
    
    # linking on coordinates
    nodes_gdf["nodeID"] = nodes_gdf.index.values.astype(int)
    nodes_gdf, edges_gdf = join_nodes_edges_by_coordinates(nodes_gdf, edges_gdf)
    edges_gdf["length"] = edges_gdf["geometry"].length # computing length
    
    if 'z' not in nodes_gdf.columns:
        nodes_gdf['z'] = 2.0
    
    return nodes_gdf, edges_gdf      

def obtain_nodes_gdf(edges_gdf, crs):
    """
    It obtains the nodes GeoDataFrame from the unique coordinates pairs in the edges_gdf GeoDataFrame.
        
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        Street segments GeoDataFrame.
    crs: str
        Coordinate reference system of the area considered .
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

    edges_gdf['u'], edges_gdf['v'] = edges_gdf['u'].astype(int), edges_gdf['v'].astype(int)
    edges_gdf = edges_gdf.rename(columns = {"u":"old_u", "v":"old_v"})    
    nodes_gdf["old_nodeID"] = nodes_gdf[nodeID].values.astype(int)
    nodes_gdf = nodes_gdf.reset_index(drop = True)
    nodes_gdf[nodeID] = nodes_gdf.index.values.astype(int)
    
    edges_gdf = pd.merge(edges_gdf, nodes_gdf[["old_nodeID", nodeID]], how="left", left_on="old_u", right_on="old_nodeID")
    edges_gdf = edges_gdf.rename(columns = {nodeID:"u"})
    edges_gdf = pd.merge(edges_gdf, nodes_gdf[["old_nodeID", nodeID]], how="left", left_on="old_v", right_on="old_nodeID")
    edges_gdf = edges_gdf.rename(columns = {nodeID:"v"})
    
    edges_gdf.drop(["old_u", "old_nodeID_x", "old_nodeID_y", "old_v"], axis = 1, inplace = True)
    nodes_gdf.drop(["old_nodeID", "index"], axis = 1, inplace = True, errors = "ignore")
    edges_gdf = edges_gdf.reset_index(drop=True)
    edges_gdf[edgeID] = edges_gdf.index.values.astype(int)
        
    return nodes_gdf, edges_gdf
    
