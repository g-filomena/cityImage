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
from shapely.ops import split
pd.set_option("precision", 10)
pd.options.mode.chained_assignment = None

from .utilities import *

"""
This set of functions handles interoperations between GeoDataFrames and graphs. It allows data conversion and the extraction of nodes and edges GeoDataFrames from roads shapefile or OpenStreetMap.

"""
    
## Graph preparation functions ###############
    
def get_network_fromOSM(place, download_method, network_type = "all", epsg = None, distance = 7000): 

    """
    The function downloads and creates a simplified OSMNx graph for a selected area. 
    Afterwards, GeoDataFrames for nodes and edges are created, assigning new nodeID and edgeID identifiers.
        
    Parameters
    ----------
    place: string
        name of cities or areas in OSM: when using "OSMpolygon" please provide the name of a "relation" in OSM as an argument of "place"; when using "distance_from_address"
        provide an existing OSM address; when using "OSMplace" provide an OSM place name
    download_method: string, {"OSMpolygon", "distance_from_address", "OSMplace"}
        it indicates the method that should be used for downloading the data.
    network_type: string,  {"walk", "bike", "drive", "drive_service", "all", "all_private", "none"}
        it indicates type of street or other network to extract - from OSMNx paramaters
    epsg: int
        epsg of the area considered; if None OSMNx is used for the projection
    distance: float
        it is used only if download_method == "distance from address"
        
    Returns
    -------
    tuple of GeoDataFrames
    """
    
    # using OSMNx to download data from OpenStreetMap     
    if download_method == "OSMpolygon":
        query = ox.osm_polygon_download(place, limit=1, polygon_geojson=1)
        OSMplace = query[0]["display_name"]
        G = ox.graph_from_place(OSMplace, network_type = network_type, simplify = True)
        
    elif download_method == "distance_from_address":
        G = ox.graph_from_address(place, network_type = network_type, distance = distance, simplify = True)
    
    # (download_method == "OSMplace")
    else: G = ox.graph_from_place(place, network_type = network_type, simplify = True)
    
    # fix list of osmid assigned to same edges
    for i, item in enumerate(G.edges()):
        if isinstance(G[item[0]][item[1]][0]["osmid"], (list,)): 
            G[item[0]][item[1]][0]["osmid"] = G[item[0]][item[1]][0]["osmid"][0]
    
    edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
    
    nodes = ox.graph_to_gdfs(G, nodes=True, edges=False, node_geometry=True, fill_edge_geometry=False)
    nodes_gdf = nodes.drop(["highway", "ref"], axis=1, errors = "ignore")
    # getting rid of OSMid and preparing geodataframes
    nodes_gdf.index = nodes_gdf.osmid.astype("int64")
    nodes_gdf, edges_gdf = reset_index_street_network_gdfs(nodes_gdf, edges_gdf)    
               
    # columns to keep (u and v represent "from" and "to" node)
    nodes_gdf = nodes_gdf[["nodeID","x","y","geometry"]]
    edges_gdf = edges_gdf[["edgeID", "u", "v", "key", "geometry", "length", "highway", "oneway", "lanes", "name", "bridge", "tunnel"]]
    edges_gdf["oneway"] *= 1
    
    # resolving lists 
    edges_gdf["highway"] = [x[0] if isinstance(x, list) else x for x in edges_gdf["highway"]]
    edges_gdf["name"] = [x[0] if isinstance(x, list) else x for x in edges_gdf["name"]]
    edges_gdf["lanes"] = [max(x) if isinstance(x, list) else x for x in edges_gdf["lanes"]]
    edges_gdf["bridge"] = [max(x) if isinstance(x, list) else x for x in edges_gdf["bridge"]]
    edges_gdf["tunnel"] = [max(x) if isinstance(x, list) else x for x in edges_gdf["tunnel"]]
    
    # finalising geodataframes
    if epsg is None: 
        nodes_gdf, edges_gdf = ox.projection.project_gdf(nodes_gdf), ox.projection.project_gdf(edges_gdf)
    else: nodes_gdf, edges_gdf = nodes_gdf.to_crs(epsg = epsg), edges_gdf.to_crs(epsg = epsg)
    
    nodes_gdf["x"], nodes_gdf["y"] = list(zip(*[(r.coords[0][0], r.coords[0][1]) for r in nodes_gdf.geometry]))
    nodes_gdf['height'] = 2 # this will be used for 3d visibility analysis
    return nodes_gdf, edges_gdf

def get_network_fromSHP(path, epsg, dict_columns = {}, other_columns = []):
    """
    The function loads a vector lines shapefile from a specified directory, along with the epsg coordinate code.
    It creates two GeoDataFrame, one for street junctions (nodes) and one for street segments (edges).
    The GeoDataFrames are built assuming a planar undirected graph. 
    The "case_study_area" polygon is optional and when provided is used to select geometries within the area + a buffer of x meters, fixed by the researcher (distance_from_boundary)
     
    Parameters
    ----------
    path: string
        the local path where the .shp file is stored
    epsg: int
        epsg of the area considered 
    dict_columns: dict
        it should be structured as: {"roadType_field": "highway",  "direction_field": "oneway", "nr. lanes": "lanes", "speed_field": None, "name_field": "name"}
        Replace the items with the field names in the input data (if the relative attributes are relevant and existing)
    
    Returns
    -------
    tuple of GeoDataFrames
    """
    
    # try reading street network from directory
    crs = {'init': 'epsg'+str(epsg), 'no_defs': True}
    edges_gdf = gpd.read_file(path)
    try:
        edges_gdf = edges_gdf.to_crs(epsg=epsg)
    except:
        edges_gdf.crs = crs
       
    edges_gdf["from"] = None
    edges_gdf["to"] = None
    edges_gdf["key"] = 0
    
    # creating the dataframes
    new_columns = ["highway", "oneway", "lanes", "maxspeed","name"]
    if len(dict_columns) > 0:
        for n, (key, value) in enumerate(dict_columns.items()):
            if (value is not None): 
                edges_gdf[new_columns[n]] = edges_gdf[value]
    else: new_columns = []
     
    standard_columns = ["geometry", "from", "to", "key"]
    edges_gdf = edges_gdf[standard_columns + new_columns + other_columns]
    
    # remove z coordinates, if any
    edges_gdf["geometry"] = edges_gdf.apply(lambda row: LineString([coor for coor in [row["geometry"].coords[i][0:2] for i in range(0, len(row["geometry"].coords))]]), axis = 1)
    edges_gdf['edgeID'] = edges_gdf.index.values.astype(int)
    edges_gdf.reset_index(inplace=True, drop=True)
    
    # assigning indexes
    edges_gdf.reset_index(inplace=True, drop=True)
    edges_gdf["edgeID"] = edges_gdf.index.values.astype(int) 
    nodes_gdf = obtain_nodes_gdf(edges_gdf, epsg)
    
    # linking on coordinates
    nodes_gdf["nodeID"] = nodes_gdf.index.values.astype(int)
    nodes_gdf, edges_gdf = join_by_coordinates(nodes_gdf, edges_gdf)
    edges_gdf["length"] = edges_gdf["geometry"].length # computing length
    nodes_gdf['height'] = 2 # this will be used for 3d visibility analysis
    
    return nodes_gdf, edges_gdf
    
def obtain_nodes_gdf(edges_gdf, epsg):
    """
    It obtains the nodes GeoDataFrame from the unique coordinates pairs in the edges_gdf GeoDataFrame.
        
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
    epsg: int
        epsg of the area considered 
    Returns
    -------
    Point GeoDataFrames
    """
    
    edges_gdf["from"] = edges_gdf.apply(lambda row: row.geometry.coords[0], axis = 1)
    edges_gdf["to"] = edges_gdf.apply(lambda row: row.geometry.coords[-1], axis = 1)
    unique_nodes_tmp = list(edges_gdf["to"].unique()) + list(edges_gdf["from"].unique())
    unique_nodes = list(set(unique_nodes_tmp))
    #preparing nodes geodataframe
    nodes_data = pd.DataFrame.from_records(unique_nodes, columns=["x", "y"]).astype("float")
    geometry = [Point(xy) for xy in zip(nodes_data.x, nodes_data.y)]
    crs = {'init': 'epsg:' + str(epsg)}
    nodes_gdf = gpd.GeoDataFrame(nodes_data, crs=crs, geometry=geometry)
    nodes_gdf.reset_index(drop=True, inplace = True)
    
    return nodes_gdf
    
def join_by_coordinates(nodes_gdf, edges_gdf):

    """
    The function merge the u-v nodes information, from the nodes GeoDataFrame, with the edges_gdf GeoDataFrame.
    The process exploits coordinates pairs of the edges for finding the relative nodes in the nodes GeoDataFrame.
    
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
   
    Returns
    -------
    tuple of GeoDataFrames
    """
    
    if not "nodeID" in nodes_gdf.columns: 
        nodes_gdf["nodeID"] = nodes_gdf.index.values.astype("int64")
    edges_gdf["from"] = edges_gdf.apply(lambda row: row.geometry.coords[0], axis = 1)
    edges_gdf["to"] = edges_gdf.apply(lambda row: row.geometry.coords[-1], axis = 1)
    nodes_gdf["coordinates"] = list(zip(nodes_gdf.x, nodes_gdf.y))

    edges_tmp = pd.merge(edges_gdf, nodes_gdf[["nodeID","coordinates"]], how="left", left_on="from", right_on="coordinates")
    edges_tmp.drop(edges_tmp[["coordinates"]], axis = 1, inplace = True)
    edges_tmp.rename(columns = {"nodeID":"u"}, inplace = True)
    
    edges_gdf = pd.merge(edges_tmp, nodes_gdf[["nodeID","coordinates"]], how="left", left_on="to", right_on="coordinates")
    edges_gdf = edges_gdf.drop(edges_gdf[["coordinates", "from", "to"]], axis = 1)
    edges_gdf = edges_gdf.rename(columns = {"nodeID":"v"})
    nodes_gdf.drop(["coordinates"], axis = 1, inplace = True)
    
    return nodes_gdf, edges_gdf

def reset_index_street_network_gdfs(nodes_gdf, edges_gdf):
    """
    The function simply reset the indexes of the two dataframes.
     
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        nodes (junctions) GeoDataFrame
    edges_gdf: LineString GeoDataFrame
        street segments GeoDataFrame
   
    Returns
    -------
    tuple of GeoDataFrames
    """
    
    edges_gdf = edges_gdf.rename(columns = {"u":"old_u", "v":"old_v"})
    nodes_gdf["old_nodeID"] = nodes_gdf.index.values.astype("int64")
    nodes_gdf = nodes_gdf.reset_index(drop = True)
    nodes_gdf["nodeID"] = nodes_gdf.index.values.astype("int64")
    
    edges_gdf = pd.merge(edges_gdf, nodes_gdf[["old_nodeID", "nodeID"]], how="left", left_on="old_u", right_on="old_nodeID")
    edges_gdf = edges_gdf.rename(columns = {"nodeID":"u"})
    edges_gdf = pd.merge(edges_gdf, nodes_gdf[["old_nodeID", "nodeID"]], how="left", left_on="old_v", right_on="old_nodeID")
    edges_gdf = edges_gdf.rename(columns = {"nodeID":"v"})

    edges_gdf.drop(["old_u", "old_nodeID_x", "old_nodeID_y", "old_v"], axis = 1, inplace = True)
    nodes_gdf.drop(["old_nodeID", "index"], axis = 1, inplace = True, errors = "ignore")
    edges_gdf = edges_gdf.reset_index(drop=True)
    edges_gdf["edgeID"] = edges_gdf.index.values.astype(int)
    
    return nodes_gdf, edges_gdf
    
class Error(Exception):
    """Base class for other exceptions"""

class downloadError(Error):
    """Raised when a wrong download method is provided"""