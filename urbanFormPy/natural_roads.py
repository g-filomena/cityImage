import networkx as nx, pandas as pd, numpy as np, geopandas as gpd
import functools
import community
from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping, MultiLineString
pd.set_option("precision", 10)

import .angles as af

"""
The concet of natural road regards the cognitive perception/representation of road entities, regardless changes in names or
interruptions. Rather, different street segments are "mentally merged" according to continuity rules (here based on the deflection
angle and the egoistic choice process).
"""
	
def identify_natural_roads(nodes_gdf, edges_gdf, tolerance = 45): 
    """
    Run the natural_roads function on an entire geodataframe of street segments.
    The geodataframes are supposed to be cleaned and can be obtained via the functions "get_fromOSM(place)" or "get_fromSHP(directory, 
    epsg)". The parameter tolerance indicates the maximux deflection allowed to consider two roads possible natural continuation.
    
    Parameters
    ----------
    nodes_gdf, edges_gdf: GeoDataFrames
	tolerance = float
    
    Returns
    -------
    Tuple of GeoDataFrames
    """
   
    if (not nodes_simplified(edges_gdf)) | (not edges_simplified(edges_gdf)): 
        raise StreetNetworkError("The street network is not simplified")

    edges_gdf.index = edges_gdf.streetID
    nodes_gdf.index = nodes_gdf.nodeID
	del edges.index.name
    del nodes_gdf.index.name
    
	edges_gdf["naturalID"] = 0
    ix_nID = edges_gdf.columns.get_loc("naturalID")+1
	naturalID = 1  
    
    for row in edges_gdf.itertuples():
        if (row[ix_nID] > 0): continue # if already assigned to a natural road
        _natural_roads(row.Index, naturalID, "fr", nodes_gdf, edges_gdf, tolerance = tolerance) 
        _natural_roads(row.Index, naturalID, "to", nodes_gdf, edges_gdf, tolerance = tolerance) 
        naturalID = naturalID + 1
                                            
    return nodes_gdf, edges_gdf
	
def _natural_roads(streetID, naturalID, direction, nodes_gdf, edges_gdf, tolerance = 45): 
    """
    This function takes a direction "to" or "fr" and two GeoDataFrames, one for roads one for nodes (or junctions).
    Since this function works only for one segment at the time, in one direction, it has to be executed in a for loop
    while iterating through all the edges, both directions.
    
    Parameters
    ----------
    streetID: int, next road to examine
    naturalID: int, current naturalID 
    direction: string, {"to", "fr"}
    nodes_gdf, edges_gdf : GeoDataFrames
    """
	
    # initialise variables
    angles = {}
    directions_dict = {}
    ix_geo = edges_gdf.columns.get_loc("geometry")+1
    ix_u, ix_v = edges_gdf.columns.get_loc("u")+1, edges_gdf.columns.get_loc("v")+1
    ix_nID = edges_gdf.columns.get_loc("naturalID")+1
    to_node, from_node = edges_gdf.loc[streetID]["u"], edges_gdf.loc[streetID]["v"]
    
	line_geometry = edges_gdf.loc[streetID]["geometry"]
      
    # continue from the to_node or from the from_node    
    if (direction == "to"): intersecting = edges_gdf[(edges_gdf["u"] == to_node) | (edges_gdf["v"] == to_node)]
    else: intersecting = edges_gdf[(edges_gdf["u"] == from_node) | (edges_gdf["v"] == from_node)]
    if (len(intersecting) == 0): return
    
    # check all possible deflection angles with the intersecting roads identified
    for connector in intersecting.itertuples():
        if ((streetID == connector.Index]) | (connector[ix_nID] > 0)): continue # already processed
        to_node_connector, from_node_connector = connector[ix_u], connector[ix_v]
        line_geometry_connector = row_connector[ix_geo]

        # where to go next, in case?
        if (to_node == from_node_connector) | (from_node == from_node_connector): towards = "to"
		else: towards = "to"

        # measuring deflection angle, adding it to the dictionary, if lower than tolerance degrees
        deflection = af.angle_line_geometries((line_geometry, line_geometry_connector, degree = True, deflection = True)
        if (deflection >= tolerance): continue
        else:
            angles[connector.Index] = deflection # dictionary with streetID and angle
            directions_dict[connector.Index] = towards # dictionary with streetID and direction
    
    # No natural continuations
    if len(angles) == 0:
        edges_gdf.set_value(streetID, "naturalID", naturalID)
        return
   
    # selecting the best continuation and continuing in its direction
    else:
        angles_sorted = sorted(angles, key = angles.get)                              
        # taking the streetID of the segment which form the gentlest angle with the segment examined
        matchID = angles_sorted[0] 
        edges_gdf.set_value(streetID, "naturalID", naturalID)
        _natural_roads(matchID, naturalID, directions_dict[matchID], nodes_gdf, edges_gdf)                    
        return                                                                            


class Error(Exception):
   """Base class for other exceptions"""
   pass
class StreetNetworkError(Error):
   """Raised when street network GDFs are not simplified"""
   pass
class epgsError(Error):
   """Raised when epsg code is not provided"""
   pass
    
