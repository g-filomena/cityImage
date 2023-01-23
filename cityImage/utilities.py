import pandas as pd
import numpy as np
import geopandas as gpd
import math
import pyproj

from typing import List
from math import sqrt
from shapely.geometry import LineString, Point, Polygon, MultiPoint, mapping
from shapely.ops import unary_union, transform, nearest_points
from shapely.affinity import scale
from shapely.geometry.base import BaseGeometry
from functools import partial
pd.set_option("display.precision", 3)
    
def scaling_columnDF(df, column, inverse = False):
    """
    It rescales the values in a dataframe's columns from 0 to 1
    
    Parameters
    ----------
    df: pandas DataFrame
        a DataFrame
    column: string
        the column name, representing the column to rescale
    inverse: boolean
        if true, rescales from 1 to 0 instead of 0 to 1
    ----------
    """
    
    df[column+"_sc"] = (df[column]-df[column].min())/(df[column].max()-df[column].min())
    if inverse: 
        df[column+"_sc"] = 1-(df[column]-df[column].min())/(df[column].max()-df[column].min())
        
    
def dict_to_df(list_dict, list_col):
    """
    It takes a list of dictionaries and creates from them a pandas DataFrame, where the dictionaries become columns.
    
    Parameters
    ----------
    list_dict: list of dict
        the list of dictionaries
    list_col: list of string
        the corresponding column names to assign to the attached dictionaries
    
    Returns:
    ----------
    df: pandas DataFrame
        the resulting DataFrame
    """
    
    df = pd.DataFrame(list_dict).T
    df.columns = ["d{}".format(i) for i, col in enumerate(df, 1)]
    df.columns = list_col
    
    return df
     
def center_line(line_geometryA, line_geometryB): 
    """
    Given two LineStrings, it derives the corresponding center line
    
    Parameters
    ----------
    line_geometryA: LineString 
        the first line
    line_geometryB: LineString
        the second line
    
    Returns:
    ----------
    center_line: LineString
        the resulting center line
    """
    line_coordsA = list(line_geometryA.coords)
    line_coordsB = list(line_geometryB.coords)
    # If not, reverse the coordinates of B
    if line_coordsA[0] != line_coordsB[-1]:
        line_coordsB = line_coordsB[::-1]

    # Remove the middle point of the longer list until both lists have the same length
    while len(line_coordsA) != len(line_coordsB):
        if len(line_coordsA) > len(line_coordsB):
            del line_coordsA[int(len(line_coordsA)/2)]
        else:
            del line_coordsB[int(len(line_coordsB)/2)]

    # Calculate the center line coordinates
    center_line_coords = [[(a[0] + b[0])/2, (a[1] + b[1])/2] for a, b in zip(line_coordsA, line_coordsB)]

    # Assign the first and last point of the line A to the first and last point of the center line 
    center_line_coords[0] = line_coordsA[0]
    center_line_coords[-1] = line_coordsA[-1]
    center_line = LineString([coor for coor in center_line_coords])

    # Create a LineString object from the center line coordinates

    return center_line

def min_distance_geometry_gdf(geometry, gdf):
    """
    Given a geometry and a geodataframe, this function finds the minimum distance between the 
    geometry and the geodataframe, and returns the index of the closest geometry in the geodataframe.
    
    Parameters
    ----------
    geometry: shapely.geometry.*
        The geometry to find the closest distance to
    gdf: geopandas.GeoDataFrame
        The geodataframe to find the closest distance from
        
    Returns
    -------
    tuple
        a tuple containing the minimum distance and the index of the closest geometry in the geodataframe
    """
    # Convert the geodataframe's geometry to a MultiPoint object
    gdf_geometry = MultiPoint(gdf.geometry)
    
    # Find the minimum distance between the input geometry and the MultiPoint object
    min_dist = geometry.distance(gdf_geometry)
    
    # Get the index of the closest point in the geodataframe
    closest_index = gdf_geometry.nearest_points(geometry)[1]
    
    return min_dist, closest_index

    """
    Given a list of line_geometries wich are connected by common "to" and "from" vertexes, the function infers the sequence, based on the coordinates, 
    and returns a merged LineString feature. As compared to existing shapely functions, this function readjusts the sequence of coordinates if they are not sequential 
    (e.g. 1.segment is: xx - yy, second is yy2 - xx2 and xx == xx2).
    
    Parameters
    ----------
    line_geometries: list of LineString
        the lines
    
    Returns:
    ----------
    newLine: LineString
        the resulting LineString
    """
    
def merge_line_geometries(line_geometries: List[LineString]) -> LineString:
    """
    Given a list of LineString geometries, this function reorders the geometries in the correct sequence based on their starting and ending point,
    and returns a merged LineString feature.
    
    Parameters:
        line_geometries (List[LineString]): A list of LineString geometries to be merged.
        
    Returns:
        LineString: A merged LineString feature.
    """
    
    if not all(isinstance(line, LineString) for line in line_geometries):
        raise ValueError("Input must be a list of LineString geometries")
    if not all(isinstance(line, BaseGeometry) for line in line_geometries):
        raise ValueError("Input must be a list of valid geometries")
    if len(line_geometries) < 2:
        raise ValueError("At least 2 LineStrings are required to merge")
    
    # Create a dictionary to store the "from" and "to" vertexes of each LineString as keys
    lines = {(line.coords[0], line.coords[-1]): line for line in line_geometries}

    line_geometries = sorted(line_geometries, key=lambda line: (line.coords[0], line.coords[-1]))
    merged = []
    first_line = line_geometries.pop(0)
    merged.append(first_line)
    for line in line_geometries:
        if merged[-1].coords[-1] == line.coords[0]:
            merged.append(line)
        elif merged[-1].coords[-1] == line.coords[-1]:
            merged.append(LineString(list(reversed(line.coords))))
    
    return LineString(list(merged))
            
def envelope_wgs(gdf):
    """
    Given a GeoDataFrame it derives its envelope in the WGS coordinate system.
    
    Parameters
    ----------
    gdf: GeoDataFrame
        the geodataframe
    
    Return
    ----------
    envelope_wgs: Polygon
        the resulting envelope
    """
    envelope = gdf.unary_union.envelope.buffer(100)

    project = partial(
        pyproj.transform,
        pyproj.Proj(gdf.crs), # source coordinate system
        pyproj.Proj(init='epsg:4326')) # destination coordinate system

    envelope_wgs = transform(project, envelope)
    return envelope_wgs 
    
def convex_hull_wgs(gdf):
    """
    Given a GeoDataFrame it derives its convex hull in the WGS coordinate system.
    
    Parameters
    ----------
    gdf: GeoDataFrame
        the geodataframe
        
    Return
    ----------
    convex_hull_wgs: Polygon
        the resulting hexagon
    """
    convex_hull = gdf.unary_union.convex_hull
    project = partial(
        pyproj.transform,
        pyproj.Proj(gdf.crs), # source coordinate system
        pyproj.Proj(init='epsg:4326')) # destination coordinate system

    convex_hull_wgs = transform(project, convex_hull)
    return convex_hull_wgs      
        
    
def rescale_ranges(n, range1, range2):
    """
    Given a value n and the range which it belongs to, the function rescale the value, given a different range.
        
    Parameters
    ----------
    n: int, float
        a value
    range1: tuple
        a certain range, e.g. (0, 1) or (10.5, 100). The value n should be within this range
    range2: tuple
        a range, e.g. (0, 1) or (10.5, 100), that should used to rescale the value.
        
    Return
    ----------
    value: float
        the rescaled value
    """
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    value = (delta2 * (n - range1[0]) / delta1) + range2[0]    
    return value
                         
def gdf_from_geometries(geometries, crs):
    """
    The function creates a GeoDataFrame from a list of geometries
    
    Parameters
    ----------
    geometries: list of LineString, Polygon or Points
        The geometries to be included in the GeoDataFrame
        
    Returns
    -------
    gdf: GeoDataFrame
        the resulting GeoDataFrame
    """
    
    df = pd.DataFrame({'geometry': geometries})
    gdf = gpd.GeoDataFrame(df, geometry = df['geometry'], crs = crs)
    gdf['length'] = gdf['geometry'].length
    return gdf
    
def line_at_centroid(line_geometry, offset):
    """
    Given a LineString, it creates a LineString that intersects the given geometry at its centroid.
    The offset determines the distance from the original line.
    This fictional line can be used to count precisely the number of trajectories intersecting a segment.
    This function should be executed per row by means of the df.apply(lambda row : ..) function. 
    
    Parameters
    ----------
    line_geometry: LineString
        A street segment geometry
    offset: float
        The offset from the geometry
    
    Returns
    -------
    LineString
    """
    
    left = line_geometry.parallel_offset(offset, 'left')
    right =  line_geometry.parallel_offset(offset, 'right')
    
    if left.geom_type == 'MultiLineString': 
        left = merge_disconnected_lines(left)
    if right.geom_type == 'MultiLineString': 
        right = merge_disconnected_lines(right)   
    
    if (left.is_empty == True) & (right.is_empty == False): 
        left = line_geometry
    if (right.is_empty == True) & (left.is_empty == False): 
        right = line_geometry
    
    left_centroid = left.interpolate(0.5, normalized = True)
    right_centroid = right.interpolate(0.5, normalized = True)
   
    fict = LineString([left_centroid, right_centroid])
    return(fict)
    
def sum_at_centroid(line_geometry, bus_lines, column):
    """
    Given a LineString geometry, it counts all the geometries in a LineString GeoDataFrame (the GeoDataFrame containing GPS trajectories).
    This function should be executed per row by means of the df.apply(lambda row : ..) function.
        
    Parameters
    ----------
    line_geometry: LineString
        A street segment geometry
    tracks_gdf: LineString GeoDataFrame
        A set of GPS tracks 
    
    Returns
    -------
    int
    """
    
    freq = bus_lines[bus_lines.geometry.intersects(line_geometry)][column].sum()
    return freq
    
