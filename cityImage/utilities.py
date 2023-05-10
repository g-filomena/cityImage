import pandas as pd
import numpy as np
import geopandas as gpd
import math
import pyproj

from typing import List
from math import sqrt
from shapely.geometry import LineString, Point, Polygon, MultiPoint, mapping
from shapely.ops import unary_union, transform, nearest_points, split
from shapely.affinity import scale
from shapely.geometry.base import BaseGeometry
from functools import partial
import pyvista as pv
pd.set_option("display.precision", 3)
    
def scaling_columnDF(series, inverse = False):
    """
    It rescales the values in a dataframe's columns from 0 to 1
    
    Parameters
    ----------
    series: pd.Series
        the pd.Series to rescale
    inverse: boolean
        if true, rescales from 1 to 0 instead of 0 to 1
    ----------
    """
    
    scaled = pd.Series((series-series.min())/(series.max()-series.min()))
    if inverse: 
        scaled = 1-scaled
    return scaled
        
    
def dict_to_df(list_dict, list_col):
    """
    It takes a list of dictionaries and creates from them a pandas DataFrame. Each dictionary becomes a Series, where keys are rows,
    values cell values. The column names are renamed trough a list of strings (list_col).
    
    Parameters
    ----------
    list_dict: list of dict
        the list of dictionaries
    list_col: list of string
        the corresponding column names to assign to the series
    
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
    Given a geometry and a GeoDataFrame, it returns the minimum distance between the geometry and the GeoDataFrame. 
    It provides also the index of the closest geometry in the GeoDataFrame.
    
    Parameters
    ----------
    geometry: Point, LineString or Polygon
    gdf: GeoDataFrame
    
    Returns:
    ----------
    distance, index: tuple
        the closest distance from the geometry, and the index of the closest geometry in the gdf
    """
    
    sindex = gdf.sindex
    closest = sindex.nearest(geometry, return_distance = True)
    iloc = closest[0][1][0]
    distance = closest[1][0]
    index = gdf.iloc[iloc].name 
   
    return distance, index
 
def split_line_at_MultiPoint(line_geometry, intersections, z = 0.0):   
    """
    The function checks whether the coordinates of Point(s) in a Point Collections coordinate are part of the sequence of coordinates of a LineString.
    When this has been ascerted or fixed, the LineString line_geometry is split at each of the intersecting points in the collection.
    
    The input intersection, must be an actual intersection.
               
    Parameters
    ----------
    line_geometry: LineString
        the LineString which has to be split
    intersections: MultiPoint
        the intersecting points
        
    Returns
    -------
    lines: List of MultiLineString
        the resulting segments composing the original line_geometry
    """
    for point in intersections:
        new_line_coords = list(line_geometry.coords)
        for n, v in enumerate(new_line_coords):
            if n == 0: 
                continue
            line = LineString([Point(new_line_coords[n-1]), Point(v)])
            if ((point.intersects(line)) | (line.distance(point) < 1e-8)):
                new_line_coords.insert(n, point.coords[0])
                break
        line_geometry = LineString([coor for coor in new_line_coords])
    
    intersections = MultiPoint(intersections)
    lines = split(line_geometry, intersections)
    lines_list = [line for line in lines.geoms]
    
    if z != None:
        lineZ = [LineString([(coords[0], coords[1], 3) for coords in line.coords]) for line in lines_list]
        
    return lines_list

 
def merge_line_geometries(line_geometries):
    """
    Given a list of LineString geometries, this function reorders the geometries in the correct sequence based on their starting and ending point,
    and returns a merged LineString feature.
    
    Parameters:
    ----------
    line_geometries: List of LineString
        A list of LineString geometries to be merged
        
    Returns:
    ----------
        LineString: The merged LineString feature
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
    Given a value n and the range which it belongs to, the function rescale the value, between a different range.
        
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
    This fictional line can be used to count precisely the number of trajectories/features intersecting a segment.
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
    
    if (left.is_empty) & (not right.is_empty): 
        left = line_geometry
    if (right.is_empty) & (not left.is_empty): 
        right = line_geometry
    
    left_centroid = left.interpolate(0.5, normalized = True)
    right_centroid = right.interpolate(0.5, normalized = True)
   
    return LineString([left_centroid, right_centroid])
    
def sum_at_centroid(line_geometry, lines_gdf, column):
    """
    Given a LineString geometry, it sums the column values of all the features in a LineString GeoDataFrame intersecting the line.
    This function should be executed per row by means of the df.apply(lambda row : ..) function.
        
    Parameters
    ----------
    line_geometry: LineString
        A street segment geometry
    lines_gdf: LineString GeoDataFrame
        The GeoDataFrame  
    column: string
        The name of the column
    
    Returns
    -------
    int
    """
    
    return lines_gdf[lines_gdf.geometry.intersects(line_geometry)][column].sum()

def polygons_gdf_multiparts_to_singleparts(polygons_gdf):
    """    
    The function extracts the individual parts of the multi-part geometries and creates a new GeoDataFrame with single part geometries.
            
    Parameters
    ----------
    polygons_gdf: GeoDataFrame
        GeoDataFrame containing building footprint geometries
    
    Returns
    -------
    single_parts_gdf: Polygon GeoDataFrame
        the new GeoDataFrame with simplified single part building footprint geometries
    """  
    
    polygons_gdf = polygons_gdf.copy()
    single_parts = gpd.geoseries.GeoSeries([geom for geom in polygons_gdf.unary_union.geoms])
    single_parts_gdf = gpd.GeoDataFrame(geometry=single_parts, crs = polygons_gdf.crs)
    
    return single_parts_gdf

def fix_multiparts_LineString_gdf(gdf):
    
    gdf = gdf.copy()

    if 'MultiLineString' in gdf.geometry.type.unique():
        condition = (gdf.geometry.type == 'MultiLineString')
        gdf.loc[condition, 'geometry'] = pd.Series([linemerge(geo) for geo in gdf[condition].geometry], index = gdf[condition].index)
        
        if 'MultiLineString' in gdf.geometry.type.unique():
            multi_gdf = gdf[gdf.geometry.type == 'MultiLineString'].copy()
            gdf = gdf[gdf.geometry.type == 'LineString']
            multi_gdf = multi_gdf.explode(ignore_index = True)
            gdf = gdf.append(multi_gdf, ignore_index = True)

            
    return gdf 
    
def polygon_2d_to_3d(building_polygon, base, height):
    """
    Convert a 2D polygon to a 3D polygon. This function takes a 2D polygon (building_polygon) and extrudes it into 3D space, creating a pv.PolyData,
    creating a 3D polygon with a base and a height elevation.
    
    Parameters
    ----------
    building_polygon (shapely.geometry.Polygon): 
        2D polygon to be extruded
    base (float): 
        base height of the 3D polygon
    height (float): 
        height of the 3D polygon
    
    Returns:
    ----------
    pv.PolyData: A 3D polygon
    """
    
    def reorient_coords(xy):
        """
        Reorient the coordinates of the polygon
        
        This function reorients the coordinates of a polygon if the polygon
        has counterclockwise orientation.
        
        Args:
        xy (list): List of tuples, each representing a coordinate of the polygon
        
        Returns:
        list: Reoriented list of tuples representing the polygon's coordinates
        """
        value = 0
        for i in range(len(xy)):
            x1, y1 = xy[i]
            x2, y2 = xy[(i+1)%len(xy)]
            value += (x2-x1)*(y2+y1)
        if value > 0:
            return xy
        else:
            return xy[::-1]
    
    poly_points = building_polygon.exterior.coords
       
    # Reorient the coordinates of the polygon
    xy = reorient_coords(poly_points)
    # Create 3D coordinates with the base height
    xyz_base = [(x,y,base) for x,y in xy]
    # Create faces of the polygon
    faces = [len(xyz_base), *range(len(xyz_base))]
    # Create the 3D polygon using pyvista
    polygon = pv.PolyData(xyz_base, faces=faces)
    
    # Extrude the 3D polygon to the specified height
    return polygon.extrude((0, 0, height - base), capping=True)
    
