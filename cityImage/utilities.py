import pandas as pd
import numpy as np
import geopandas as gpd
import math
import pyproj

from math import sqrt
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.ops import unary_union, transform
from shapely.affinity import scale
from functools import partial


pd.set_option("precision", 10)

    
def scaling_columnDF(df, i, inverse = False):
    """
    It rescales the values in a dataframe"s from 0 to 1
    
    Parameters
    ----------
    df: pandas dataframe
    i: string (column name)
    ----------
    """
    
    df[i+"_sc"] = (df[i]-df[i].min())/(df[i].max()-df[i].min())
    if inverse == True: 
        df[i+"_sc"] = 1-(df[i]-df[i].min())/(df[i].max()-df[i].min())
        
    
def dict_to_df(list_dict, list_col):

    """
    It takes a list of dictionaries and merge them in a df, as columns.
    
    Parameters
    ----------
    list_dict: list of dictionaries
    list_col: list of str
    
    Returns:
    ----------
    DataFrame
    """
    
    df = pd.DataFrame(list_dict).T
    df.columns = ["d{}".format(i) for i, col in enumerate(df, 1)]
    df.columns = list_col
    
    return df
    
def center_line(line_geometryA, line_geometryB): 

    """
    Given two lines, it constructs the corresponding center line
    
    Parameters
    ----------
    line_geometryA, line_geometryB: LineString
    
    Returns:
    ----------
    LineString
    """
        
    line_coordsA = list(line_geometryA.coords)
    line_coordsB = list(line_geometryB.coords)
        
    if ((line_coordsA[0] == line_coordsB[-1]) | (line_coordsA[-1] == line_coordsB[0])): 
        line_coordsB.reverse()  
    
    if line_coordsA == line_coordsB:
        center_line = LineString([coor for coor in line_coordsA]) 
    else:
        while len(line_coordsA) > len(line_coordsB):
            index = int(len(line_coordsA)/2)
            del line_coordsA[index]
        while len(line_coordsB) > len(line_coordsA):
            index = int(len(line_coordsB)/2)
            del line_coordsB[index]          
    
        new_line = line_coordsA
        for n, i in enumerate(line_coordsA):
            link = LineString([coor for coor in [line_coordsA[n], line_coordsB[n]]])
            np = link.centroid.coords[0]       
            new_line[n] = np
            
        new_line[0] = line_coordsA[0]
        new_line[-1] = line_coordsA[-1]
        center_line = LineString([coor for coor in new_line])

    return center_line

def distance_geometry_gdf(geometry, gdf):
    """
    Given a geometry and a GeoDataFrame, it returns the minimum distance between the geometry and the GeoDataFrame. 
    It provides also the index of the closest geometry in the GeoDataFrame
    
    Parameters
    ----------
    geometry: Point, LineString or Polygon
    gdf: GeoDataFrame
    
    Returns:
    ----------
    tuple
    """
    gdf = gdf.copy()
    gdf["dist"] = gdf.apply(lambda row: geometry.distance(row['geometry']),axis=1)
    geoseries = gdf.iloc[gdf["dist"].argmin()]
    distance  = geoseries.dist
    index = geoseries.name
    return distance, index


def merge_lines(line_geometries):

    """
    Given a list of line_geometries wich are connected by common to and from vertexes, the function infers the sequence, based on the coordinates, and return a merged LineString feature.
    
    Parameters
    ----------
    line_geometries: list of LineString
    
    Returns:
    ----------
    LineString
    """
    
    first = list(line_geometries[0].coords)
    second = list(line_geometries[1].coords)
    coords = []
    
    # determining directions
    reverse = False
    if first[0] == second[0]: 
        reverse = True
        first.reverse()
    if first[-1] == second[-1]: 
        second.reverse()
    if first[0] == second[-1]:
        first.reverse()
        second.reverse()
        reverse = True
    
    coords = first + second
    last = second
    for n,i in enumerate(line_geometries):
        if n < 2: 
            continue
        next_coords = list(i.coords)
        if (next_coords[-1] == last[-1]):
            next_coords.reverse()
            last = next_coords
            
    if reverse:
        coords.reverse()
    return LineString([coor for coor in coords])
            
def envelope_wgs(gdf):
    envelope = gdf.unary_union.envelope.buffer(100)
    coords = mapping(envelope)["coordinates"][0]
    d = [(Point(coords[0])).distance(Point(coords[1])), (Point(coords[1])).distance(Point(coords[2]))]
    distance = max(d)
    project = partial(
        pyproj.transform,
        pyproj.Proj(gdf.crs), # source coordinate system
        pyproj.Proj(init='epsg:4326')) # destination coordinate system

    envelope_wgs = transform(project, envelope)
    return envelope_wgs            
            
def create_hexagon(side_length, x, y):

    """
    Create a hexagon centered on (x, y)
    
    Parameters
    ----------
    side_length: length of the hexagon's edgeline_geometries: list of LineString
    x: x-coordinate of the hexagon's center
    y: y-coordinate of the hexagon's center
    
    Return
    ----------
    Polygon
    """
    
    
    c = [[x + math.cos(math.radians(angle)) * side_length, y + math.sin(math.radians(angle)) * side_length] for angle in range(0, 360, 60)]

    return Polygon(c)


def create_grid(gdf, side_length = 150):
    xmin, ymin,xmax,ymax = gdf.total_bounds # lat-long of 2 corners
    EW = Point(xmin,ymin).distance(Point(xmax,ymin))
    NS = Point(xmin,ymin).distance(Point(xmin,ymax))

    height = int(side_length*1.73)
    width = side_length*2

    cols = list(range(int(np.floor(xmin)), int(np.ceil(xmax)), width))
    rows = list(range(int(np.floor(ymin)), int(np.ceil(ymax)), height))
    rows.reverse()
    polygons = []
    odd = False

    to_reach = cols[-1]
    x = cols[0]
    while (x < to_reach):
        if odd: 
            x = x-side_length/2
        for y in rows:
            if odd: 
                y = y-height/2
            centroid = Polygon([(x,y), (x+side_length, y), (x+side_length, y-side_length), (x, y-side_length)]).centroid       
            polygons.append(create_hexagon(side_length, centroid.coords[0][0], centroid.coords[0][1] ))
        if odd: 
            x = x + width-side_length/2
        else: x = x+width
        odd = not odd


    grid = gpd.GeoDataFrame({'geometry':polygons}, crs = gdf.crs)
    return grid          
    
def rescale_ranges(n, range1, range2):
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]                
            
def rescale_geometry(original_geometry, factor):
 
    rescaled_geometry = scale(original_geometry, xfact= factor, yfact= factor, zfact=factor, origin='center') 
    return rescaled_geometry
            


    
