import pandas as pd
import numpy as np
import geopandas as gpd
import math
import pyproj

from math import sqrt
from shapely.geometry import LineString, Point, Polygon, mapping
from shapely.ops import unary_union, transform, nearest_points
from shapely.affinity import scale
from functools import partial


pd.set_option("precision", 10)

    
def scaling_columnDF(df, column, inverse = False):
    """
    It rescales the values in a dataframe"s from 0 to 1
    
    Parameters
    ----------
    df: pandas DataFrame
        a DataFrame
    column: string
        the column name, representing the column to rescale_geometry
    inverse: boolean
        if true, rescales from 1 to 0 instead of 0 to 1
    ----------
    """
    
    df[column+"_sc"] = (df[column]-df[column].min())/(df[column].max()-df[column].min())
    if inverse: 
        df[column+"_sc"] = 1-(df[column]-df[column].min())/(df[column].max()-df[column].min())
        
    
def dict_to_df(list_dict, list_col):

    """
    It takes a list of dictionaries and creates from them a pandas DataFrame a df, with the dictionaries become columns.
    
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
    
    center_line_coords = _center_line_coords(line_geometryA, line_geometryB)    
    line_coordsA = list(line_geometryA.coords)
            
    center_line_coords[0] = line_coordsA[0]
    center_line_coords[-1] = line_coordsA[-1]
    center_line = LineString([coor for coor in center_line_coords])

    return center_line

def _center_line_coords(line_geometryA, line_geometryB):
    """
    Given two LineStrings, it derives the corresponding center line's sequence of coordinates
    
    Parameters
    ----------
    line_geometryA: LineString 
        the first line
    line_geometryB: LineString
        the second line
    
    Returns:
    ----------
    center_line_coords: list
        the resulting center line's sequence of coords
    """

    line_coordsA = list(line_geometryA.coords)
    line_coordsB = list(line_geometryB.coords)
    
    if ((line_coordsA[0] == line_coordsB[-1]) | (line_coordsA[-1] == line_coordsB[0])): 
        line_coordsB.reverse()  
    
    if line_coordsA == line_coordsB:
        center_line_coords = line_coordsA
    
    else:
        while len(line_coordsA) > len(line_coordsB):
            index = int(len(line_coordsA)/2)
            del line_coordsA[index]
        while len(line_coordsB) > len(line_coordsA):
            index = int(len(line_coordsB)/2)
            del line_coordsB[index]      
        
        center_line_coords = line_coordsA
        for n, i in enumerate(line_coordsA):
            link = LineString([coor for coor in [line_coordsA[n], line_coordsB[n]]])
            np = link.centroid.coords[0]           
            center_line_coords[n] = np
    
    return center_line_coords

        
def split_line_at_interpolation(point, line_geometry): #ok
    
    line_coords = list(line_geometry.coords)
    starting_point = Point(line_coords[0])
    np = nearest_points(point, line_geometry)[1]
    distance_start = line_geometry.project(np)
    
    new_line_A = []
    new_line_B = []

    if len(line_coords) == 2:
        new_line_A = [line_coords[0],  np.coords[0]]
        new_line_B = [np.coords[0], line_coords[-1]]
        line_geometry_A = LineString([coor for coor in new_line_A])
        line_geometry_B = LineString([coor for coor in new_line_B])

    else:
        new_line_A.append(line_coords[0])
        new_line_B.append(np.coords[0])

        for n, i in enumerate(line_coords):
            if (n == 0) | (n == len(line_coords)-1): 
                continue
            if line_geometry.project(Point(i)) < distance_start: 
                new_line_A.append(i)
            else: new_line_B.append(i)

        new_line_A.append(np.coords[0])
        new_line_B.append(line_coords[-1])
        line_geometry_A = LineString([coor for coor in new_line_A])
        line_geometry_B = LineString([coor for coor in new_line_B])
    
    return((line_geometry_A, line_geometry_B), np)    

def distance_geometry_gdf(geometry, gdf):
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
    gdf = gdf.copy()
    gdf["dist"] = gdf.apply(lambda row: geometry.distance(row['geometry']),axis=1)
    geoseries = gdf.iloc[gdf["dist"].argmin()]
    distance  = geoseries.dist
    index = geoseries.name
    return distance, index

def merge_lines(line_geometries):
    """
    Given a list of line_geometries wich are connected by common "to" and "from" vertexes, the function infers the sequence, based on the coordinates, 
    and returns a merged LineString feature.
    As compared to existing shapely functions, this readjust the sequence of coordinates if they are 
    not sequential (e.g. 1.segment is: xx - yy, second is yy2 - xx2 and xx == xx2).
    
    Parameters
    ----------
    line_geometries: list of LineString
    
    Returns:
    ----------
    newLine: LineString
        the resulting LineString
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
    newLine = LineString([coor for coor in coords])
    return newLine
            
def envelope_wgs(gdf):
    """
    Given a certain GeoDataFrame it derives its envelope in the WGS coordinate system.
    
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
    Given a certain GeoDataFrame it derives its convex hull in the WGS coordinate system.
    
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
            
def create_hexagon(side_length, x, y):
    """
    Create a hexagon centered on (x, y)
    
    Parameters
    ----------
    side_length: float
        length of the hexagon's edgeline_geometries
    x: float
        x-coordinate of the hexagon's center
    y: float
        y-coordinate of the hexagon's center
    
    Return
    ----------
    polygon: Polygon
        the resulting hexagon
    """
       
    c = [[x + math.cos(math.radians(angle)) * side_length, y + math.sin(math.radians(angle)) * side_length] for angle in range(0, 360, 60)]
    polygon = Polygon(c)
    return polygon


def create_grid(gdf, side_length = 150):
    """
    Create a grid of hexagons, for a given GeoDataFrame's extent.
    
    Parameters
    ----------
    gdf: GeoDataFrame
    side_length: float
        length of the hexagon's edgeline_geometries
    
    Return
    ----------
    grid: Polygon GeoDataFrame
        the resulting grid of hexagons
    """
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
    
            
def rescale_geometry(geometry, factor):
    """
    It rescales a geometry of a certain factor. The same factor is applied to each dimension (x, y,z), from the center of the geometry.
    See shapely.affinity.scale for details
    
    Parameters
    ----------
    geometry: Polygon
        a polygon
    factor: float
        the factor/distance to use for rescaling
    
    Return
    ----------
    rescaled_geometry: Polygon
        the resulting rescaled geometry        
    """
    rescaled_geometry = scale(geometry, xfact= factor, yfact= factor, zfact=factor, origin='center') 
    return rescaled_geometry
            
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
    
