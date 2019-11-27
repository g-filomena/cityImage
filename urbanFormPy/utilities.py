import pandas as pd, numpy as np, geopandas as gpd
import math
from math import sqrt
from shapely.geometry import Point, LineString, MultiLineString

pd.set_option("precision", 10)

    
def scaling_columnDF(df, i, inverse = False):
    
    """
    It rescales the values in a dataframe"s from 0 to 1
    
    Parameters
    df: pandas dataframe
    i: string (column name)
    ----------
    """
	
    df[i+"_sc"] = (df[i]-df[i].min())/(df[i].max()-df[i].min())
    if (inverse == True): df[i+"_sc"] = 1-(df[i]-df[i].min())/(df[i].max()-df[i].min())
        
	
def dict_to_df(list_dict, list_col):

    """
    It takes a list of dictionaries and merge them in a df, as columns.
    
    ----------
    Parameters
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
    
    ----------
    Parameters
	line_geometryA, line_geometryB: LineString
    
    Returns:
    ----------
    LineString
    """
	    
    line_coordsA = list(line_geometryA.coords)
    line_coordsB = list(line_geometryB.coords)
        
    if ((line_coordsA[0] == line_coordsB[-1]) | (line_coordsA[-1] == line_coordsB[0])): line_coordsB.reverse()  
    if line_coordsA == line_coordsB: center_line = LineString([coor for coor in line_coordsA]) 
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

    return LineString([coor for coor in new_line])

def distance_geometry_gdf(geometry, gpd):
    """
    Given a geometry and a GeoDataFrame, it returns the minimum distance between the geometry and the GeoDataFrame. 
	It provides also the index of the closest geometry in the GeoDataFrame
    
    ----------
    Parameters
	geometry: Point, LineString or Polygon
	gpd: GeoDataFrame
    
    Returns:
    ----------
    tuple
    """
    gpd = gpd.copy()
    gpd["dist"] = gpd.apply(lambda row: point.distance(row.geometry),axis=1)
    geoseries = gpd.loc[gpd["dist"].argmin()]
    distance  = geoseries.dist
    index = geoseries.name
    return distance, index


def merge_lines(line_geometries):

    """
    Given a list of line_geometries wich are connected by common to and from vertexes, the function infers the sequence, based on the coordinates and return a merged LineString feature.
	
    ----------
    Parameters
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
    if first[-1] == second[-1]: second.reverse()
    if first[0] == second[-1]:
        first.reverse()
        second.reverse()
        reverse = True
	
    coords = first + second
    last = second
    for n,i in enumerate(line_geometries):
        if n < 2: continue
        next_coords = list(i.coords)
        if (next_coords[-1] == last[-1]) :
            next_coords.reverse()
            last = next_coords
            
    if reverse: coords.reverse()
    return LineString([coor for coor in coords])
            
    
def merge_disconnected_lines(line_geometries):
    """
    Given a list of line_geometries wich are disconnected, the function infers the sequence, based on the coordinates and return a merged LineString feature.
	
    ----------
    Parameters
	line_geometries: list of LineString
    
    Returns:
    ----------
    LineString
    """
	
    new_line = []
    for n, i in enumerate(list_lines):
        coords = list(i.coords)
        if n < len(list_lines)-1: coords.append(list_lines[n+1].coords[-1])
        new_line = new_line + coords
    
	return LineString([coor for coor in new_line])

            
            
            
            
            
            


    
