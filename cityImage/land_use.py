import pandas as pd
import numpy as np
import geopandas as gpd

from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping, MultiLineString
from shapely.ops import cascaded_union, linemerge
from scipy.sparse import linalg
pd.set_option("display.precision", 3)

def classify_land_use(buildings_gdf, new_land_use_field, land_use_field, categories, strings):
    """
    The function reclassifies land-use descriptors in a land-use field according to the categorisation presented below. 
    (Not exhaustive)
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
		the buildings GeoDataFrame
    land_use: string
		the land use field in the buildings_gdf
   
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        the updated buildings' GeoDataFrame
    """
    
    buildings_gdf = buildings_gdf.copy()
    
    # reclassifying: replacing original values with relative categories
    buildings_gdf[new_land_use_field] = buildings_gdf[land_use_field]
    for n, category in enumerate(categories):
        buildings_gdf[new_land_use_field] = buildings_gdf[new_land_use_field].map(lambda x: strings[n] if x in category else x)
    
    return buildings_gdf

def land_use_from_polygons(buildings_gdf, other_source_gdf, column, land_use_field):
    """
    It assigns land-use attributes to buildings in a buildings GeoDataFrame, looking for possible matches in "other_source_gdf", a Polygon GeoDataFrame
    Possible matches here means the buildings in "other_source_gdf" whose area of interesection with the examined building (y), covers at least
    60% of the building's (y) area. The best match is chosen. 
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        buildings GeoDataFrame
	other_source_gdf: Polygon GeoDataFrame
		the GeoDataFrame wherein looking for land_use attributes
    column: string
		name of the column in buildings_gdf to which assign the land_use descriptor
    land_use_field: string, name of the column in other_source_gdf wherein the land_use attribute is stored
   
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        the updated buildings' GeoDataFrame
    """
    
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf[column] = None
    # spatial index
    sindex = other_source_gdf.sindex
    buildings_gdf[column] = buildings_gdf.apply(lambda row: _assign_land_use_from_polygons(row["geometry"], other_source_gdf,
                                                                                           sindex, land_use_field), axis = 1)
    
    return buildings_gdf
    
def _assign_land_use_from_polygons(building_geometry, other_source_gdf, other_source_gdf_sindex, land_use_field):
    """
    It assigns land-use attributes to a building, looking for possible matches in "other_source_gdf".
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        buildings GeoDataFrame
	other_source_gdf: Polygon GeoDataFrame
		the GeoDataFrame wherein looking for land_use attributes
    other_source_gdf_sindex: Rtree spatial index
    land_use_field: string
        name of the column in other_source_gdf wherein the land_use attribute is stored
   
    Returns
    -------
    Object
    """   
    possible_matches_index = list(other_source_gdf_sindex.intersection(building_geometry.bounds)) # looking for intersecting geometries
    possible_matches = other_source_gdf.iloc[possible_matches_index]
    pm = possible_matches[possible_matches.intersects(building_geometry)]
    pm["area"] = 0.0
    if (len(pm) == 0): 
        return None# no intersecting features in the other_source_gdf

    for row in pm.itertuples(): # for each possible candidate, computing the extension of the area of intersection
        other_geometry = pm.loc[row.Index]['geometry']
        try:
            overlapping_area = other_geometry.intersection(building_geometry).area
        except: 
            continue
        pm.at[row.Index, "area"] = overlapping_area

    # sorting the matches based on the extent of the area of intersection
    pm = pm.sort_values(by="area", ascending=False).reset_index()
    # assigning the match land-use category if the area of intersection covers at least 60% of the building's areas
    if (pm["area"].iloc[0] >= (building_geometry.area * 0.60)): 
        return pm[land_use_field].iloc[0]
     
    return None



def land_use_from_points(buildings_gdf, other_source_gdf, column, land_use_field):
    """
    It assigns land-use attributes to buildings in "buildings_gdf", looking for possible matches in "other_source_gdf", a Point GeoDataFrame.
    Possible matches means features in "other_source_gdf" which lies within the examined building's area.
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        buildings GeoDataFrame
    other_source_gdf: Point GeoDataFrame
        The GeoDataFrame wherein looking for land_use attributes
    column: string
        name of the column in buildings_gdf to which assign the land_use descriptor
    land_use_field: string
        name of the column in other_source_gdf, wherein the land_use attribute is stored
   
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        the updated buildings' GeoDataFrame
    """
    
    buildings_gdf = buildings_gdf.copy()    
    other_source_gdf["nr"] = 1
    buildings_gdf[column] = None
    sindex = other_source_gdf.sindex
    buildings_gdf[column] = buildings_gdf.apply(lambda row: _assign_land_use_from_points(row["geometry"], other_source_gdf,
                                                                                         sindex, land_use_field), axis = 1)
               
    return buildings_gdf

def _assign_land_use_from_points(building_geometry, other_source_gdf, other_source_gdf_sindex, land_use_field):
    """
    It assigns land-use attributes to a building, looking for possible matches in "other_source_gdf", a Point GeoDataFrame.
     
    Parameters
    ----------
    building_geometry: Polygon
    other_source_gdf: Point GeoDataFrame
        The GeoDataFrame wherein looking for land_use attributes
    other_source_gdf_sindex: Rtree spatial index
    land_use_field: string
        name of the column in other_source_gdf, wherein the land_use attribute is stored
   
    Returns
    -------
    Object
    """

    possible_matches_index = list(other_source_gdf_sindex.intersection(building_geometry.bounds))
    possible_matches = other_source_gdf.iloc[possible_matches_index]
    pm = possible_matches[possible_matches.intersects(building_geometry)]
    
    if (len(pm)==0): 
        return None # no intersecting features in the other_source_gdf
    
    # counting nr of features
    pm.groupby([land_use_field],as_index=False)["nr"].sum().sort_values(by="nr", ascending=False).reset_index()
    return pm[land_use_field].iloc[0]