import pandas as pd
import numpy as np
import geopandas as gpd

from shapely.geometry import Point, Polygon
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
    
    # Create a new column with the same values as the land_use_field column
    buildings_gdf[new_land_use_field] = buildings_gdf[land_use_field].copy()
    # Create a dictionary to map the old category values to the new strings
    map_dict = {category: strings[n] for n, category in enumerate(categories)}
    # Use the map function and the dictionary to update the new column
    buildings_gdf[new_land_use_field].replace(map_dict, inplace=True)
    
    return buildings_gdf


def land_use_from_other_gdf(buildings_gdf, other_gdf, column, land_use_field):
    """
    It assigns land-use attributes to buildings in a buildings GeoDataFrame, looking for possible matches in "other_gdf", a Polygon or Point GeoDataFrame
    Polygon: Possible matches here means the buildings in "other_gdf" whose area of interesection with the examined building (y), covers at least
    60% of the building's (y) area. The best match is chosen. 
    
    Point: Possible matches are identified when points lie within the buildings_gdf polygons. The most represented category is chosen when more than one points
    lies inside a building footprint.
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        buildings GeoDataFrame
	other_gdf: Point or Polygon GeoDataFrame
		the GeoDataFrame wherein looking for land_use attributes
    column: string
		name of the column in buildings_gdf to which assign the land_use descriptor
    land_use_field: string, 
        name of the column in other_gdf wherein the land_use attribute is stored
   
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        the updated buildings' GeoDataFrame
    """
        
    buildings_gdf = buildings_gdf.copy()    
    buildings_gdf[column] = None
    
    if (other_gdf.iloc[0].geom_type == 'Point')
        other_gdf["nr"] = 1
    
    # spatial index
    sindex = other_gdf.sindex
        
    if (other_gdf.iloc[0].geom_type == 'Point'):
        buildings_gdf[column] = buildings_gdf.geometry.apply(lambda row: _land_use_from_points(row, other_gdf,
                                                                                         sindex, land_use_field))
    else:
        buildings_gdf[column] = buildings_gdf.geometry.apply(lambda row: _land_use_from_polygons(row, other_gdf,
                                                                                              sindex, land_use_field))
    
    return buildings_gdf
    
def _land_use_from_polygons(building_geometry, other_gdf, other_gdf_sindex, land_use_field):
    """
    It assigns land-use attributes to a building, looking for possible matches in a"other_gdf", a Polygon GeoDataFrame.
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        buildings GeoDataFrame
	other_gdf: Polygon GeoDataFrame
		the GeoDataFrame wherein looking for land_use attributes
    other_gdf_sindex: Rtree spatial index
    land_use_field: string
        name of the column in other_gdf wherein the land_use attribute is stored
   
    Returns
    -------
    Object
    """   
    # Find the possible matches
    possible_matches_index = list(other_gdf_sindex.intersection(building_geometry.bounds))
    possible_matches = other_gdf.iloc[possible_matches_index]
    # Keep only the polygons that intersect with the building
    pm = possible_matches[possible_matches.intersects(building_geometry)]
    if len(pm) == 0: 
        return None # no intersecting features in the other_gdf
    
    # calculate area of intersection between building and each possible match
    pm["area_int"] = pm.loc[pm.intersects(building_geometry), 'geometry'].apply(lambda row: row.intersection(building_geometry).area)
    # sort the matches based on the extent of the area of intersection
    pm = pm.sort_values(by="area_int", ascending=False)
    # Assign the match land-use category if the area of intersection covers at least 60% of the building's area
    if pm["area_int"].iloc[0] >= (building_geometry.area * 0.60): 
        return pm[land_use_field].iloc[0]
    
    return None


def _land_use_from_points(building_geometry, other_gdf, other_source_gdf_sindex, land_use_field):
    """
    It assigns land-use attributes to a building, looking for possible matches in a "other_gdf", a Point GeoDataFrame.
     
    Parameters
    ----------
    building_geometry: Polygon
    other_gdf: Point GeoDataFrame
        The GeoDataFrame wherein looking for land_use attributes
    other_gdf_sindex: Rtree spatial index
    land_use_field: string
        name of the column in other_gdf, wherein the land_use attribute is stored
   
    Returns
    -------
    Object
    """

    possible_matches_index = list(other_source_gdf_sindex.intersection(building_geometry.bounds))
    possible_matches = other_source_gdf.iloc[possible_matches_index]
    pm = possible_matches[possible_matches.intersects(building_geometry)]
    
    if (len(pm)==0): 
        return None # no intersecting features in the other_source_gdf
    
    # counting nr of features and using the most represented one
    pm.groupby([land_use_field],as_index=False)["nr"].sum().sort_values(by="nr", ascending=False).reset_index()
    return pm[land_use_field].iloc[0]
    
    