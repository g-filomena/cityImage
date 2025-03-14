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
		The buildings GeoDataFrame.
    land_use: string
		The land use field in the buildings_gdf.
   
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        The updated buildings' GeoDataFrame.
    """
    
    buildings_gdf = buildings_gdf.copy()
    
    # Create a new column with the same values as the land_use_field column
    buildings_gdf[new_land_use_field] = buildings_gdf[land_use_field].copy()
    # reclassifying: replacing original values with relative categories
    buildings_gdf[new_land_use_field] = buildings_gdf[land_use_field]
    
    for n, category in enumerate(categories):
        buildings_gdf[new_land_use_field] = buildings_gdf[new_land_use_field].map(lambda x: strings[n] if x in category else x)
    
    return buildings_gdf


def land_use_from_other_gdf(buildings_gdf, other_gdf, new_land_use_field, land_use_field):
    """
    It assigns land-use attributes to buildings in a buildings GeoDataFrame, looking for possible matches in "other_gdf", a Polygon or Point GeoDataFrame
    Polygon: Possible matches here means the buildings in "other_gdf" whose area of interesection with the examined building (y), covers at least
    60% of the building's (y) area. The best match is chosen. 
    
    Point: Possible matches are identified when points lie within the buildings_gdf polygons. The most represented category is chosen when more than one points
    lies inside a building footprint.
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
		The buildings GeoDataFrame.
	other_gdf: Point or Polygon GeoDataFrame
		The GeoDataFrame wherein looking for land_use attributes.
    new_land_use_field: str
		Name of the column in buildings_gdf to which assign the land_use descriptor.
    land_use_field: str 
        Name of the column in other_gdf wherein the land_use attribute is stored.
   
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        The updated buildings' GeoDataFrame.
    """
        
    buildings_gdf = buildings_gdf.copy()    
    buildings_gdf[new_land_use_field] = None
    
    if (other_gdf.iloc[0].geometry.geom_type == 'Point'):
        other_gdf["nr"] = 1
    
    # spatial index
    sindex = other_gdf.sindex
        
    if (other_gdf.iloc[0].geometry.geom_type == 'Point'):
        buildings_gdf[new_land_use_field] = buildings_gdf.geometry.apply(lambda row: _land_use_from_points(row, other_gdf,
                                                                                         sindex, land_use_field))
    else:
        buildings_gdf[new_land_use_field] = buildings_gdf.geometry.apply(lambda row: _land_use_from_polygons(row, other_gdf,
                                                                                              sindex, land_use_field))
    return buildings_gdf
    
def _land_use_from_polygons(building_geometry, other_gdf, other_gdf_sindex, land_use_field):
    """
    It assigns land-use attributes to a building, looking for possible matches in a"other_gdf", a Polygon GeoDataFrame.
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
		The buildings GeoDataFrame.
	other_gdf: Polygon GeoDataFrame
		The GeoDataFrame wherein looking for land_use attributes
    other_gdf_sindex: Spatial Index
        The Spatial Index on the other GeoDataFrame.
    land_use_field: str
        name of the column in other_gdf wherein the land_use attribute is stored.
   
    Returns
    -------
    Object
        The assigned land-use attribute or None if no match is found.
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
        A building's geometry.
    other_gdf: Point GeoDataFrame
        The GeoDataFrame wherein looking for land_use attributes.
    other_gdf_sindex: Spatial Index
        The Spatial Index on the other GeoDataFrame.
    land_use_field: str
        name of the column in other_gdf wherein the land_use attribute is stored.
   
    Returns
    -------
    Object
        The assigned land-use attribute or None if no match is found.
    """

    possible_matches_index = list(other_source_gdf_sindex.intersection(building_geometry.bounds))
    possible_matches = other_source_gdf.iloc[possible_matches_index]
    pm = possible_matches[possible_matches.intersects(building_geometry)]
    
    if (len(pm)==0): 
        return None # no intersecting features in the other_source_gdf
    
    # counting nr of features and using the most represented one
    pm.groupby([land_use_field],as_index=False)["nr"].sum().sort_values(by="nr", ascending=False).reset_index()
    return pm[land_use_field].iloc[0]