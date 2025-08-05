import pandas as pd
import numpy as np
import geopandas as gpd

from shapely.geometry import Point, Polygon
pd.set_option("display.precision", 3)

def classify_land_use(buildings_gdf, raw_land_use_column, new_land_use_column, categories, strings):
    """
    Reclassifies land-use descriptors in a land-use field according to 
    the provided categories. Handles both single values and lists of values.

    Parameters
    ----------
    buildings_gdf : GeoDataFrame (Polygon)
        The buildings GeoDataFrame.
    raw_land_use_column : str
        Column in buildings_gdf containing land use values 
        (can be strings or lists of strings).
    new_land_use_column : str
        Name of the new column to store the reclassified land use.
    categories : list of lists
        Each sublist contains the original land use values that should 
        be grouped together into a category.
    strings : list of str
        The new land use category names corresponding to `categories`.

    Returns
    -------
    buildings_gdf : GeoDataFrame
        Updated GeoDataFrame with a new column `new_land_use_column` containing 
        reclassified land uses (lists if multiple values, single values otherwise).
    """

    buildings_gdf = buildings_gdf.copy()

    # Helper: reclassify a single value
    def reclassify_single(value):
        for idx, category in enumerate(categories):
            if value in category:
                return strings[idx]
        return value

    # Helper: reclassify a single value or list of values
    def reclassify_value(value):
        if isinstance(value, list):
            reclassified = [reclassify_single(v) for v in value]
            # Remove duplicates (order not preserved)
            return list(set(reclassified))
        else:
            return reclassify_single(value)

    # Apply reclassification to each row
    buildings_gdf[new_land_use_column] = buildings_gdf[raw_land_use_column].apply(reclassify_value)

    return buildings_gdf

def land_use_from_other_gdf(buildings_gdf, other_gdf, new_land_use_column, other_land_use_column, min_overlap_threshold = 0.30):
    """
    Assign land-use attributes to buildings in a buildings GeoDataFrame, 
    matching against polygons or points from another GeoDataFrame.

    - Polygon mode: Possible matches are polygons in "other_gdf" where 
      the area of intersection with the building covers at least 60% of the building area. 
      All qualifying land_use values are collected into a list. 
      If no match, returns an empty list.

    - Point mode: Possible matches are points lying inside the building polygon. 
      All distinct land_use categories found are returned as a list. 
      If no match, returns an empty list.
     
    Parameters
    ----------
    buildings_gdf : Polygon GeoDataFrame
        Buildings' GeoDataFrame.
    other_gdf : Point or Polygon GeoDataFrame
        GeoDataFrame to search for land_use attributes.
    new_land_use_column : str
        Name of the column to create/overwrite in buildings_gdf for assigned land_use(s).
    other_land_use_column : str
        Column name in other_gdf where land_use attribute(s) are stored.
    min_overlap_threshold : float
        Minimum required overlap between geometries, expressed as a fraction between 0.01 and 1.00. This threshold defines the minimum proportion of a building's area that must 
        intersect with another polygon for it to be considered a match.

    Returns
    -------
    buildings_gdf : Polygon GeoDataFrame
        Updated buildings GeoDataFrame with `new_land_use_column` populated 
        with a list of values (empty list if no matches).
    """
    
    if buildings_gdf.crs != other_gdf.crs:
        raise ValueError("CRS mismatch: buildings_gdf and other_gdf must have the same CRS")

    buildings_gdf = buildings_gdf.copy()
    buildings_gdf[new_land_use_column] = [[] for _ in range(len(buildings_gdf))]

    geom_type = other_gdf.iloc[0].geometry.geom_type
    sindex = other_gdf.sindex

    if geom_type == 'Point':
        def _land_use_from_points_list(building_geometry, other_gdf, sindex, other_land_use_column):
            possible_matches_idx = list(sindex.intersection(building_geometry.bounds))
            possible_matches = other_gdf.iloc[possible_matches_idx]
            matches = possible_matches[possible_matches.intersects(building_geometry)]
            if matches.empty:
                return []
            return list(matches[other_land_use_column].unique())

        buildings_gdf[new_land_use_column] = buildings_gdf.geometry.apply(
            lambda row: _land_use_from_points_list(row, other_gdf, sindex, other_land_use_column)
        )

    else:  # Polygon
        def _land_use_from_polygons_list(building_geometry, other_gdf, sindex, other_land_use_column):
            possible_matches_idx = list(sindex.intersection(building_geometry.bounds))
            possible_matches = other_gdf.iloc[possible_matches_idx]
            matches = possible_matches[possible_matches.intersects(building_geometry)]
            if matches.empty:
                return []

            selected = []
            for _, match in matches.iterrows():
                intersection_area = building_geometry.intersection(match.geometry).area
                if intersection_area >= min_overlap_threshold * building_geometry.area:
                    selected.append(match[other_land_use_column])

            return list(set(selected)) if selected else []

        buildings_gdf[new_land_use_column] = buildings_gdf.geometry.apply(
            lambda row: _land_use_from_polygons_list(row, other_gdf, sindex, other_land_use_column)
        )

    return buildings_gdf

