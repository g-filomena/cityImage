import osmnx as ox
import pandas as pd
import numpy as np
import geopandas as gpd
import pyvista as pv

from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping, MultiLineString
from shapely.ops import linemerge, unary_union
from scipy.sparse import linalg
pd.set_option("display.precision", 3)

import concurrent.futures
from .utilities import scaling_columnDF, downloader
from .visibility import visibility_polygon2d, compute_3d_sight_lines
from .angles import get_coord_angle

def get_buildings_fromFile(input_path, crs, case_study_area = None, distance_from_center = 1000, min_area = 200, min_height = 5, height_field = None, base_field = None, 
    land_use_field = None):
    """
    Reads building footprints from a file and returns two GeoDataFrames: 
    1) buildings within the case-study area, and 
    2) a larger area containing adjacent buildings ("obstructions").

    The case-study area can be specified either by providing a polygon (`case_study_area`)
    or a distance buffer from the centroid of the loaded buildings. 
    If neither is provided, all buildings are included in both outputs.

    Height, base elevation, and land use fields from the source can be mapped to standard columns.

    Parameters
    ----------
    input_path : str
        Path to the building footprint file (.shp or .gpkg).
    crs : str, or pyproj.CRS
        Coordinate Reference System for the study area. Can be a string (e.g. 'EPSG:32633'), or a pyproj.CRS object.
    case_study_area : shapely Polygon or None, optional
        Polygon defining the area of interest (case-study). If None, uses `distance_from_center`.
    distance_from_center : float or None, optional
        If `case_study_area` is None, a circular buffer of this radius (in CRS units) around the centroid of all buildings defines the case-study area.
        Default is 1000.
    min_area : float, optional
        Minimum area threshold (in CRS units, e.g. square meters) for a building to be included. Default is 200.
    min_height : float, optional
        Minimum building height threshold for inclusion. Default is 5.
    height_field : str or None, optional
        Name of the field containing building heights in the input data.
    base_field : str or None, optional
        Name of the field containing building base elevations in the input data. If None, base is set to 0.0.
    land_use_field : str or None, optional
        Name of the field containing land use information in the input data.

    Returns
    -------
    buildings_gdf : GeoDataFrame
        GeoDataFrame of buildings in the case-study area.
    obstructions_gdf : GeoDataFrame
        GeoDataFrame of all valid buildings (including the case-study area and adjacent "obstructions").

    Notes
    -----
    - Buildings with area < `min_area` or height < `min_height` are dropped.
    - Output columns: 'height', 'base', 'geometry', 'area', 'land_use_raw', 'buildingID'.
    """
    
    obstructions_gdf = gpd.read_file(input_path).to_crs(crs)  
    
    # computing area, reassigning columns
    obstructions_gdf["area"] = obstructions_gdf["geometry"].area

    if height_field is not None: 
       obstructions_gdf["height"] = obstructions_gdf[height_field]
    if base_field is None: 
        obstructions_gdf["base"] = 0.0
    else: 
        obstructions_gdf["base"] = obstructions_gdf[base_field]
    if land_use_field is not None: 
        obstructions_gdf["land_use_raw"] = obstructions_gdf[land_use_field]
    else:
        obstructions_gdf["land_use_raw"] = None

    # dropping small buildings and buildings with null height
    obstructions_gdf = obstructions_gdf[(obstructions_gdf["area"] >= min_area) & (obstructions_gdf["height"] >= min_height)]
    obstructions_gdf = obstructions_gdf[["height", "base","geometry", "area", "land_use_raw"]]
    # assigning ID
    obstructions_gdf["buildingID"] = obstructions_gdf.index.values.astype(int)
    
    # if case-study area and distance not defined
    if (case_study_area is None) and (distance_from_center is None or distance_from_center == 0):
        buildings_gdf = obstructions_gdf.copy()
        return buildings_gdf, obstructions_gdf
    if (case_study_area is None):     # define a case study area
        case_study_area = obstructions_gdf.geometry.unary_union.centroid.buffer(distance_from_center)
    buildings_gdf = obstructions_gdf[obstructions_gdf.geometry.within(case_study_area)]

    return buildings_gdf, obstructions_gdf
    
def get_buildings_fromOSM(OSMplace, download_method: str, crs = None, distance = 1000):
    """    
    The function downloads and cleans building footprint geometries and create a buildings GeoDataFrames for the area of interest.
    The function exploits OSMNx functions for downloading the data as well as for projecting it.
    The land use classification for each building is extracted. Only relevant columns are kept.   
            
    Parameters
    ----------
    OSMplace: str, tuple, Shapely Polygon
        Name of cities or areas in OSM: 
        - when using "distance_from_point" please provide a (lat, lon) tuple to create the bounding box around it; 
        - when using "distance_from_address" provide an existing OSM address; 
        - when using "OSMplace" provide an OSM place name. The query must be geocodable and OSM must have polygon boundaries for the geocode result.  
        - when using "polygon" please provide a Shapely Polygon in unprojected latitude-longitude degrees (EPSG:4326) CRS;
    download_method: str, {"distance_from_address", "distance_from_point", "OSMplace", "polygon"}
        It indicates the method that should be used for downloading the data.
    crs : str, or pyproj.CRS
        Coordinate Reference System for the study area. Can be a string (e.g. 'EPSG:32633'), or a pyproj.CRS object.
    distance: float
        Used when download_method == "distance from address" or == "distance from point".
    
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        The buildings GeoDataFrame.
    """   
    columns_to_keep = ['amenity', 'building', 'geometry', 'historic', 'land_use_raw']
    tags = {"building": True}
    buildings_gdf = downloader(place = OSMplace, download_method = download_method, tags = tags, distance = distance)
    
    if crs is None:
        buildings_gdf = ox.projection.project_gdf(buildings_gdf)
    else:
        buildings_gdf = buildings_gdf.to_crs(crs)

    buildings_gdf['land_use_raw'] = None
    buildings_gdf['land_use_raw'] = buildings_gdf.filter(regex='^building:use:').apply(lambda x: x.name[13:] if x.notnull().any() else None)
    buildings_gdf.drop(columns=[col for col in buildings_gdf.columns if col not in columns_to_keep], inplace=True)

    # remove the empty geometries
    buildings_gdf = buildings_gdf[~buildings_gdf['geometry'].is_empty]
    # replace 'yes' with NaN in 'building' column
    buildings_gdf['building'] = buildings_gdf['building'].replace('yes', np.nan)
    # fill missing values in 'building' column with 'amenity' values
    buildings_gdf['building'] = buildings_gdf['building'].fillna(value=buildings_gdf['amenity'])

    # fill missing values in 'land_use_raw' column with 'building' values
    # Create a mask for rows where the 'building' column is NA
    mask = buildings_gdf['building'].isna()
    # Use the mask to assign values from 'amenity' to 'building' only where 'building' is NA
    buildings_gdf.loc[mask, 'building'] = buildings_gdf.loc[mask, 'amenity']

    buildings_gdf['land_use_raw'] = buildings_gdf['land_use_raw'].fillna(value=buildings_gdf['building'])
    # fill remaining missing values in 'land_use_raw' column with 'residential'
    buildings_gdf['land_use_raw'] = buildings_gdf['land_use_raw'].fillna(value='residential')

    buildings_gdf = buildings_gdf[['geometry', 'historic', 'land_use_raw']]
    buildings_gdf['area'] = buildings_gdf.geometry.area
    buildings_gdf = buildings_gdf[buildings_gdf['area'] >= 50] 
    
    # reset index
    buildings_gdf = buildings_gdf.reset_index(drop = True)
    buildings_gdf['buildingID'] = buildings_gdf.index.values.astype('int')  
    
    return buildings_gdf

def assign_building_heights_from_other_gdf(buildings_gdf, detailed_buildings_gdf, crs, min_overlap=0.4):
    """
    Assigns 'base' and 'height' attributes to each building in `buildings_gdf` by extracting
    information from a more detailed building GeoDataFrame (`detailed_buildings_gdf`).

    The assignment logic is as follows:
    - If a building fully contains a detailed building, assign the lowest 'base' and highest 'height' among contained buildings.
    - If a detailed building contains a main building, assign the lowest 'base' and highest 'height' among containers.
    - If a detailed building intersects only one main building, assign its values to that building.
    - If a detailed building intersects multiple main buildings, assign its values to the one with the highest overlap (if overlap ≥ `min_overlap`).
    - The overlap ratio is defined as (intersection_area / detailed_building_area).
    - The reverse intersection is also considered: if a main building overlaps multiple detailed buildings, attributes are borrowed from the one with the highest overlap ratio (if overlap ≥ `min_overlap`).

    Parameters
    ----------
    buildings_gdf : GeoDataFrame
        GeoDataFrame of main building polygons (to be assigned base/height).
    detailed_buildings_gdf : GeoDataFrame
        More detailed building dataset with valid 'base' and 'height' attributes.
    crs : str, or pyproj.CRS
        Coordinate Reference System for the output GeoDataFrame. Can be a string (e.g. 'EPSG:32633'), or a pyproj.CRS object.
    min_overlap : float, optional
        Minimum required overlap ratio (as a fraction, default 0.4).

    Returns
    -------
    GeoDataFrame
        Copy of `buildings_gdf` with updated 'base' and 'height' columns.

    Raises
    ------
    ValueError
        If CRS of the input GeoDataFrames does not match the provided CRS.
    """
    
    # Ensure CRS matches
    if (buildings_gdf.crs != crs) or (detailed_buildings_gdf.crs !=crs):
        raise ValueError(f"CRS mismatch: buildings_gdf ({buildings_gdf.crs}) and detailed_buildings_gdf ({detailed_buildings_gdf.crs}) must have the same CRS.")

    buildings_gdf = buildings_gdf.copy()
    detailed_buildings_gdf = gdf_multipolygon_to_polygon(detailed_buildings_gdf)
    
    buildings_gdf["base"] = 9999.0 
    buildings_gdf["height"] = -9999.0 
        
    # # **Step 1: Handle Full Containment (Buildings that contain detailed ones)**
    # **Step 1: Identify buildings that contain detailed buildings**
    containment = gpd.sjoin(buildings_gdf, detailed_buildings_gdf, predicate="contains", how="left")

    # Compute min(base) and max(height) for each containing building
    contained_bases = containment.groupby(containment.index)["base_right"].min()
    contained_height = containment.groupby(containment.index)["height_right"].max()

    # Apply updates for contained buildings
    buildings_gdf["base"] = buildings_gdf["base"].combine(contained_bases, min)
    buildings_gdf["height"] = buildings_gdf["height"].combine(contained_height, max)
    buildings_gdf.index = buildings_gdf.index.astype(int)
    
    # **Step 2: Identify detailed buildings that contain buildings_gdf**
    reverse_containment = gpd.sjoin(detailed_buildings_gdf, buildings_gdf, predicate="contains", how="left")
    container_bases = reverse_containment.groupby("index_right")["base_left"].min()
    container_height = reverse_containment.groupby("index_right")["height_left"].max()
    
    buildings_gdf["base"] = buildings_gdf["base"].combine(container_bases, min)
    buildings_gdf["height"] = buildings_gdf["height"].combine(container_height, max)
    buildings_gdf.index = buildings_gdf.index.astype(int)

    # **Step 3: Handle Intersections (Buildings that partially overlap detailed ones)** 
    buildings_gdf['geo_check'] = buildings_gdf.geometry
    buildings_gdf['ix'] = buildings_gdf.index
    
    intersections = gpd.sjoin(detailed_buildings_gdf, buildings_gdf, predicate="intersects", how="left")
    intersections = intersections[intersections.geo_check != None]
    
    # Compute intersection area
    intersections["area_intersection"] = intersections.apply(lambda row: row["geometry"].intersection(row["geo_check"]).area, axis=1)

    # Compute overlap ratio (intersection_area / detailed_building_area)
    intersections["overlap_ratio"] = intersections["area_intersection"] / intersections["geometry"].area
    # Keep only valid matches where the overlap is at least `min_overlap`
    valid_matches = intersections[intersections["overlap_ratio"] >= min_overlap]

    # Keep only the building in `buildings_gdf` that has the highest coverage for each detailed building
    best_matches = valid_matches.loc[valid_matches.groupby(valid_matches.index)["overlap_ratio"].idxmax()]
    best_matches = best_matches.set_index("ix")
    best_matches.index = best_matches.index.astype(int)
    
    # Compute min(base) and max(height) for the selected matches
    intersection_bases = best_matches.groupby(best_matches.index)["base_left"].min()
    intersection_height = best_matches.groupby(best_matches.index)["height_left"].max()

    # Apply updates efficiently
    buildings_gdf["base"] = buildings_gdf["base"].combine(intersection_bases, min)
    buildings_gdf["height"] = buildings_gdf["height"].combine(intersection_height, max)
    buildings_gdf = buildings_gdf.drop(["geo_check", "ix"], axis = 1)
    buildings_gdf.index = buildings_gdf.index.astype(int)
    
    # **Step 4: Handle Intersections in Reverse (Detailed buildings donate attributes to smaller ones)** ----------
    detailed_buildings_gdf['geo_check'] = detailed_buildings_gdf.geometry
    buildings_gdf['ix'] = buildings_gdf.index

    # Perform spatial join: find detailed buildings that intersect with non-detailed ones
    intersections = gpd.sjoin(buildings_gdf, detailed_buildings_gdf, predicate="intersects", how="left")
    intersections = intersections[intersections.geo_check.notnull()]

    # Compute intersection area
    intersections["area_intersection"] = intersections.apply(lambda row: row["geometry"].intersection(row["geo_check"]).area, axis=1)

    # Compute overlap ratio (intersection_area / non-detailed building area)
    intersections["overlap_ratio"] = intersections["area_intersection"] / intersections["geometry"].area

    # Keep only matches where the **smaller building's** overlap is at least threshold value
    valid_matches = intersections[intersections["overlap_ratio"] >= min_overlap]

    # For each **non-detailed building**, find the detailed one with the highest overlap
    best_matches = valid_matches.loc[valid_matches.groupby(valid_matches.index)["overlap_ratio"].idxmax()]

    # Compute min(base) and max(height) for the selected matches
    intersection_bases = best_matches.groupby(best_matches.index)["base_right"].min()
    intersection_height = best_matches.groupby(best_matches.index)["height_right"].max()

    # Apply updates efficiently: non-detailed buildings take values from detailed ones
    buildings_gdf["base"] = buildings_gdf["base"].combine(intersection_bases, min)
    buildings_gdf["height"] = buildings_gdf["height"].combine(intersection_height, max)
    buildings_gdf.index = buildings_gdf.index.astype(int)

    # # Replace 9999.0 in "base" with NaN, and -9999.0 in "height" with NaN
    buildings_gdf["base"] = buildings_gdf["base"].replace(9999.0, np.nan)
    buildings_gdf["height"] = buildings_gdf["height"].replace(-9999.0, np.nan)
    buildings_gdf = buildings_gdf.drop(["geo_check", "ix"], axis = 1, errors = 'ignore')
    
    return buildings_gdf

def select_buildings_by_study_area(larger_buildings_gdf, method='polygon', polygon=None, distance=1000):
    """
    Selects buildings from a GeoDataFrame that fall within a defined study area.

    Parameters
    ----------
    larger_buildings_gdf : GeoDataFrame
        GeoDataFrame containing building polygons to filter.
    method : {'polygon', 'distance'}, optional
        Method to define the study area:
        - 'polygon': use the provided `polygon` argument (default).
        - 'distance': use the centroid of all buildings, buffered by `distance`.
    polygon : shapely Polygon or MultiPolygon, optional
        Study area polygon. Required if method is 'polygon'.
    distance : float, optional
        Buffer distance (in CRS units) if method is 'distance'. Default is 1000.

    Returns
    -------
    GeoDataFrame
        GeoDataFrame of buildings within the study area. If no area is provided, returns empty GeoDataFrame.
    """
    # Validate input GeoDataFrame
    if larger_buildings_gdf.empty:
        return gpd.GeoDataFrame(columns=larger_buildings_gdf.columns)

    # Define study area
    if method == 'distance':
        study_area = larger_buildings_gdf.geometry.unary_union.centroid.buffer(distance)
    elif method == 'polygon':
        study_area = polygon
    else:
        raise ValueError("Method must be either 'polygon' or 'distance'.")

    # Filter buildings within the study area
    if study_area is not None:
        buildings_gdf = larger_buildings_gdf[larger_buildings_gdf.geometry.within(study_area)]
        return buildings_gdf
    else:
        return gpd.GeoDataFrame(columns=larger_buildings_gdf.columns)
        
def structural_score(buildings_gdf, obstructions_gdf, edges_gdf, advance_vis_expansion_distance = 300, neighbours_radius = 150):
    """
    The function computes the "Structural Landmark Component" sub-scores of each building.
    It considers:
    - distance from the street network:
    - advance 2d visibility polygon;
    - number of neighbouring buildings in a given radius.
         
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        Buildings GeoDataFrame - case study area.
    edges_gdf: LineString GeoDataFrame
        Street segmetns GeoDataFrame.
    obstructions_gdf: Polygon GeoDataFrame
        Obstructions GeoDataFrame.  
    advance_vis_expansion_distance: float
        2d advance visibility - it indicates up to which distance from the building boundaries the 2dvisibility polygon can expand.
    neighbours_radius: float
        Neighbours - research radius for other adjacent buildings.
        
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        The updated buildings GeoDataFrame.
    """  
    buildings_gdf = buildings_gdf.copy()
    
    # remove z coordinates if they are there already - issue with 2dvis
    if len(buildings_gdf.geometry.iloc[0].exterior.coords[0]) == 3: 
        buildings_gdf['geometry'] = buildings_gdf['geometry'].apply(lambda g: type(g)([(x, y) for x, y, *_ in g.exterior.coords]))
   
    obstructions_gdf = buildings_gdf if obstructions_gdf is None else obstructions_gdf
    sindex = obstructions_gdf.sindex
    street_network = edges_gdf.geometry.unary_union

    buildings_gdf["road"] = buildings_gdf.geometry.distance(street_network)
    buildings_gdf["2dvis"] = buildings_gdf.geometry.apply(lambda row: visibility_polygon2d(row, obstructions_gdf, sindex, max_expansion_distance=
                                    advance_vis_expansion_distance))
    buildings_gdf["neigh"] = buildings_gdf.geometry.apply(lambda row: _number_neighbours(row, obstructions_gdf, sindex, radius=neighbours_radius))

    return buildings_gdf
    
def _number_neighbours(geometry, obstructions_gdf, obstructions_sindex, radius):
    """
    The function counts the number of neighbours, in a GeoDataFrame, around a given geometry, within a
    research radius.
     
    Parameters
    ----------
    geometry: Shapely Geometry
        The geometry for which neighbors are counted.
    obstructions_gdf: GeoDataFrame
        The GeoDataFrame containing the obstructions.
    obstructions_sindex: Spatial Index
        The spatial index of the obstructions GeoDataFrame.
    radius: float
        The research radius for neighboring buildings.

    Returns
    -------
    int
        The number of neighbors.
    """
    buffer = geometry.buffer(radius)
    possible_neigh_index = list(obstructions_sindex.intersection(buffer.bounds))
    possible_neigh = obstructions_gdf.iloc[possible_neigh_index]
    precise_neigh = possible_neigh[possible_neigh.intersects(buffer)]
    return len(precise_neigh)
  
def visibility_score(buildings_gdf, sight_lines = pd.DataFrame({'a': []}), method = 'longest'):
    """
    The function calculates the sub-scores of the "Visibility Landmark Component".
    - 3d visibility;
    - facade area;
    - (height).
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        Buildings GeoDataFrame - case study area.
    sight_lines: LineString GeoDataFrame
        The Sight Lines GeoDataFrame.
        
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        The updated buildings GeoDataFrame.
    """  
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf["fac"] = 0.0
    if ("height" not in buildings_gdf.columns) | (sight_lines.empty): 
        return buildings_gdf, sight_lines

    sight_lines['nodeID'] = sight_lines['nodeID'].astype(int)
    sight_lines['buildingID'] = sight_lines['buildingID'].astype(int)

    buildings_gdf["fac"] = buildings_gdf.apply(lambda row: _facade_area(row["geometry"], row["height"]), axis = 1)

    stats = sight_lines.groupby('buildingID').agg({'length': ['mean','max', 'count']}) 
    stats.columns = stats.columns.droplevel(0)
    stats.rename(columns = {"count": "nr_lines"}, inplace = True)

    stats["max"].fillna((stats["max"].min()), inplace = True)
    stats["mean"].fillna((stats["mean"].min()), inplace = True)
    stats["nr_lines"].fillna((stats["nr_lines"].min()), inplace = True)
    stats.reset_index(inplace = True)
    columns = ["max", "mean", "nr_lines"]

    for column in columns:
        stats[column+"_sc"] = scaling_columnDF(stats[column])

    if method == 'longest':
        stats["3dvis"] = stats["max_sc"]
    elif method == 'combined':
        stats["3dvis"] = stats["max_sc"]*0.5+stats["mean_sc"]*0.25+stats["nr_lines_sc"]*0.25

    buildings_gdf = pd.merge(buildings_gdf, stats[["buildingID", "3dvis"]], on = "buildingID", how = "left") 
    buildings_gdf['3dvis'] = buildings_gdf['3dvis'].where(pd.notnull(buildings_gdf['3dvis']), 0.0)
    
    return buildings_gdf

def _facade_area(building_geometry, building_height):
    """
    Compute the approximate facade area of a building given its geometry and height.

    Parameters
    ----------
    building_geometry: Polygon
        The geometry of the building.
    building_height: float
        The height of the building.

    Returns
    -------
    float
        The computed approximate facade area of the building.
    """    
    envelope = building_geometry.envelope
    coords = mapping(envelope)["coordinates"][0]
    d = [(Point(coords[0])).distance(Point(coords[1])), (Point(coords[1])).distance(Point(coords[2]))]
    width = min(d)
    return width*building_height
 
def get_historic_buildings_fromOSM(OSMplace, download_method, crs = None, distance = 1000):
    """    
    The function downloads and cleans building footprint geometries and create a buildings GeoDataFrames for the area of interest.
    However, it only keeps the buildings that are considered historic buildings or heritage buildings in OSM. 
            
    Parameters
    ----------
    OSMplace: str, tuple, Shapely Polygon
        Name of cities or areas in OSM: 
        - when using "distance_from_point" please provide a (lat, lon) tuple to create the bounding box around it; 
        - when using "distance_from_address" provide an existing OSM address; 
        - when using "OSMplace" provide an OSM place name. The query must be geocodable and OSM must have polygon boundaries for the geocode result.  
        - when using "polygon" please provide a Shapely Polygon in unprojected latitude-longitude degrees (EPSG:4326) CRS;
    download_method: str, {"distance_from_address", "distance_from_point", "OSMplace", "polygon"}
        It indicates the method that should be used for downloading the data.
    crs : str, or pyproj.CRS
        Coordinate Reference System for the study area. Can be a string (e.g. 'EPSG:32633'), or a pyproj.CRS object.
    distance: float
        Used when download_method == "distance from address" or == "distance from point".
    
    Returns
    -------
    historic_buildings: Polygon GeoDataFrame
        The historic buildings GeoDataFrame.
    """   
    
    columns = ['geometry', 'historic']
    tags = {"building": True}
    historic_buildings = downloader(place = OSMplace, download_method = download_method, tags = tags, distance = distance)
    
    if 'heritage' in historic_buildings:
        columns.append('heritage')
    historic_buildings = historic_buildings[columns]

    if 'heritage' in historic_buildings:
        historic_buildings = historic_buildings[~(historic_buildings.historic.isnull() & historic_buildings.heritage.isnull())]
    else:
        historic_buildings = historic_buildings[~historic_buildings.historic.isnull()]
    
    if crs is None:
        historic_buildings = ox.projection.project_gdf(historic_buildings)
    else:
        historic_buildings = historic_buildings.to_crs(crs)

    historic_buildings.loc[historic_buildings["historic"] != 0, "historic"] = 1
    historic_buildings = historic_buildings[['geometry', 'historic']]
    historic_buildings['area'] = historic_buildings.geometry.area
       
    return historic_buildings
 
def cultural_score(buildings_gdf, historic_elements_gdf = pd.DataFrame({'a': []}), score_column = None, from_OSM = False):
    """
    The function computes the "Cultural Landmark Component" based on the number of features listed in historic/cultural landmarks datasets. It can be
    obtained either on the basis of a score given by the data-provider or on the number of features intersecting the buildings object 
    of analysis.
    
    "score_column" indicates the attribute field containing scores assigned to historic buildings, if existing.
    Alternatively, if the column has been already assigned to the buildings_gdf, one can use OSM historic categorisation (binbary).
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        Buildings GeoDataFrame - case study area.
    historic_elements_gdf: Point or Polygon GeoDataFrame
        The GeoDataFrame containing information about listed historic buildings or elements.
    score_column: str
        The name of the column in the historic_elements_gdf that provides information on the classification of the historic listings; this could be issued by governamental agencies, for example.
    from_OSM: boolean
        If using the historic field from OSM. NOTE: This column should be already in the buildings_gdf columns.
   
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        The updated buildings GeoDataFrame.
    """  
    
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf["cult"] = 0.0
    
    def _cultural_score_building(building_geometry, historic_elements_gdf, historic_elements_gdf_sindex, score_column = None):

        possible_matches_index = list(historic_elements_gdf_sindex.intersection(building_geometry.bounds)) # looking for possible candidates in the external GDF
        possible_matches = historic_elements_gdf.iloc[possible_matches_index]
        matches = possible_matches[possible_matches.intersects(building_geometry)]

        if (score_column is None): 
            cultural_score = len(matches) # score only based on number of intersecting elements
        elif len(matches) == 0: 
            cultural_score = 0
        else: 
            cultural_score = matches[score_column].sum() # otherwise sum the scores of the intersecting elements
        return cultural_score
    
    # using OSM binary value - this means that the buildings dataset if from OSM
    if (from_OSM) & ("historic" in buildings_gdf.columns):
        # Set 'historic' column to 0 where it is currently null
        buildings_gdf.loc[buildings_gdf["historic"].isnull(), "historic"] = 0
        # Set 'historic' column to 1 where it is not 0
        buildings_gdf.loc[buildings_gdf["historic"] != 0, "historic"] = 1
        buildings_gdf["cult"] = buildings_gdf["historic"]
        return buildings_gdf
    
    # spatial index
    sindex = historic_elements_gdf.sindex 
    buildings_gdf["cult"] = buildings_gdf.geometry.apply(lambda row: _cultural_score_building(row, historic_elements_gdf, sindex, score_column = score_column))
    return buildings_gdf
    
def pragmatic_score(buildings_gdf, research_radius = 200):
    """
    The function computes the "Pragmatic Landmark Component" based on the frequency, and therefore unexpectedness, of a land_use class in an area around a building.
    The area is defined by the parameter "research_radius".
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        Buildings GeoDataFrame - case study area.
    research_radius: float
        The radius to be used around the given building to identify neighbours.
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        The updated buildings GeoDataFrame.
    """  
    
    buildings_gdf = buildings_gdf.copy()   
    buildings_gdf["nr"] = 1 # to count
    sindex = buildings_gdf.sindex # spatial index
    
    if 'land_use' not in buildings_gdf.columns:
        buildings_gdf['land_use'] = buildings_gdf['land_use_raw']
        
    def _pragmatic_meaning_building(building_geometry, building_land_use, buildings_gdf, buildings_gdf_sindex, radius):

        buffer = building_geometry.buffer(radius)
        possible_matches_index = list(buildings_gdf_sindex.intersection(buffer.bounds))
        possible_matches = buildings_gdf.iloc[possible_matches_index]
        matches = possible_matches[possible_matches.intersects(buffer)]
        neigh = matches.groupby(["land_use"], as_index = True)["nr"].sum() 

        Nj = neigh.loc[building_land_use] # nr of neighbours with same land_use
        Pj = 1-(Nj/matches["nr"].sum()) # inverting the value # Pj = Nj/N
        return Pj    
        
    buildings_gdf["prag"] = buildings_gdf.apply(lambda row: _pragmatic_meaning_building(row.geometry, row.land_use, buildings_gdf, 
                                        sindex, radius = research_radius), axis = 1)
    buildings_gdf.drop('nr', axis = 1, inplace = True)
    return buildings_gdf
    
def compute_global_scores(buildings_gdf, global_indexes_weights, global_components_weights):
    """
    Computes the component and global scores for a buildings GeoDataFrame, rescaling values when necessary and assigning weights to the different properties measured.

    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        The input GeoDataFrame containing buildings information.
    global_indexes_weights: dict
        Dictionary with index names (string) as keys and weights as values.
    global_components_weights: dict
        Dictionary with component names (string) as keys and weights as values.

    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        The updated buildings GeoDataFrame with computed scores.
    
    Examples
    --------
    # Example 1: Computing global scores for a buildings GeoDataFrame
    >>> buildings_gdf = gpd.GeoDataFrame(...)
    >>> global_indexes_weights = {"3dvis": 0.50, "fac": 0.30, "height": 0.20, "area": 0.30, "2dvis": 0.30, "neigh": 0.20, "road": 0.20}
    >>> global_components_weights = {"vScore": 0.50, "sScore": 0.30, "cScore": 0.20, "pScore": 0.10}
    """  
    # scaling
    cols = {"direct": ["3dvis", "fac", "height", "area", "2dvis", "cult", "prag"], "inverse": ["neigh", "road"]}
   
    if not (abs(sum(global_components_weights.values()) - 1.0) < 1e-6):
        raise ValueError("Global components weights must sum to 1.0")
   
    # Check if vScore should be computed
    compute_vScore = (
        "vScore" in global_components_weights and
        "height" in buildings_gdf.columns and
        buildings_gdf["height"].max() > 0.0
    )
    
    # Rescale values if the column exists in buildings_gdf
    for col in cols["direct"] + cols["inverse"]:
        if col in buildings_gdf.columns:
            buildings_gdf[col + "_sc"] = scaling_columnDF(buildings_gdf[col], inverse=(col in cols["inverse"]))

    # Compute component scores
    # Visual Score
    if compute_vScore:
        buildings_gdf["vScore"] = sum(
            buildings_gdf[f"{col}_sc"] * global_indexes_weights[col]
            for col in ["fac", "height", "3dvis"] if f"{col}_sc" in buildings_gdf
        )
        buildings_gdf["vScore_sc"] = scaling_columnDF(buildings_gdf["vScore"])
        
    # Structural Score

    buildings_gdf["sScore"] = sum(
        buildings_gdf[f"{col}_sc"] * global_indexes_weights[col]
        for col in ["area", "neigh", "2dvis", "road"] if f"{col}_sc" in buildings_gdf
    )
    buildings_gdf["sScore_sc"] = scaling_columnDF(buildings_gdf["sScore"])
    
    buildings_gdf["cScore"], buildings_gdf["pScore"] = buildings_gdf["cult_sc"], buildings_gdf["prag_sc"]
    
    # Cultural and Pragmatic Scores
    if "cult_sc" in buildings_gdf.columns:
        buildings_gdf["cScore"] = buildings_gdf["cult_sc"]
    if "prag_sc" in buildings_gdf.columns:
        buildings_gdf["pScore"] = buildings_gdf["prag_sc"]
    
    # Final global score: Compute dynamically
    buildings_gdf["gScore"] = sum(
        buildings_gdf[f"{component}_sc"] * global_components_weights[component]
        for component in global_components_weights
        if f"{component}_sc" in buildings_gdf and (component != "vScore" or compute_vScore)
    )
    
    buildings_gdf["gScore_sc"] = scaling_columnDF(buildings_gdf["gScore"])
    return buildings_gdf

def compute_local_scores(buildings_gdf, local_indexes_weights, local_components_weights, rescaling_radius = 1500):
    """
    The function computes landmarkness at the local level. The components' weights may be different from the ones used to calculate the
    global score. The radius parameter indicates the extent of the area considered to rescale the landmarkness local score.
    - local_indexes_weights: keys are index names (string), items are weights.
    - local_components_weights: keys are component names (string), items are weights.
    
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        The input GeoDataFrame containing buildings information.
    local_indexes_weights: dict
        Dictionary with index names (string) as keys and weights as values.
    local_components_weights: dict
        Dictionary with component names (string) as keys and weights as values.
   
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        The updated buildings GeoDataFrame.
    
    Examples
    --------
    >>> # local landmarkness indexes weights, cScore and pScore have only 1 index each
    >>> local_indexes_weights = {"3dvis": 0.50, "fac": 0.30, "height": 0.20, "area": 0.40, "2dvis": 0.00, "neigh": 0.30 , "road": 0.30}
    >>> # local landmarkness components weights
    >>> local_components_weights = {"vScore": 0.25, "sScore": 0.35, "cScore":0.10 , "pScore": 0.30} 
    """  
    
    buildings_gdf = buildings_gdf.set_index("buildingID").copy()
    buildings_gdf.index.name = None
    sindex = buildings_gdf.sindex # spatial index
    
    # Validate that local_components_weights sum to 1.0
    if not (abs(sum(local_components_weights.values()) - 1.0) < 1e-6):
        raise ValueError("Local components weights must sum to 1.0")

    # Initialize scores conditionally
    compute_vScore = (
        "vScore" in local_components_weights and
        "height" in buildings_gdf.columns and
        buildings_gdf["height"].max() > 0.0
    )
    if compute_vScore:
        buildings_gdf["vScore_l"] = 0.0  # Initialize only if valid height data exists
   
    buildings_gdf["sScore_l"] = 0.0
    buildings_gdf["lScore"] = 0.0
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_scores = {
            executor.submit(
                _building_local_score,
                row["geometry"],
                idx,
                buildings_gdf,
                sindex,
                local_components_weights,
                local_indexes_weights,
                rescaling_radius,
            ): idx
            for idx, row in buildings_gdf.iterrows()
        }
        for future in concurrent.futures.as_completed(future_scores):
            buildingID = future_scores[future]
            try:
                buildings_gdf.loc[buildingID, "lScore"] = future.result()
            except Exception:
                buildings_gdf.loc[buildingID, "lScore"] = 0.0
    
    buildings_gdf["lScore_sc"] = scaling_columnDF(buildings_gdf["lScore"])
    return buildings_gdf
    
def _building_local_score(building_geometry, buildingID, buildings_gdf, buildings_gdf_sindex, local_components_weights, local_indexes_weights, radius):
    """
    The function computes landmarkness at the local level for a single building. 
    
    Parameters
    ----------
    building_geometry  Polygon
        The geometry of the building.
    buildingID: int
        The ID of the building.
    buildings_gdf: Polygon GeoDataFrame
        The GeoDataFrame containing the buildings.
    buildings_gdf_sindex: Spatial Index
        The spatial index of the buildings GeoDataFrame.
    local_components_weights: dictionary
        The weights assigned to local-level components.
    local_indexes_weights: dictionary
        The weights assigned to local-level indexes.
    radius: float
        The radius that regulates the area around the building within which the scores are recomputed.

    Returns
    -------
    score : float
        The computed local-level landmarkness score for the building.
    """
                                             
    cols = {"direct": ["3dvis", "fac", "height", "area", "2dvis", "cult", "prag"], "inverse": ["neigh", "road"]}
    
    buffer = building_geometry.buffer(radius)
    possible_matches_index = list(buildings_gdf_sindex.intersection(buffer.bounds))
    matches = buildings_gdf.iloc[list(sindex.intersection(buffer.bounds))].copy()
    matches = matches[matches.intersects(buffer)]
                
    # Rescale all values dynamically if the column exists in matches
    for column in cols["direct"] + cols["inverse"]:
        if column in matches.columns:
            matches[column + "_sc"] = scaling_columnDF(matches[column], inverse=(column in cols["inverse"]))
  
    # Compute structural score (sScore)
    if "sScore" in local_components_weights:
        matches["sScore_l"] = sum(
            matches[f"{col}_sc"] * local_indexes_weights[col] for col in ["area", "2dvis", "neigh", "road"] if f"{col}_sc" in matches
        )
  
    # Recomputing visual scores only if "height" is valid
    # Determine if vScore should be computed
    compute_vScore = (
        "vScore" in local_components_weights and
        "height" in matches.columns and
        matches["height"].max() > 0.0
    )
    
    if compute_vScore:
        matches["vScore_l"] = sum(
            matches[f"{col}_sc"] * local_indexes_weights[col] for col in ["fac", "height", "3dvis"] if f"{col}_sc" in matches
        )
  
    # Compute cultural and pragmatic scores if defined
    if "cScore" in local_components_weights:
        matches["cScore_l"] = matches["cult_sc"]
    if "pScore" in local_components_weights:
        matches["pScore_l"] = matches["prag_sc"]
    
    # Rescale component scores dynamically
    for component in local_components_weights.keys():
        if f"{component}_l" in matches and (component != "vScore" or compute_vScore):
            matches[f"{component}_l_sc"] = scaling_columnDF(matches[f"{component}_l"])

    # Compute the final local score
    matches["lScore"] = sum(
        matches[f"{component}_l_sc"] * local_components_weights[component]
        for component in local_components_weights
        if f"{component}_l_sc" in matches and (component != "vScore" or compute_vScore)
    )

    # Return the local score for the specified building
    return round(matches.loc[buildingID, "lScore"], 3)