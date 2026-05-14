import osmnx as ox
import pandas as pd
import numpy as np
import geopandas as gpd
import pyvista as pv

from shapely.ops import unary_union
pd.set_option("display.precision", 3)

from .utilities import downloader, gdf_multipolygon_to_polygon
from .land_use_derive import derive_land_uses_raw_fromOSM

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
    obstructions_gdf = obstructions_gdf[obstructions_gdf["area"] >= min_area]
    if "height" in obstructions_gdf.columns:
        mean_height = obstructions_gdf["height"].mean()
    if mean_height > 5:
        obstructions_gdf = obstructions_gdf[obstructions_gdf["height"] >= min_height]

    obstructions_gdf = obstructions_gdf[["height", "base","geometry", "area", "land_use_raw"]]
    # assigning ID
    obstructions_gdf["buildingID"] = obstructions_gdf.index.values.astype(int)
    obstructions_gdf = gdf_multipolygon_to_polygon(obstructions_gdf, columnID="buildingID")
    
    # if case-study area and distance not defined
    if (case_study_area is None) and (distance_from_center is None or distance_from_center == 0):
        buildings_gdf = obstructions_gdf.copy()
        return buildings_gdf, obstructions_gdf
    if (case_study_area is None):     # define a case study area
        case_study_area = obstructions_gdf.geometry.union_all().centroid.buffer(distance_from_center)
    buildings_gdf = obstructions_gdf[obstructions_gdf.geometry.within(case_study_area)]

    return buildings_gdf, obstructions_gdf
 
def get_buildings_fromOSM(OSMplace, download_method: str, crs=None, distance=1000, min_area=200):
    """    
    Download and clean building footprints from OSM and return a buildings GeoDataFrame
    for the area of interest. Uses OSMnx for download/projection and derive_land_use_raw
    to build a land_use_raw list per building.
            
    Parameters
    ----------
    OSMplace : str, tuple, Shapely Polygon
        Name of cities or areas in OSM: 
        - when using "distance_from_point" provide a (lat, lon) tuple;
        - when using "distance_from_address" provide an existing OSM address;
        - when using "OSMplace" provide an OSM place name with polygon boundaries;
        - when using "polygon" provide a Shapely Polygon in EPSG:4326.
    download_method : str, {"distance_from_address", "distance_from_point", "OSMplace", "polygon"}
        Method used to download the data.
    crs : str or pyproj.CRS, optional
        Target CRS for the study area (e.g. 'EPSG:32633'). If None, use OSMnx projection.
    distance : float, optional
        Used when download_method == "distance_from_address" or "distance_from_point".
    min_area : float, optional
        Minimum area threshold (in CRS units, e.g. square meters) for a building
        to be included. Default is 200.
    
    Returns
    -------
    buildings_gdf : GeoDataFrame
        Polygon GeoDataFrame containing building footprints with:
        - geometry
        - historic (if present, else NA)
        - land_use_raw (list of land-use descriptors)
        - buildingID (int)
    """   

    # 1) Download only building features
    tags = {"building": True}
    buildings_gdf = downloader(
        OSMplace=OSMplace,
        download_method=download_method,
        tags=tags,
        distance=distance
    )

    # 2) CRS handling
    if crs is None:
        buildings_gdf = ox.projection.project_gdf(buildings_gdf)
    else:
        buildings_gdf = buildings_gdf.to_crs(crs)

    # 3) Remove empty geometries
    buildings_gdf = buildings_gdf[~buildings_gdf["geometry"].is_empty]

    # 4) Derive land_use_raw as list using building:use:*, amenity, shop, etc.
    buildings_gdf = derive_land_uses_raw_fromOSM(
        buildings_gdf,
        default="residential"
    )
    
    # 7) Keep only final columns for output
    buildings_gdf = buildings_gdf[["geometry", "land_uses_raw"]]

    # 8) Explode multipolygons and assign buildingID
    buildings_gdf = gdf_multipolygon_to_polygon(buildings_gdf)
    buildings_gdf["area"] = buildings_gdf.geometry.area
    buildings_gdf = buildings_gdf[buildings_gdf["area"] >= min_area]
    buildings_gdf = buildings_gdf.reset_index(drop=True)
    buildings_gdf["buildingID"] = buildings_gdf.index.values.astype(int)
    
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