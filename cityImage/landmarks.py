import osmnx as ox
import pandas as pd
import numpy as np
import geopandas as gpd

from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping, MultiLineString
from shapely.ops import cascaded_union, linemerge
from scipy.sparse import linalg
pd.set_option("precision", 10)

from .utilities import *
from .angles import *

"""
This set of functions is designed for extracting the computational Image of The City.
Computational landmarks can be extracted employing the following functions.

"""
  
 
def get_buildings_fromSHP(path, epsg, case_study_area = None, distance_from_center = 1000, height_field = None, base_field = None, land_use_field = None):

    """    
    The function take a sets of buildings, returns two smaller GDFs of buildings: the case-study area, plus a larger area containing other 
    buildings, called "obstructions" (for analyses which include adjacent buildings). If the area for clipping the obstructions is not
    provided a buffer from the case-study is used to build the obstructions GDF.
            
    Parameters
    ----------
    path: string
        path where the file is stored
    epsg: int
        epsg of the area considered; if None OSMNx is used for the projection
    case_study_area: Polygon
        The Polygon to use for clipping and identifying the case-study area, within the input .shp. If not available, use "distance_from_center"
    height_field, base_field: str 
        height and base fields name in the original data-source
    distance_from_center: float
        so to identify the case-study area on the basis of distance from the center of the input .shp
    
    Returns
    -------
    buildings_gdf, obstructions_gdf: tuple of GeoDataFrames
        the buildings and the obstructions GeoDataFrames
    """   
    
    # trying reading buildings footprints shapefile from directory
    obstructions_gdf = gpd.read_file(path).to_crs(epsg=epsg)  
    
    # computing area, reassigning columns
    obstructions_gdf["area"] = obstructions_gdf["geometry"].area
    if height_field is not None: 
        obstructions_gdf["height"] = obstructions_gdf[height_field]
    if base_field is None: 
        obstructions_gdf["base"] = 0.0
        if height_field is not None: 
            obstructions_gdf["height_r"] = obstructions_gdf["height"]
    else: 
        obstructions_gdf["base"] = obstructions_gdf[base_field]
        if height_field is not None: 
            obstructions_gdf["height_r"] = obstructions_gdf["height"]+obstructions_gdf["base"] # relative_height
    if land_use_field is not None: 
        obstructions_gdf["land_use_raw"] = obstructions_gdf[land_use_field]
    else:
        obstructions_gdf["land_use_raw"] = None
        
    # dropping small buildings and buildings with null height
    obstructions_gdf = obstructions_gdf[obstructions_gdf["area"] >= 50]
    if height_field is not None: 
        obstructions_gdf = obstructions_gdf[obstructions_gdf["height"] >= 1]
    obstructions_gdf = obstructions_gdf[["height", "height_r", "base","geometry", "area", "land_use_raw"]]
    
    # assigning ID
    obstructions_gdf["buildingID"] = obstructions_gdf.index.values.astype(int)
    
    # if case-study area and distance not defined
    if (case_study_area is None) & (distance_from_center is None):
        buildings_gdf = obstructions_gdf.copy()
        return buildings_gdf, obstructions_gdf

    # if case-study area is not defined
    if (case_study_area is None): 
        case_study_area = obstructions_gdf.geometry.unary_union.centroid.buffer(distance_from_center)

    buildings_gdf = obstructions_gdf[obstructions_gdf.geometry.within(case_study_area)]
    # clipping buildings in the case-study area

    return buildings_gdf, obstructions_gdf
    
def get_buildings_fromOSM(place, download_method, epsg = None, distance = 1000):
    """    
    The function downloads and cleans buildings footprint geometries and create a buildings GeoDataFrames for the area of interest.
    The function exploits OSMNx functions for downloading the data as well as for projecting it.
    The land use classification for each building is extracted. Only relevant columns are kept.   
            
    Parameters
    ----------
    place: string, tuple
        name of cities or areas in OSM: when using "from point" please provide a (lat, lon) tuple to create the bounding box around it; when using "distance_from_address"
        provide an existing OSM address; when using "OSMplace" provide an OSM place name
    download_method: string, {"from_point", "distance_from_address", "OSMplace"}
        it indicates the method that should be used for downloading the data.
    epsg: int
        epsg of the area considered; if None OSMNx is used for the projection
    distance: float
        Specify distance from address or point
    
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        the buildings GeoDataFrame
    """   
    
    columns_to_keep = ['amenity', 'building', 'geometry', 'historic', 'land_use_raw']

    if download_method == "distance_from_address": 
        buildings_gdf = ox.geometries_from_address(address = place, distance = distance, tags={"building": True})
    elif download_method == "OSMplace": 
        buildings_gdf = ox.geometries_from_place(place, tags={"building": True})
    elif download_method == "from_point": 
        buildings_gdf = ox.geometries_from_point(point = place, distance = distance, tags={"building": True})
    elif download_method == "OSMpolygon": 
        buildings_gdf = ox.geometries_from_polygon(place, tags={"building": True})
    else: raise downloadError('Provide a download method amongst {"from_point", "distance_from_address", "OSMplace", "OSMpolygon}')
    
    if epsg is None:
        buildings_gdf = ox.projection.project_gdf(buildings_gdf)
    else:
        crs = {'init': 'epsg:'+str(epsg), 'no_defs': True}
        buildings_gdf = buildings_gdf.to_crs(crs)

    buildings_gdf['land_use_raw'] = None
    for column in buildings_gdf.columns: 
        if column.startswith('building:use:'): 
            buildings_gdf.loc[pd.notnull(buildings_gdf[column]), 'land_use_raw'] = column[13:]
        if column not in columns_to_keep: 
            buildings_gdf.drop(column, axis = 1, inplace = True)

    buildings_gdf = buildings_gdf[~buildings_gdf['geometry'].is_empty]
    buildings_gdf['building'].replace('yes', np.nan, inplace = True)
    buildings_gdf['building'][buildings_gdf['building'].isnull()] = buildings_gdf['amenity']
    buildings_gdf['land_use_raw'][buildings_gdf['land_use_raw'].isnull()] = buildings_gdf['building']
    buildings_gdf['land_use_raw'][buildings_gdf['land_use_raw'].isnull()] = 'residential'

    buildings_gdf = buildings_gdf[['geometry', 'historic', 'land_use_raw']]
    buildings_gdf['area'] = buildings_gdf.geometry.area
    buildings_gdf = buildings_gdf[buildings_gdf['area'] >= 50] 
    
    # reset index
    buildings_gdf = buildings_gdf.reset_index(drop = True)
    buildings_gdf['buildingID'] = buildings_gdf.index.values.astype('int')  
    
    return buildings_gdf

def simplify_footprints(buildings_gdf, crs):
    """    
    The function downloads and cleans buildings footprint geometries and create a buildings GeoDataFrames for the area of interest.
    The function exploits OSMNx functions for downloading the data as well as for projecting it.
    The land use classification for each building is extracted. Only relevant columns are kept.   
            
    Parameters
    ----------
    place: string, tuple
        name of cities or areas in OSM: when using "from point" please provide a (lat, lon) tuple to create the bounding box around it; when using "distance_from_address"
        provide an existing OSM address; when using "OSMplace" provide an OSM place name
    download_method: string, {"from_point", "distance_from_address", "OSMplace"}
        it indicates the method that should be used for downloading the data.
    epsg: int
        epsg of the area considered; if None OSMNx is used for the projection
    distance: float
        Specify distance from address or point
    
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        the buildings GeoDataFrame
    """  
    
    buildings_gdf = buildings_gdf.copy()
    single_parts = gpd.geoseries.GeoSeries([geom for geom in buildings_gdf.unary_union.geoms])
    single_parts_gdf = gpd.GeoDataFrame(geometry=single_parts, crs = crs)
    
    return single_parts_gdf

def attach_attributes(buildings_gdf, attributes_gdf, height_field, base_field, land_use_field):
    """    
    The function downloads and cleans buildings footprint geometries and create a buildings GeoDataFrames for the area of interest.
    The function exploits OSMNx functions for downloading the data as well as for projecting it.
    The land use classification for each building is extracted. Only relevant columns are kept.   
            
    Parameters
    ----------
    place: string, tuple
        name of cities or areas in OSM: when using "from point" please provide a (lat, lon) tuple to create the bounding box around it; when using "distance_from_address"
        provide an existing OSM address; when using "OSMplace" provide an OSM place name
    download_method: string, {"from_point", "distance_from_address", "OSMplace"}
        it indicates the method that should be used for downloading the data.
    epsg: int
        epsg of the area considered; if None OSMNx is used for the projection
    distance: float
        Specify distance from address or point
    
    Returns
    -------
    new_buildings_gdf: Polygon GeoDataFrame
        the buildings GeoDataFrame
    """  
    
    buildings_gdf = buildings_gdf.copy()
    attributes_gdf['area'] = attributes_gdf.geometry.area
    attributes_gdf = attributes_gdf[attributes_gdf.area > 50].copy()
    attributes_gdf[land_use_field] = attributes_gdf[land_use_field].where(pd.notnull(attributes_gdf[land_use_field]), None)
    if land_use_field in buildings_gdf:
        buildings_gdf.drop(land_use_field, axis = 1, inplace = True)
    buildings_gdf = gpd.sjoin(buildings_gdf, attributes_gdf[['area', 'geometry', height_field, land_use_field]], how="left", op= 'intersects')
    
    buildings_gdf['land_use_raw'] = None
    new_buildings_gdf = buildings_gdf.copy()
    new_buildings_gdf = new_buildings_gdf[0:0]
    buildings_gdf.reset_index(inplace = True, drop = True)

    new_index = 0
    builgindIDs = buildings_gdf.buildingID.unique()
    
    for bID in builgindIDs:
        gdf_tmp = buildings_gdf[buildings_gdf.buildingID == bID].copy()
        new_buildings_gdf.loc[new_index] = gdf_tmp.iloc[0]
        
        if len(gdf_tmp) == 1:
            new_buildings_gdf.at[new_index,'land_use_raw'] = gdf_tmp.iloc[0][land_use_field]
        else:
            index_lu = gdf_tmp['area_right'].idxmax()
            land_use = gdf_tmp.loc[index_lu][land_use_field]
            if land_use is None:
                pass
            else:
                new_buildings_gdf.at[new_index,'land_use_raw'] = land_use
       
        new_buildings_gdf.at[new_index, 'height'] = gdf_tmp[height_field].max()
        new_buildings_gdf.at[new_index, 'base'] = gdf_tmp[base_field].max()
        new_index += 1
        
    new_buildings_gdf['area'] = new_buildings_gdf.geometry.area
    new_buildings_gdf.drop([land_use_field, 'area_left', 'area_right', 'index_right'], axis = 1, inplace = True)
    
    return new_buildings_gdf

       
def structural_score(buildings_gdf, obstructions_gdf, edges_gdf, max_expansion_distance = 300, distance_along = 50, radius = 150):
    """
    The function computes the structural properties of each building properties.
    
    neighbours
    "radius" 

     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        buildings GeoDataFrame - case study area
    edges_gdf: LineString GeoDataFrame
        street segmetns GeoDataFrame
    obstructions_gdf: Polygon GeoDataFrame
        obstructions GeoDataFrame  
    max_expansion_distance: float
        2d advance visibility - it indicates up to which distance from the building boundaries the 2dvisibility polygon can expand.
    distance_along: float
        2d advance visibility - it defines the interval between each line's destination, namely the search angle.
    radius: float
        neighbours - research radius for other adjacent buildings.
        
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        the updated buildings GeoDataFrame
    """  
    
    buildings_gdf = buildings_gdf.copy()
    if (obstructions_gdf is None): 
        obstructions_gdf = buildings_gdf.copy()
    # spatial index
    sindex = obstructions_gdf.sindex
    street_network = edges_gdf.geometry.unary_union

    # distance from road
    buildings_gdf["road"] =  buildings_gdf.apply(lambda row: row["geometry"].distance(street_network), axis = 1)
    # 2d advance visibility
    buildings_gdf["2dvis"] = buildings_gdf.apply(lambda row: _advance_visibility(row["geometry"], obstructions_gdf, sindex, max_expansion_distance = max_expansion_distance,
                                distance_along = distance_along), axis = 1)
    # neighbours
    buildings_gdf["neigh"] = buildings_gdf.apply(lambda row: _number_neighbours(row["geometry"], obstructions_gdf, sindex, radius = radius), axis = 1)
    
    return buildings_gdf
    
def _number_neighbours(building_geometry, obstructions_gdf, obstructions_sindex, radius):
    """
    The function computes a buildings" number of neighbours.
    "radius" research radius for other adjacent buildings.
     
    Parameters
    ----------
    building_geometry: Polygon
    obstructions_gdf: Polygon GeoDataFrame
        obstructions GeoDataFrame  
    obstructions_sindex: Rtree Spatial Index
    radius: float
        research radius for other adjacent buildings.
    Returns
    -------
    int
    """
        
    buffer = building_geometry.buffer(radius)
    possible_neigh_index = list(obstructions_sindex.intersection(buffer.bounds))
    possible_neigh = obstructions_gdf.iloc[possible_neigh_index]
    precise_neigh = possible_neigh[possible_neigh.intersects(buffer)]
    return len(precise_neigh)

def _advance_visibility(building_geometry, obstructions_gdf, obstructions_sindex, max_expansion_distance = 600, distance_along = 20):

    """
    It creates a 2d polygon of visibility around a building. The extent of this polygon is assigned as a 2d advance
    visibility measure. The polygon is built constructing lines around the centroid, breaking them at obstructions and connecting 
    the new formed geometries to get the final polygon.
    "max_expansion_distance" indicates up to which distance from the building boundaries the visibility polygon can expand.
    "distance_along" defines the interval between each line's destination, namely the search angle.
     
    Parameters
    ----------
    building_geometry: Polygon
    obstructions_gdf: Polygon GeoDataFrame
        obstructions GeoDataFrame  
    obstructions_sindex: Rtree Spatial Index
    max_expansion_distance: float
        it indicates up to which distance from the building boundaries the 2dvisibility polygon can expand.
    distance_along: float
        it defines the interval between each line's destination, namely the search angle.

    Returns
    -------
    float
    """
      
    # creating buffer
    origin = building_geometry.centroid
    if building_geometry.geom_type == 'MultiPolygon':
        building_geometry = building_geometry.convex_hull
    exteriors = list(building_geometry.exterior.coords)
    no_holes = Polygon(exteriors)
    max_expansion_distance = max_expansion_distance + origin.distance(building_geometry.envelope.exterior)
    
    # identifying obstructions in an area of x (max_expansion_distance) mt around the building
    possible_obstacles_index = list(obstructions_sindex.intersection(origin.buffer(max_expansion_distance).bounds))
    possible_obstacles = obstructions_gdf.iloc[possible_obstacles_index]
    possible_obstacles = obstructions_gdf[obstructions_gdf.geometry != building_geometry]
    possible_obstacles = obstructions_gdf[~obstructions_gdf.geometry.within(no_holes)]

    start = 0.0
    i = start
    list_lines = [] # list of lines
    
    # creating lines all around the building till a defined distance
    while(i <= 360):
        coords = get_coord_angle([origin.x, origin.y], distance = max_expansion_distance, angle = i)
        line = LineString([origin, Point(coords)])
        
        # finding actual obstacles to this line
        obstacles = possible_obstacles[possible_obstacles.crosses(line)]
        ob = cascaded_union(obstacles.geometry)
        
        """
        if there are obstacles: indentify where the line from the origin is interrupted, create the geometry and
        append it to the list of lines
        """
        
        if len(obstacles) > 0:
            t = line.intersection(ob)
            # taking the coordinates
            try: 
                intersection = t[0].coords[0]
            except: 
                intersection = t.coords[0]
            lineNew = LineString([origin, Point(intersection)])
        
        # the line is not interrupted, keeping the original one
        else: lineNew = line 

        list_lines.append(lineNew)
        # increase the angle
        i = i+distance_along
   
    # creating a polygon of visibility based on the lines and their progression, taking into account the origin Point too    
    list_points = [Point(origin)]
    for i in list_lines: 
        list_points.append(Point(i.coords[1]))
    list_points.append(Point(origin))
    poly = Polygon([[p.x, p.y] for p in list_points])
    
    # subtracting th area of the building and computing the area of the polygon (area of visibility)
    try: 
        poly_vis = poly.difference(building_geometry)
    except:
        pp = poly.buffer(0)
        poly_vis = pp.difference(building_geometry)      
    
    return poly_vis.area
    
def visibility_score(buildings_gdf, sight_lines = pd.DataFrame({'a' : []}), method = 'longest'):

    """
    The function calculates a 3d visibility score making use of precomputed 3d sight lines.
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        buildings GeoDataFrame - case study area
    sight_lines: LineString GeoDataFrame
    
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        the updated buildings GeoDataFrame
    """  
    sight_lines = sight_lines.copy()
    sight_lines['nodeID'] = sight_lines['nodeID'].astype(int)
    sight_lines['buildingID'] = sight_lines['buildingID'].astype(int)
    
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf["fac"] = 0.0
    if "height" not in buildings_gdf.columns: 
        return buildings_gdf
        
    #facade area (roughly computed)
    buildings_gdf["fac"] = buildings_gdf.apply(lambda row: _facade_area(row["geometry"], row["height"]), axis = 1)
    if sight_lines.empty: 
        return buildings_gdf
    
    # 3d visibility
    sight_lines = sight_lines.copy()
    sight_lines.drop(["Shape_Leng", "DIST_ALONG", "visible", "Visibility"], axis = 1, inplace = True, errors = "ignore")
    sight_lines["length"] = sight_lines["geometry"].length
    sight_lines = sight_lines.sort_values(['buildingID', 'nodeID', 'length'],ascending=[False, False, False]).drop_duplicates(['buildingID', 'nodeID'], keep='first')
    sight_lines.reset_index(inplace = True, drop = True)
       
    # stats
    stats = sight_lines.groupby('buildingID').agg({'length': ['mean','max', 'count']}) 
    stats.columns = stats.columns.droplevel(0)
    stats.rename(columns = {"count": "nr_lines"}, inplace = True)  
        
    # computing score on rescaled values
    stats["max"].fillna((stats["max"].min()), inplace = True) # longest sight_line to each building
    stats["mean"].fillna((stats["mean"].min()), inplace = True) # average distance sigh_lines to each buildings
    stats["nr_lines"].fillna((stats["nr_lines"].min()), inplace = True) # number of sigh_lines to each buildings
    stats.reset_index(inplace = True)
    col = ["max", "mean", "nr_lines"]      

    for i in col: 
        scaling_columnDF(stats, i)
    # computing the 3d visibility score
    if method == 'longest':
        stats["3dvis"] = stats["max_sc"]
    elif method == 'combined':
        stats["3dvis"] = stats["max_sc"]*0.5+tmp["mean_sc"]*0.25+tmp["nr_lines_sc"]*0.25

    # merging and building the final output
    buildings_gdf = pd.merge(buildings_gdf, stats[["buildingID", "3dvis"]], on = "buildingID", how = "left") 
    buildings_gdf['3dvis'] = buildings_gdf['3dvis'].where(pd.notnull(buildings_gdf['3dvis']), 0.0)
    
    return buildings_gdf, sight_lines

def _facade_area(building_geometry, building_height):

    """
    The function roughly computes the facade area of a building, given its geometry and height
     
    Parameters
    ----------
    building_geometry: Polygon
    building_height: float
   
    Returns
    -------
    float
    """
    
    envelope = building_geometry.envelope
    coords = mapping(envelope)["coordinates"][0]
    d = [(Point(coords[0])).distance(Point(coords[1])), (Point(coords[1])).distance(Point(coords[2]))]
    width = min(d)
    return width*building_height
 
def get_historical_buildings_fromOSM(place, download_method, epsg = None, distance = 1000):
    """    
    The function downloads and cleans buildings footprint geometries and create a buildings GeoDataFrames for the area of interest.
    The function exploits OSMNx functions for downloading the data as well as for projecting it.
    The land use classification for each building is extracted. Only relevant columns are kept.   
            
    Parameters
    ----------
    place: string, tuple
        name of cities or areas in OSM: when using "from point" please provide a (lat, lon) tuple to create the bounding box around it; when using "distance_from_address"
        provide an existing OSM address; when using "OSMplace" provide an OSM place name
    download_method: string, {"from_point", "distance_from_address", "OSMplace"}
        it indicates the method that should be used for downloading the data.
    epsg: int
        epsg of the area considered; if None OSMNx is used for the projection
    distance: float
        Specify distance from address or point
    
    Returns
    -------
    GeoDataFrames
    """   
    
    columns_to_keep = ['geometry', 'historic', 'heritage']

    if download_method == "distance_from_address": 
        historic_buildings = ox.geometries_from_address(address = place, distance = distance, tags={"building": True})
    elif download_method == "OSMplace": 
        historic_buildings = ox.geometries_from_place(place, tags={"building": True})
    elif download_method == "from_point": 
        historic_buildings = ox.geometries_from_point(point = place, distance = distance, tags={"building": True})
    else: raise downloadError('Provide a download method amongst {"from_point", "distance_from_address", "OSMplace"}')
    
    historic_buildings = historic_buildings[['geometry', 'historic', 'heritage']]
    historic_buildings = historic_buildings[~(historic_buildings.historic.isnull() & historic_buildings.heritage.isnull())]
    
    if epsg is None:
        historic_buildings = ox.projection.project_gdf(historic_buildings)
    else:
        crs = {'init': 'epsg:'+str(epsg), 'no_defs': True}
        historic_buildings = historic_buildings.to_crs(crs)

    historic_buildings["historic"] = 1
    historic_buildings["historic"][historic_buildings["historic"] != 0] = 1
    historic_buildings = historic_buildings[['geometry', 'historic']]
    historic_buildings['area'] = historic_buildings.geometry.area
       
    return historic_buildings
 
def cultural_score_from_dataset(buildings_gdf, historical_elements_gdf, score = None):

    """
    The function computes a cultural score based on the number of features listed in historical/cultural landmarks datasets. It can be
    obtained either on the basis of a score given by the data-provider or on the number of features intersecting the buildings object 
    of analysis.
    
    "score" indicates the attribute field containing scores assigned to historical buildings, if existing.
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        buildings GeoDataFrame - case study area
    historical_elements_gdf: Point or Polygon GeoDataFrame
    score: string
   
    Returns
    -------
    GeoDataFrame
    """
    
    buildings_gdf = buildings_gdf.copy()
    # spatial index
    sindex = historical_elements_gdf.sindex 
    buildings_gdf["cult"] = 0
    ix_geo = buildings_gdf.columns.get_loc("geometry")+1 
    
    buildings_gdf["cult"] = buildings_gdf.apply(lambda row: _compute_cultural_score_building(row["geometry"], historical_elements_gdf, sindex, score = score), axis = 1)
    return buildings_gdf
    
def _compute_cultural_score_building(building_geometry, historical_elements_gdf, historical_elements_gdf_sindex, score = None):

    """
    Compute pragmatic for a single building. It supports the function "pragmatic_meaning" 
     
    Parameters
    ----------
    buildings_geometry: Polygon
    building_land_use: string
    historical_elements_gdf_sindex: Rtree spatial index
    score: string
   
    Returns
    -------
    float
    """

    possible_matches_index = list(historical_elements_gdf_sindex.intersection(building_geometry.bounds)) # looking for possible candidates in the external GDF
    possible_matches = historical_elements_gdf.iloc[possible_matches_index]
    pm = possible_matches[possible_matches.intersects(building_geometry)]

    if (score is None):
        cs = len(pm) # score only based on number of intersecting elements
    elif len(pm) == 0: 
        cs = 0
    else: cs = pm[score].sum() # otherwise sum the scores of the intersecting elements
    
    return cs



def cultural_score_from_OSM(buildings_gdf):

    """
    The function computes a cultural score simply based on a binary categorisation. This function exploits the tag "historic" in OSM buildings data.
    When such field is filled in OSM, the building is considered semantically meaningful. 
    Therefore, the score is either 0 or 1.
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        buildings GeoDataFrame - case study area
   
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        the updated buildings GeoDataFrame
    """  
    
    buildings_gdf = buildings_gdf.copy()    
    buildings_gdf["cult"] = 0
    if "historic" not in buildings_gdf.columns:
        return buildings_gdf
    buildings_gdf["historic"][buildings_gdf["historic"].isnull()] = 0
    buildings_gdf["historic"][buildings_gdf["historic"] != 0] = 1
    buildings_gdf["cult"] = buildings_gdf["historic"]
        
    return buildings_gdf

def pragmatic_score(buildings_gdf, radius = 200):

    """
    Compute pragmatic score based on the frequency, and therefore unexpctdness, of a land_use class in an area around a building.
    The area is defined by the parameter "radius".
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        buildings GeoDataFrame - case study area
    buffer: float
   
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        the updated buildings GeoDataFrame
    """  
    
    buildings_gdf = buildings_gdf.copy()   
    buildings_gdf["nr"] = 1 # to count
    sindex = buildings_gdf.sindex # spatial index
    buildings_gdf["prag"] = buildings_gdf.apply(lambda row: _compute_pragmatic_meaning_building(row.geometry, row.land_use, buildings_gdf, sindex, radius), axis = 1)
    buildings_gdf.drop('nr', axis = 1, inplace = True)
    return buildings_gdf
    
def _compute_pragmatic_meaning_building(building_geometry, building_land_use, buildings_gdf, buildings_gdf_sindex, radius):

    """
    Compute pragmatic for a single building. It supports the function "pragmatic_meaning" 
     
    Parameters
    ----------
    buildings_geometry: Polygon
    building_land_use: String
    buildings_gdf: Polygon GeoDataFrame
        buildings GeoDataFrame - case study area
    buildings_gdf_sindex: Rtree Spatial index
    radius: float
   
    Returns
    -------
    float
    """

    buffer = building_geometry.buffer(radius)
    possible_matches_index = list(buildings_gdf_sindex.intersection(buffer.bounds))
    possible_matches = buildings_gdf.iloc[possible_matches_index]
    pm = possible_matches [possible_matches.intersects(buffer)]
    neigh = pm.groupby(["land_use"], as_index = True)["nr"].sum() 

    Nj = neigh.loc[building_land_use] # nr of neighbours with same land_use
    # Pj = Nj/N
    Pj = 1-(Nj/pm["nr"].sum()) # inverting the value
        
    return Pj
        
        
def compute_global_scores(buildings_gdf, g_cW, g_iW):
    """
    The function computes component and global scores, rescaling values when necessary and assigning weights to the different 
    properties measured.
    The user must provide two dictionaries:
    - g_cW: keys are component names (string), items are weights
    - g_iW: keys are index names (string), items are weights
    
    Example:
    g_cW = {"vScore": 0.50, "sScore" : 0.30, "cScore": 0.20, "pScore": 0.10}
    g_iW = {"3dvis": 0.50, "fac": 0.30, "height": 0.20, "area": 0.30, "2dvis":0.30, "neigh": 0.20 , "road": 0.20}
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
    g_cW, g_iW: dictionaries
   
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        the updated buildings GeoDataFrame
    """  
    
    # scaling
    col = ["3dvis", "fac", "height", "area","2dvis", "cult", "prag"]
    col_inverse = ["neigh", "road"]
    
    if "height" not in buildings_gdf.columns: 
        buildings_gdf['height'] = 0.0
    
    for i in col: 
        if buildings_gdf[i].max() == 0.0: 
            buildings_gdf[i+"_sc"] = 0.0
        else: 
            scaling_columnDF(buildings_gdf, i)
    
    for i in col_inverse: 
        if buildings_gdf[i].max() == 0.0: 
            buildings_gdf[i+"_sc"] = 0.0
        else: 
            scaling_columnDF(buildings_gdf, i, inverse = True) 
  
    # computing scores   
    buildings_gdf["vScore"] = buildings_gdf["fac_sc"]*g_iW["fac"] + buildings_gdf["height_sc"]*g_iW["height"] + buildings_gdf["3dvis_sc"]*g_iW["3dvis"]
    buildings_gdf["sScore"] = buildings_gdf["area_sc"]*g_iW["area"] + buildings_gdf["neigh_sc"]*g_iW["neigh"] + buildings_gdf["2dvis_sc"]*g_iW["2dvis"] + buildings_gdf["road_sc"]*g_iW["road"]
    
    # rescaling components
    col = ["vScore", "sScore"]
    for i in col: 
        if buildings_gdf[i].max() == 0.0: 
            buildings_gdf[i+"_sc"] = 0.0
        else: 
            scaling_columnDF(buildings_gdf, i)
    
    buildings_gdf["cScore"] = buildings_gdf["cult_sc"]
    buildings_gdf["pScore"] = buildings_gdf["prag_sc"]
    
    # final global score
    buildings_gdf["gScore"] = (buildings_gdf["vScore_sc"]*g_cW["vScore"] + buildings_gdf["sScore_sc"]*g_cW["sScore"] + 
                               buildings_gdf["cScore"]*g_cW["cScore"] + buildings_gdf["pScore"]*g_cW["pScore"])

    scaling_columnDF(buildings_gdf, "gScore")
    
    return buildings_gdf



def compute_local_scores(buildings_gdf, l_cW, l_iW, radius = 1500):

    """
    The function computes landmarkness at the local level. The components' weights may be different from the ones used to calculate the
    global score. The radius parameter indicates the extent of the area considered to rescale the landmarkness local score.
    - l_cW: keys are component names (string), items are weights.
    - l_iW: keys are index names (string), items are weights.
    
    # local landmarkness components weights
    l_cW = {"vScore": 0.25, "sScore" : 0.35, "cScore":0.10 , "pScore": 0.30}
    # local landmarkness indexes weights, cScore and pScore have only 1 index each
    l_iW = {"3dvis": 0.50, "fac": 0.30, "height": 0.20, "area": 0.40, "2dvis": 0.00, "neigh": 0.30 , "road": 0.30}
    
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
    l_cW, l_iW: dictionaries
   
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        the updated buildings GeoDataFrame
    """  
    
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf.index = buildings_gdf.buildingID
    buildings_gdf.index.name = None
    
    sindex = buildings_gdf.sindex # spatial index
    buildings_gdf["lScore"] = 0.0
    buildings_gdf["vScore_l"], buildings_gdf["sScore_l"] = 0.0, 0.0
    
    # recomputing the scores per each building in relation to its neighbours, in an area whose extent is regulated by the parameter "radius"
    buildings_gdf["lScore"] = buildings_gdf.apply(lambda row: _building_local_score(row["geometry"], row["buildingID"], buildings_gdf, sindex, l_cW, l_iW, radius), axis = 1)
    scaling_columnDF(buildings_gdf, "lScore")
    return buildings_gdf
    
def _building_local_score(building_geometry, buildingID, buildings_gdf, buildings_gdf_sindex, l_cW, l_iW, radius):

    """
    The function computes landmarkness at the local level for a single building. 

    
    Parameters
    ----------
    building_geometry: Polygon
    buildingID: int
    buildings_gdf: Polygon GeoDataFrame
    buildings_gdf_sindex: Rtree spatial index
    l_cW: dictionary
    l_iW: dictionary
    radius: float
        regulates the extension of the area wherein the scores are recomputed, around the building
   
    Returns
    -------
    score: Float
    """
                                             
    col = ["3dvis", "fac", "height", "area","2dvis", "cult","prag"]
    col_inverse = ["neigh", "road"]
    
    buffer = building_geometry.buffer(radius)
    possible_matches_index = list(buildings_gdf_sindex.intersection(buffer.bounds))
    possible_matches = buildings_gdf.iloc[possible_matches_index].copy()
    pm = possible_matches[possible_matches.intersects(buffer)]
    
    # rescaling the values 
    for i in col: 
        if pm[i].max() == 0.0: 
            pm[i+"_sc"] = 0.0
        else:
            scaling_columnDF(pm, i) 
    for i in col_inverse: 
        if pm[i].max() == 0.0: 
            pm[i+"_sc"] = 0.0
        else:
            scaling_columnDF(pm, i, inverse = True)
    
    # and recomputing scores
    pm["vScore_l"] =  pm["fac_sc"]*l_iW["fac"] + pm["height_sc"]*l_iW["height"] + pm["3dvis"]*l_iW["3dvis"]
    pm["sScore_l"] =  pm["area_sc"]*l_iW["area"]+ pm["neigh_sc"]*l_iW["neigh"] + pm["road_sc"]*l_iW["road"] + pm["2dvis_sc"]*l_iW["fac"]
    pm["cScore_l"] = pm["cult_sc"]
    pm["pScore_l"] = pm["prag_sc"]
    
    col_rs = ["vScore_l", "sScore_l"]
    for i in col_rs: 
        if pm[i].max() == 0.0: 
            pm[i+"_sc"] = 0.0
        else:
            scaling_columnDF(pm, i)
        
    pm["lScore"] =  pm["vScore_l_sc"]*l_cW["vScore"] + pm["sScore_l_sc"]*l_cW["sScore"] + pm["cScore_l"]*l_cW["cScore"] + pm["pScore_l"]*l_cW["pScore"]
    score = float("{0:.3f}".format(pm["lScore"].loc[buildingID]))
    # return the so obtined score
    return score

class Error(Exception):
    """Base class for other exceptions"""

class downloadError(Error):
    """Raised when a wrong download method is provided"""