import osmnx as ox
import pandas as pd
import numpy as np
import geopandas as gpd
import pyvista as pv

from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping, MultiLineString
from shapely.ops import linemerge, unary_union
from scipy.sparse import linalg
pd.set_option("display.precision", 3)

from .utilities import scaling_columnDF, polygon_2d_to_3d
from .angles import get_coord_angle

"""
This set of functions is designed for extracting the computational Image of The City.
Computational landmarks can be extracted employing the following functions.
"""
def downloadError(Exception):
    pass
 
def get_buildings_fromSHP(path, epsg, case_study_area = None, distance_from_center = 1000, height_field = None, base_field = None, 
    land_use_field = None):
    """    
    The function take a building footprint shapefile, returns two GDFs of buildings: the case-study area, plus a larger area containing other 
    buildings, called "obstructions" (for analyses which include adjacent buildings). Otherise, the user can provide a "distance from center" 
    value; in this case, the buildings_gdf are extracted by selecting buildings within a buffer from the center, with a radius equal to the 
    distance_from_center value. If none are passed, the buildings_gdf and the obstructions_gdf will be the same. 
    Additionally, provide the fields containing height, base elevation, and land use information.
            
    Parameters
    ----------
    path: string
        path where the file is stored
    epsg: int
        epsg of the area considered; if None OSMNx is used for the projection
    case_study_area: Polygon
        The Polygon to be use for clipping and identifying the case-study area, within the input .shp. If not available, use "distance_from_center"
    distance_from_center: float
        so to identify the case-study area on the basis of distance from the center of the input .shp
    height_field, base_field: str 
        height and base elevation fields name in the original data-source
    land_use_field: str 
        field that describes the land use of the buildings
    
    Returns
    -------
    buildings_gdf, obstructions_gdf: tuple of GeoDataFrames
        the buildings and the obstructions GeoDataFrames
    """   
    
    # trying reading buildings footprints shapefile from directory
    crs = 'EPSG:'+str(epsg)
    obstructions_gdf = gpd.read_file(path).to_crs(crs)  
    
    # computing area, reassigning columns
    obstructions_gdf["area"] = obstructions_gdf["geometry"].area

    obstructions_gdf["height"] = obstructions_gdf.get(height_field) or None
    obstructions_gdf["base"] = obstructions_gdf.get(base_field) or 0.0
    obstructions_gdf["land_use_raw"] = obstructions_gdf.get(land_use_field) or None

    # dropping small buildings and buildings with null height
    obstructions_gdf = obstructions_gdf[(obstructions_gdf["area"] >= 50) & (obstructions_gdf["height"] >= 1)]
    obstructions_gdf = obstructions_gdf[["height", "base","geometry", "area", "land_use_raw"]]
    # assigning ID
    obstructions_gdf["buildingID"] = obstructions_gdf.index.values.astype(int)
    
    # if case-study area and distance not defined
    if (case_study_area is None) and (distance_from_center is None):
        buildings_gdf = obstructions_gdf.copy()
        return buildings_gdf, obstructions_gdf

    if case_study_area is None:
        case_study_area = obstructions_gdf.geometry.unary_union.centroid.buffer(distance_from_center)

    buildings_gdf = obstructions_gdf[obstructions_gdf.geometry.within(case_study_area)]
    # clipping buildings in the case-study area

    return buildings_gdf, obstructions_gdf
    
def get_buildings_fromOSM(place, download_method: str, epsg = None, distance = 1000):
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
    download_options = {"distance_from_address", "OSMplace", "from_point", "OSMpolygon"}

    if download_method not in download_options:
        raise downloadError('Provide a download method amongst {}'.format(download_options))

    buildings_gdf = ox.geometries_from_address(address = place, dist = distance, tags={"building": True}) if download_method == "distance_from_address" else \
                    ox.geometries_from_place(place, tags={"building": True}) if download_method == "OSMplace" else \
                    ox.geometries_from_point(center_point = place, dist = distance, tags={"building": True}) if download_method == "from_point" else \
                    ox.geometries_from_polygon(place, tags={"building": True})
                    
    if epsg is None:
        buildings_gdf = ox.projection.project_gdf(buildings_gdf)
    else:
        crs = 'EPSG:'+str(epsg)
        buildings_gdf = buildings_gdf.to_crs(crs)

    buildings_gdf['land_use_raw'] = None
    buildings_gdf.loc[:, 'land_use_raw'] = buildings_gdf.filter(regex='^building:use:').apply(lambda x: x.name[13:] if x.notnull().any() else None)
    buildings_gdf.drop(columns=[col for col in buildings_gdf.columns if col not in columns_to_keep], inplace=True)

    # remove the empty geometries
    buildings_gdf = buildings_gdf[~buildings_gdf['geometry'].is_empty]
    # replace 'yes' with NaN in 'building' column
    buildings_gdf['building'].replace('yes', np.nan, inplace=True)
    # fill missing values in 'building' column with 'amenity' values
    buildings_gdf['building'].fillna(value=buildings_gdf['amenity'], inplace=True)
    # fill missing values in 'land_use_raw' column with 'building' values
    buildings_gdf['land_use_raw'].fillna(value=buildings_gdf['building'], inplace=True)
    # fill remaining missing values in 'land_use_raw' column with 'residential'
    buildings_gdf['land_use_raw'].fillna(value='residential', inplace=True)

    buildings_gdf = buildings_gdf[['geometry', 'historic', 'land_use_raw']]
    buildings_gdf['area'] = buildings_gdf.geometry.area
    buildings_gdf = buildings_gdf[buildings_gdf['area'] >= 50] 
    
    # reset index
    buildings_gdf = buildings_gdf.reset_index(drop = True)
    buildings_gdf['buildingID'] = buildings_gdf.index.values.astype('int')  
    
    return buildings_gdf

def structural_score(buildings_gdf, obstructions_gdf, edges_gdf, advance_vis_expansion_distance = 300, radius_neighbours = 150):
    """
    The function computes the "Structural Landmark Component" sub-scores of each building.
    It considers:
    - distance from the street network:
    - advance 2d visibility polygon;
    - number of neighbouring buildings in a given radius.
         
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        buildings GeoDataFrame - case study area
    edges_gdf: LineString GeoDataFrame
        street segmetns GeoDataFrame
    obstructions_gdf: Polygon GeoDataFrame
        obstructions GeoDataFrame  
    advance_vis_expansion_distance: float
        2d advance visibility - it indicates up to which distance from the building boundaries the 2dvisibility polygon can expand.
    radius_neighbours: float
        neighbours - research radius for other adjacent buildings.
        
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        the updated buildings GeoDataFrame
    """  
    
    buildings_gdf = buildings_gdf.copy()
    obstructions_gdf = buildings_gdf if obstructions_gdf is None else obstructions_gdf
    sindex = obstructions_gdf.sindex
    street_network = edges_gdf.geometry.unary_union

    buildings_gdf["road"] = buildings_gdf.geometry.distance(street_network)
    buildings_gdf["2dvis"] = buildings_gdf.geometry.apply(lambda row: visibility_polygon2d(row, obstructions_gdf, sindex, max_expansion_distance=
                                    advance_vis_expansion_distance))
    buildings_gdf["neigh"] = buildings_gdf.geometry.apply(lambda row: number_neighbours(row, obstructions_gdf, sindex, radius=radius_neighbours))

    return buildings_gdf
    
def number_neighbours(geometry, obstructions_gdf, obstructions_sindex, radius):
    """
    The function counts the number of neighbours, in a GeoDataFrame, around a given geometry, within a
    research radius.
     
    Parameters
    ----------
    geometry: Shapely Geometry
    obstructions_gdf: Polygon GeoDataFrame
        obstructions GeoDataFrame  
    obstructions_sindex: Rtree Spatial Index
    radius: float
        research radius for other adjacent buildings.
    
    Returns
    -------
    int
    """
        
    buffer = geometry.buffer(radius)
    possible_neigh_index = list(obstructions_sindex.intersection(buffer.bounds))
    possible_neigh = obstructions_gdf.iloc[possible_neigh_index]
    precise_neigh = possible_neigh[possible_neigh.intersects(buffer)]
    return len(precise_neigh)

def visibility_polygon2d(building_geometry, obstructions_gdf, obstructions_sindex, max_expansion_distance = 600):
    """
    It creates a 2d polygon of visibility around a polygon geometry (e.g. building footprint) and computes its area. 
    This can be considered as a measure of 2d advance visibility. Such a polygon is built by constructing lines around the centroid of the building,
    breaking them at obstructions and connecting the new formed geometries to get the final polygon.
    The "max_expansion_distance" parameter indicates up to which distance from the building boundaries the visibility polygon can expand.
     
    Parameters
    ----------
    building_geometry: Polygon
        the building geometry
    obstructions_gdf: Polygon GeoDataFrame
        obstructions GeoDataFrame  
    obstructions_sindex: Rtree Spatial Index
    max_expansion_distance: float
        it indicates up to which distance from the building boundaries the 2dvisibility polygon can expand.

    Returns
    -------
    float
    """

    # creating buffer
    distance_along = 10
    origin = building_geometry.centroid
    building_geometry = building_geometry.convex_hull if building_geometry.geom_type == 'MultiPolygon' else building_geometry
    max_expansion_distance += origin.distance(building_geometry.envelope.exterior)

    angles = np.arange(0, 360, distance_along)
    coords = np.array([get_coord_angle([origin.x, origin.y], distance=max_expansion_distance, angle=i) for i in angles])
    lines = [LineString([origin, Point(x)]) for x in coords]
    
    obstacles = obstructions_gdf[obstructions_gdf.crosses(unary_union(lines))]
    obstacles = obstacles[obstacles.geometry != building_geometry]
    obstacles = obstacles[~obstacles.geometry.within(building_geometry.convex_hull)]
    # creating lines all around the building till a defined distance
    
    if len(obstacles) > 0:
        ob = unary_union(obstacles.geometry)

        intersections = [line.intersection(ob) for line in lines]    
        clipped_lines = [LineString([origin, Point(intersection.geoms[0].coords[0])]) 
                         if ((type(intersection) == MultiLineString) & (not intersection.is_empty)) 
                         else  LineString([origin, Point(intersection.coords[0])]) 
                         if ((type(intersection) == LineString) & (not intersection.is_empty))                               
                         else LineString([origin, Point(intersection[0].coords[0])]) 
                         if ((type(intersection) == Point) & (not intersection.is_empty))
                         else line for intersection, line in zip(intersections, lines)]
    # the line are not interrupted, keeping the original ones
    else:
        clipped_lines = lines

    # creating a polygon of visibility based on the lines and their progression, taking into account the origin Point too    
    poly = Polygon([[p.x, p.y] for p in [origin] + [Point(line.coords[1]) for line in clipped_lines ] + [origin]])
    poly_vis = poly.difference(building_geometry)
    if poly_vis.is_empty:
        poly_vis = poly.buffer(0).difference(building_geometry) 
    
    return poly_vis.area  

def compute_3d_sight_lines(nodes_gdf, buildings_gdf, distance_along = 200, distance_min_observer_target = 300):
    """
    Computes the 3D sight lines between observer points in a Point GeoDataFrame and target buildings, based on a given distance along the buildings' exterior and a 
    minimum distance between observer and target (e.g. lines will not be constructed for observer points and targets whose distance is lower 
    than "distance_min_observer_target")
    
    Parameters
    ----------
    nodes_gdf: Point GeoDataFrame
        A GeoDataFrame containing the observer nodes, with at least a Point geometry column
    buildings_gdf: Polygon GeoDataFrame
        A GeoDataFrame containing the target buildings, with at least a Polygon geometry column
    distance_along: float 
        The distance along the exterior of the buildings for selecting target points.
    distance_min_observer_target: float 
        The minimum distance between observer and target
    
    Returns:
    ----------
    sight_lines: LineString GeoDataFrame
        the visibile sight lines GeoDataFrame
    
    """
    
    nodes_gdf = nodes_gdf.copy()
    buildings_gdf = buildings_gdf.copy()
    
    # add a 'base' column to the buildings GeoDataFrame with a default value of 0.0, if not provided
    if 'base' not in buildings_gdf.columns:
        buildings_gdf["base"] = 0.0
    
    def add_height(exterior, base, height):
        # create a new list of Point objects with the z-coordinate set to (height - base)
        coords = exterior.coords
        new_coords = [Point(x, y, height-base) for x, y in coords]
         # return a LineString constructed from the new Point objects
        return LineString(new_coords)
    
    # create a Series with the height-extended LineString objects
    building_exteriors_roof = buildings_gdf.apply(lambda row: add_height(row.geometry.exterior, row.base, row.height), axis=1)
    # create a list with the number of intervals along the exterior LineString to divide the line into
    num_intervals = [max(1, int(exterior.length / distance_along)) for exterior in building_exteriors_roof]
    # create a new column with the list of targets along the exterior LineString
    buildings_gdf['target'] = pd.Series([([(exterior.interpolate(min(exterior.length/2, distance_along)*i)) for i in range(x)]) 
              for exterior, x in zip(building_exteriors_roof, num_intervals)], index=buildings_gdf.index)
    
    # create a new column in the nodes GeoDataFrame with the Point objects representing the observer positions
    nodes_gdf['observer'] = nodes_gdf.apply(lambda row: Point(row.geometry.x, row.geometry.y, row.height), axis=1)
    # create a temporary dataframe with building targets
    tmp_buildings = buildings_gdf.explode('target')
    tmp_nodes = nodes_gdf[['nodeID', 'observer']].copy()
    tmp_buildings = tmp_buildings[['buildingID', 'target']].copy()
    
    # obtain the sight_lines dataframe by means of a cartesian product between GeoDataFrames
    sight_lines = pd.merge(tmp_nodes.assign(key=1), tmp_buildings.assign(key=1), on='key').drop('key', axis=1)
    # calculate the distance between observer and target and filter
    sight_lines['distance']= [p1.distance(p2) for p1, p2 in zip(sight_lines.observer, sight_lines.target)]
    sight_lines = sight_lines[sight_lines['distance'] >= distance_min_observer_target]
    # add geometry column with LineString connecting observer and target
    sight_lines['geometry'] = [LineString([p1, p2]) for p1, p2 in zip(sight_lines.observer, sight_lines.target)]
    sight_lines.reset_index(drop = True, inplace = True)

    # create pyvista 3d objects for buildings
    buildings_gdf['building_3d'] = [polygon_2d_to_3d(geo, base, height) for geo,base, height in zip(buildings_gdf['geometry'], buildings_gdf['base'], buildings_gdf['height'])]
    # extract tuples for starting and target points of the sight lines
    sight_lines['start'] = [[observer.x, observer.y, observer.z] for observer in sight_lines.observer]
    sight_lines['stop'] =[[target.x, target.y, target.z] for target in sight_lines.target]
    buildings_sindex = buildings_gdf.sindex
    
    # check intervisibility and filter out sight_lines that are obstructed
    sight_lines['visible'] = sight_lines.apply(lambda row: intervisibility(row.geometry, row.buildingID, row.start, row.stop, buildings_gdf, buildings_sindex), axis =1)
    sight_lines = sight_lines[sight_lines['visible'] == True]
    sight_lines.drop(['start', 'stop', 'observer', 'target', 'visible'], axis = 1, inplace = True)
    sight_lines = sight_lines.set_geometry('geometry').set_crs(buildings_gdf.crs)
    
    return sight_lines
   
def intervisibility(line2d, buildingID, start, stop, buildings_gdf, buildings_sindex) -> bool:
    """
    Check if a 3d line of sight between two points is obstructed by any buildings.
    
    Parameters
    ----------
    line2d: Shapely LineString
        The 2D line of sight to check for obstruction.
    buildingID: int
        The buildingID of the building that the line of sight points at.
    start: Tuple
        The starting point of the line of sight in the form of (x, y, z).
    stop: Tuple
        The ending point of the line of sight in the form of (x, y, z).
    
    Returns:
    ----------
    bool: True if the line of sight is not obstructed, False otherwise.
    """

    # first just check for 2d intersections and considers the ones that have a 2d intersection.
    # if there is no 2d intersection, it is not necessary to check for the 3d one.
    possible_matches_index = list(buildings_sindex.intersection(line2d.buffer(5).bounds)) # looking for possible candidates in the external GDF
    possible_matches = buildings_gdf.iloc[possible_matches_index]
    pm = possible_matches[possible_matches.intersects(line2d)]
    pm = pm[pm.buildingID != buildingID]
    if len(pm) == 0:
        return True
    
    # if there are 2d intersections, check for 3d ones
    visible = True
    for _, row_building in pm.iterrows():
        building_3d = row_building.building_3d.extract_surface().triangulate()
        points, intersections = building_3d.ray_trace(start, stop)
        if len(intersections) > 0:
            return False
            
    return visible 
  
   
def visibility_score(buildings_gdf, sight_lines = pd.DataFrame({'a' : []}), method = 'longest'):
    """
    The function calculates the sub-scores of the "Visibility Landmark Component".
    - 3d visibility;
    - facade area;
    - (height).
     
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

    
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf["fac"] = 0.0
    if ("height" not in buildings_gdf.columns) | (sight_lines.empty): 
        return buildings_gdf, sight_lines

    sight_lines['nodeID'] = sight_lines['nodeID'].astype(int)
    sight_lines['buildingID'] = sight_lines['buildingID'].astype(int)

    buildings_gdf["fac"] = buildings_gdf.apply(lambda row: _facade_area(row["geometry"], row["height"]), axis = 1)
    sight_lines["length"] = sight_lines["geometry"].length
    sight_lines = sight_lines.sort_values(['buildingID', 'nodeID', 'length'],ascending=[False, False, False]).drop_duplicates(['buildingID', 'nodeID'], keep='first')
    sight_lines.reset_index(inplace = True, drop = True)

    stats = sight_lines.groupby('buildingID').agg({'length': ['mean','max', 'count']}) 
    stats.columns = stats.columns.droplevel(0)
    stats.rename(columns = {"count": "nr_lines"}, inplace = True)

    stats["max"].fillna((stats["max"].min()), inplace = True)
    stats["mean"].fillna((stats["mean"].min()), inplace = True)
    stats["nr_lines"].fillna((stats["nr_lines"].min()), inplace = True)
    stats.reset_index(inplace = True)
    col = ["max", "mean", "nr_lines"]

    for i in col:
        scaling_columnDF(stats, i)

    if method == 'longest':
        stats["3dvis"] = stats["max_sc"]
    elif method == 'combined':
        stats["3dvis"] = stats["max_sc"]*0.5+stats["mean_sc"]*0.25+stats["nr_lines_sc"]*0.25

    buildings_gdf = pd.merge(buildings_gdf, stats[["buildingID", "3dvis"]], on = "buildingID", how = "left") 
    buildings_gdf['3dvis'] = buildings_gdf['3dvis'].where(pd.notnull(buildings_gdf['3dvis']), 0.0)
    
    return buildings_gdf, sight_lines

def facade_area(building_geometry, building_height):
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
    
    columns = ['geometry', 'historic']

    method_mapping = {
        "distance_from_address": ox.geometries_from_address,
        "OSMplace": ox.geometries_from_place,
        "from_point": ox.geometries_from_point,
        "polygon": ox.geometries_from_polygon
    }

    try:
        download_method_func = method_mapping[download_method]
        if download_method == "distance_from_address":
            historic_buildings = download_method_func(place, dist = distance, tags={"building": True})
        elif download_method == "from_point":
            historic_buildings = download_method_func(place, dist = distance, tags={"building": True})
        else:
            historic_buildings = download_method_func(place, tags={"building": True})
    except KeyError:
        raise downloadError('Provide a download method amongst {"from_point", "distance_from_address", "OSMplace", "polygon"}')
    
    if 'heritage' in historic_buildings:
        columns.append('heritage')
    historic_buildings = historic_buildings[columns]

    if 'heritage' in historic_buildings:
        historic_buildings = historic_buildings[~(historic_buildings.historic.isnull() & historic_buildings.heritage.isnull())]
    else:
        historic_buildings = historic_buildings[~historic_buildings.historic.isnull()]
    
    if epsg is None:
        historic_buildings = ox.projection.project_gdf(historic_buildings)
    else:
        crs = 'EPSG:'+str(epsg)
        historic_buildings = historic_buildings.to_crs(crs)

    historic_buildings.loc[historic_buildings["historic"] != 0, "historic"] = 1
    historic_buildings = historic_buildings[['geometry', 'historic']]
    historic_buildings['area'] = historic_buildings.geometry.area
       
    return historic_buildings
 
def cultural_score(buildings_gdf, historical_elements_gdf = pd.DataFrame({'a' : []}), historical_score = None, from_OSM = False, ):
    """
    The function computes the "Cultural Landmark Component" based on the number of features listed in historical/cultural landmarks datasets. It can be
    obtained either on the basis of a score given by the data-provider or on the number of features intersecting the buildings object 
    of analysis.
    
    "historical_score" indicates the attribute field containing scores assigned to historical buildings, if existing.
    Alternatively, if the column has been already assigned to the buildings_gdf, one can use OSM historic categorisation (binbary).
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        buildings GeoDataFrame - case study area
    historical_elements_gdf: Point or Polygon GeoDataFrame
        The GeoDataFrame containing information about listed historical buildings or elements
    historical_score: string
        The name of the column in the historical_elements_gdf that provides information on the classification of the historical listings, Optional
    from_OSM: boolean
        if using the historic field from OSM. This column should be already in the buildings_gdf columns
   
    Returns
    -------
    GeoDataFrame
    """
    
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf["cult"] = 0
    
    # using OSM binary value
    if (from_OSM) & ("historic" in buildings_gdf.columns):
        buildings_gdf["historic"][buildings_gdf["historic"].isnull()] = 0
        buildings_gdf["historic"][buildings_gdf["historic"] != 0] = 1
        buildings_gdf["cult"] = buildings_gdf["historic"]
        return buildings_gdf
    
    def _cultural_score_building(building_geometry, historical_elements_gdf, historical_elements_gdf_sindex, score = None):
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
        else: 
            cs = pm[score].sum() # otherwise sum the scores of the intersecting elements
        
        return cs

    # spatial index
    sindex = historical_elements_gdf.sindex 
    buildings_gdf["cult"] = buildings_gdf.geometry.apply(lambda row: _cultural_score_building(row, historical_elements_gdf, sindex, score = historical_score), axis = 1)
    return buildings_gdf

def pragmatic_score(buildings_gdf, research_radius = 200):
    """
    The function computes the "Pragmatic Landmark Component" based on the frequency, and therefore unexpctdness, of a land_use class in an area around a building.
    The area is defined by the parameter "research_radius".
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        buildings GeoDataFrame - case study area
    research_radius: float
   
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        the updated buildings GeoDataFrame
    """  
    
    buildings_gdf = buildings_gdf.copy()   
    buildings_gdf["nr"] = 1 # to count
    sindex = buildings_gdf.sindex # spatial index
    
    if 'land_use' not in buildings_gdf.columns:
        buildings_gdf['land_use'] = buildings_gdf['land_use_raw']
        
    def _pragmatic_meaning_building(building_geometry, building_land_use, buildings_gdf, buildings_gdf_sindex, radius):
        """
        Compute the pragmatic score for a single building.
         
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
        pm = possible_matches[possible_matches.intersects(buffer)]
        neigh = pm.groupby(["land_use"], as_index = True)["nr"].sum() 

        Nj = neigh.loc[building_land_use] # nr of neighbours with same land_use
        Pj = 1-(Nj/pm["nr"].sum()) # inverting the value # Pj = Nj/N
            
        return Pj    
        
    buildings_gdf["prag"] = buildings_gdf.apply(lambda row: _pragmatic_meaning_building(row.geometry, row.land_use, buildings_gdf, 
                                        sindex, radius = research_radius), axis = 1)
    buildings_gdf.drop('nr', axis = 1, inplace = True)
    return buildings_gdf
    
def compute_global_scores(buildings_gdf, g_cW, g_iW):
    """
    The function computes the component and global scores, rescaling values when necessary and assigning weights to the different 
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

    # keeping out visibility analysis if there is no information about buildings' elevation. Reassigning the weights to the other
    # components if any was given to the visibility
    if ("height" not in buildings_gdf.columns) or (("height" in buildings_gdf.columns) and (buildings_gdf.height.max() == 0.0)):
        buildings_gdf[['height','3dvis', 'fac']]  = 0.0
        if g_cW['vScore'] != 0.0:
            to_add = g_cW['vScore']/3
            g_cW['sScore'] += to_add
            g_cW['cScore'] += to_add
            g_cW['pScore'] += to_add
            g_cW['vScore'] = 0.0
    
    # rescaling values from 0 to 1
    for i in col + col_inverse:
        if buildings_gdf[i].max() == 0.0:
            buildings_gdf[i+"_sc"] = 0.0
        else:
            scaling_columnDF(buildings_gdf, i, inverse = i in col_inverse)
  
    # computing component scores   
    vScore_terms = [buildings_gdf["fac_sc"] * weights["fac"],
                    buildings_gdf["height_sc"] * weights["height"],
                    buildings_gdf["3dvis_sc"] * weights["3dvis"]]
    buildings_gdf["vScore"] = sum(vScore_terms)

    sScore_terms = [buildings_gdf["area_sc"] * weights["area"],
                    buildings_gdf["neigh_sc"] * weights["neigh"],
                    buildings_gdf["2dvis_sc"] * weights["2dvis"],
                    buildings_gdf["road_sc"] * weights["road"]]    
    buildings_gdf["sScore"] = sum(sScore_terms)
    
    # rescaling them
    col = ["vScore", "sScore"]
    buildings_gdf = buildings_gdf.assign(**{f'{i}_sc': scaling_columnDF(buildings_gdf, i) if buildings_gdf[i].max() != 0.0 else 0.0 for i in col})
    
    buildings_gdf["cScore"] = buildings_gdf["cult_sc"]
    buildings_gdf["pScore"] = buildings_gdf["prag_sc"]
    # final global score
    buildings_gdf["gScore"] = (buildings_gdf["vScore_sc"]*g_cW["vScore"] + buildings_gdf["sScore_sc"]*g_cW["sScore"] + 
                               buildings_gdf["cScore"]*g_cW["cScore"] + buildings_gdf["pScore"]*g_cW["pScore"])

    scaling_columnDF(buildings_gdf, "gScore")
    
    return buildings_gdf

def compute_local_scores(buildings_gdf, l_cW, l_iW, rescaling_radius = 1500):
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
        col_all = col + col_inverse
        
        buffer = building_geometry.buffer(radius)
        possible_matches_index = list(buildings_gdf_sindex.intersection(buffer.bounds))
        pm = buildings_gdf.iloc[possible_matches_index].copy()
        pm = pm[pm.intersects(buffer)]
        
        # rescaling the values
        pm[col_all + [col+"_sc" for col in col_all]] = (pm[col_all] != 0.0).select(
            lambda x: scaling_columnDF(pm, x, inverse = x in col_inverse), 0.0)
        # recomputing scores
        vScore_terms = [pm["fac_sc"] * weights["fac"],
                        pm["height_sc"] * weights["height"],
                        pm["3dvis"] * weights["3dvis"]]
        pm["vScore_l"] = sum(vScore_terms)

        sScore_terms = [pm["area_sc"] * weights["area"],
                        pm["neigh_sc"] * weights["neigh"],
                        pm["road_sc"] * weights["road"],
                        pm["2dvis_sc"] * weights["fac"]]
        pm["sScore_l"] = sum(sScore_terms)
       
        pm["cScore_l"] = pm["cult_sc"]
        pm["pScore_l"] = pm["prag_sc"]
        pm[["vScore_l", "sScore_l"] + [col+"_sc" for col in ["vScore_l", "sScore_l"]]] = (pm[["vScore_l", "sScore_l"]] != 0.0).select(
            lambda x: scaling_columnDF(pm, x), 0.0)
            
        pm["lScore"] = pm["vScore_l_sc"]*l_cW["vScore"] + pm["sScore_l_sc"]*l_cW["sScore"] + pm["cScore_l"]*l_cW["cScore"] + pm["pScore_l"]*l_cW["pScore"]
        local_score = float("{0:.3f}".format(pm.loc[buildingID, "lScore"]))
        
        return local_score
    
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_scores = {executor.submit(_building_local_score, row["geometry"], row["buildingID"], buildings_gdf, sindex, l_cW, l_iW, rescaling_radius): row["buildingID"] 
                                            for _, row in buildings_gdf.iterrows()}
        for future in concurrent.futures.as_completed(future_scores):
            buildingID = future_scores[future]
            try:
                score = future.result()
                buildings_gdf.loc[buildingID, "lScore"] = score
            except Exception as exc:
                print(f'{buildingID} generated an exception: {exc}')
    
    scaling_columnDF(buildings_gdf, "lScore")
    
    return buildings_gdf
    
class Error(Exception):
    """Base class for other exceptions"""

class downloadError(Error):
    """Raised when a wrong download method is provided"""