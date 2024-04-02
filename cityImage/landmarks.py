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
from .utilities import scaling_columnDF, polygon_2d_to_3d, downloader
from .angles import get_coord_angle

def get_buildings_fromFile(path, epsg, case_study_area = None, distance_from_center = 1000, height_field = None, base_field = None, 
    land_use_field = None):
    """    
    The function take a building footprint .shp or .gpkg, returns two GDFs of buildings: the case-study area, plus a larger area containing other 
    buildings, called "obstructions" (for analyses which include adjacent buildings). Otherise, the user can provide a "distance from center" 
    value; in this case, the buildings_gdf are extracted by selecting buildings within a buffer from the center, with a radius equal to the 
    distance_from_center value. If none are passed, the buildings_gdf and the obstructions_gdf will be the same. 
    Additionally, provide the fields containing height, base elevation, and land use information.
            
    Parameters
    ----------
    path: str
        Path where the file is stored.
    epsg: int
        Epsg of the area considered; if None OSMNx is used for the projection.
    case_study_area: Polygon
        The Polygon to be use for clipping and identifying the case-study area, within the input .shp. If not available, use "distance_from_center".
    distance_from_center: float
        So to identify the case-study area on the basis of distance from the center of the input .shp.
    height_field, base_field: str 
        Height and base elevation fields name in the original data-source.
    land_use_field: str 
        Field that describes the land use of the buildings.
    
    Returns
    -------
    buildings_gdf, obstructions_gdf: tuple of GeoDataFrames
        The buildings and the obstructions GeoDataFrames.
    """   
    # trying reading buildings footprints shapefile from directory
    crs = 'EPSG:'+str(epsg)
    obstructions_gdf = gpd.read_file(path).to_crs(crs)  
    
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
    The function downloads and cleans building footprint geometries and create a buildings GeoDataFrames for the area of interest.
    The function exploits OSMNx functions for downloading the data as well as for projecting it.
    The land use classification for each building is extracted. Only relevant columns are kept.   
            
    Parameters
    ----------
    place: str, tuple, Shapely Polygon
        Name of cities or areas in OSM: 
        - when using "distance_from_point" please provide a (lat, lon) tuple to create the bounding box around it; 
        - when using "distance_from_address" provide an existing OSM address; 
        - when using "OSMplace" provide an OSM place name. The query must be geocodable and OSM must have polygon boundaries for the geocode result.  
        - when using "polygon" please provide a Shapely Polygon in unprojected latitude-longitude degrees (EPSG:4326) CRS;
    download_method: str, {"distance_from_address", "distance_from_point", "OSMplace", "polygon"}
        It indicates the method that should be used for downloading the data.
    epsg: int
        Epsg of the area considered; if None OSMNx is used for the projection.
    distance: float
        Used when download_method == "distance from address" or == "distance from point".
    
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        The buildings GeoDataFrame.
    """   
    columns_to_keep = ['amenity', 'building', 'geometry', 'historic', 'land_use_raw']
    tags = {"building": True}
    buildings_gdf = downloader(place = place, download_method = download_method, tags = tags, distance = distance)
    
    if epsg is None:
        buildings_gdf = ox.projection.project_gdf(buildings_gdf)
    else:
        crs = 'EPSG:'+str(epsg)
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
    obstructions_gdf = buildings_gdf if obstructions_gdf is None else obstructions_gdf
    sindex = obstructions_gdf.sindex
    street_network = edges_gdf.geometry.unary_union

    buildings_gdf["road"] = buildings_gdf.geometry.distance(street_network)
    buildings_gdf["2dvis"] = buildings_gdf.geometry.apply(lambda row: visibility_polygon2d(row, obstructions_gdf, sindex, max_expansion_distance=
                                    advance_vis_expansion_distance))
    buildings_gdf["neigh"] = buildings_gdf.geometry.apply(lambda row: number_neighbours(row, obstructions_gdf, sindex, radius=neighbours_radius))

    return buildings_gdf
    
def number_neighbours(geometry, obstructions_gdf, obstructions_sindex, radius):
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

def visibility_polygon2d(building_geometry, obstructions_gdf, obstructions_sindex, max_expansion_distance = 600):
    """
    It creates a 2d polygon of visibility around a polygon geometry (e.g. building footprint) and computes its area. 
    This can be considered as a measure of 2d advance visibility. Such a polygon is built by constructing lines around the centroid of the building,
    breaking them at obstructions and connecting the new formed geometries to get the final polygon.
    The "max_expansion_distance" parameter indicates up to which distance from the building boundaries the visibility polygon can expand.
     
    Parameters
    ----------
    building_geometry: Polygon
        The building geometry.
    obstructions_gdf: Polygon GeoDataFrame
        Obstructions GeoDataFrame.
    obstructions_sindex: Spatial Index
        The spatial index of the obstructions GeoDataFrame.
    max_expansion_distance: float
        It indicates up to which distance from the building boundaries the 2dvisibility polygon can expand.

    Returns
    -------
    float
        The area of visibility.
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
                         else LineString([origin, Point(intersection.coords[0])]) 
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
        A GeoDataFrame containing the observer nodes, with at least a Point geometry column.
    buildings_gdf: Polygon GeoDataFrame
        A GeoDataFrame containing the target buildings, with at least a Polygon geometry column.
    distance_along: float 
        The distance along the exterior of the buildings for selecting target points.
    distance_min_observer_target: float 
        The minimum distance between observer and target.
    
    Returns:
    ----------
    sight_lines: LineString GeoDataFrame
        The visibile sight lines GeoDataFrame.
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
    nodes_gdf['observer'] = nodes_gdf.apply(lambda row: Point(row.geometry.x, row.geometry.y, row.z), axis=1)
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
   
def intervisibility(line2d, buildingID, start, stop, buildings_gdf, buildings_sindex):
    """
    Check if a 3d line of sight between two points is obstructed by any buildings.
    
    Parameters
    ----------
    line2d: Shapely LineString
        The 2D line of sight to check for obstruction.
    buildingID: int
        The buildingID of the building that the line of sight points at.
    start: tuple
        The starting point of the line of sight in the form of (x, y, z).
    stop: tuple
        The ending point of the line of sight in the form of (x, y, z).
    
    Returns:
    ----------
    bool: True 
        When the line of sight is not obstructed, False otherwise.
    """
    # first just check for 2d intersections and considers the ones that have a 2d intersection.
    # if there is no 2d intersection, it is not necessary to check for the 3d one.
    possible_matches_index = list(buildings_sindex.intersection(line2d.buffer(5).bounds)) # looking for possible candidates in the external GDF
    possible_matches = buildings_gdf.iloc[possible_matches_index]
    matches = possible_matches[possible_matches.intersects(line2d)]
    matches = matches[matches.buildingID != buildingID]
    if len(matches) == 0:
        return True
    
    # if there are 2d intersections, check for 3d ones
    visible = True
    for _, row_building in matches.iterrows():
        building_3d = row_building.building_3d.extract_surface().triangulate()
        points, intersections = building_3d.ray_trace(start, stop)
        if len(intersections) > 0:
            return False
            
    return visible 
  
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

    buildings_gdf["fac"] = buildings_gdf.apply(lambda row: facade_area(row["geometry"], row["height"]), axis = 1)
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
    columns = ["max", "mean", "nr_lines"]

    for column in columns:
        stats[column+"_sc"] = scaling_columnDF(stats[column])

    if method == 'longest':
        stats["3dvis"] = stats["max_sc"]
    elif method == 'combined':
        stats["3dvis"] = stats["max_sc"]*0.5+stats["mean_sc"]*0.25+stats["nr_lines_sc"]*0.25

    buildings_gdf = pd.merge(buildings_gdf, stats[["buildingID", "3dvis"]], on = "buildingID", how = "left") 
    buildings_gdf['3dvis'] = buildings_gdf['3dvis'].where(pd.notnull(buildings_gdf['3dvis']), 0.0)
    
    return buildings_gdf, sight_lines

def facade_area(building_geometry, building_height):
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
 
def get_historical_buildings_fromOSM(place, download_method, epsg = None, distance = 1000):
    """    
    The function downloads and cleans building footprint geometries and create a buildings GeoDataFrames for the area of interest.
    However, it only keeps the buildings that are considered historical buildings or heritage buildings in OSM. 
            
    Parameters
    ----------
    place: str, tuple, Shapely Polygon
        Name of cities or areas in OSM: 
        - when using "distance_from_point" please provide a (lat, lon) tuple to create the bounding box around it; 
        - when using "distance_from_address" provide an existing OSM address; 
        - when using "OSMplace" provide an OSM place name. The query must be geocodable and OSM must have polygon boundaries for the geocode result.  
        - when using "polygon" please provide a Shapely Polygon in unprojected latitude-longitude degrees (EPSG:4326) CRS;
    download_method: str, {"distance_from_address", "distance_from_point", "OSMplace", "polygon"}
        It indicates the method that should be used for downloading the data.
    epsg: int
        Epsg of the area considered; if None OSMNx is used for the projection.
    distance: float
        Used when download_method == "distance from address" or == "distance from point".
    
    Returns
    -------
    historic_buildings: Polygon GeoDataFrame
        The historic buildings GeoDataFrame.
    """   
    
    columns = ['geometry', 'historic']
    tags = {"building": True}
    historic_buildings = downloader(place = place, download_method = download_method, tags = tags, distance = distance)
    
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
 
def cultural_score(buildings_gdf, historical_elements_gdf = pd.DataFrame({'a': []}), historical_score = None, from_OSM = False):
    """
    The function computes the "Cultural Landmark Component" based on the number of features listed in historical/cultural landmarks datasets. It can be
    obtained either on the basis of a score given by the data-provider or on the number of features intersecting the buildings object 
    of analysis.
    
    "historical_score" indicates the attribute field containing scores assigned to historical buildings, if existing.
    Alternatively, if the column has been already assigned to the buildings_gdf, one can use OSM historic categorisation (binbary).
     
    Parameters
    ----------
    buildings_gdf: Polygon GeoDataFrame
        Buildings GeoDataFrame - case study area.
    historical_elements_gdf: Point or Polygon GeoDataFrame
        The GeoDataFrame containing information about listed historical buildings or elements.
    historical_score: str
        The name of the column in the historical_elements_gdf that provides information on the classification of the historical listings.
    from_OSM: boolean
        If using the historic field from OSM. This column should be already in the buildings_gdf columns.
   
    Returns
    -------
    buildings_gdf: Polygon GeoDataFrame
        The updated buildings GeoDataFrame.
    """  
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf["cult"] = 0
    # using OSM binary value
    if (from_OSM) & ("historic" in buildings_gdf.columns):
        # Set 'historic' column to 0 where it is currently null
        buildings_gdf.loc[buildings_gdf["historic"].isnull(), "historic"] = 0
        # Set 'historic' column to 1 where it is not 0
        buildings_gdf.loc[buildings_gdf["historic"] != 0, "historic"] = 1
        buildings_gdf["cult"] = buildings_gdf["historic"]
        return buildings_gdf
    
    def _cultural_score_building(building_geometry, historical_elements_gdf, historical_elements_gdf_sindex, score = None):
        """
        Compute the cultural score for a single building based on its intersection with historical elements.

        Parameters
        ----------
        building_geometry: Polygon
            The geometry of the building.
        historical_elements_gdf: GeoDataFrame
            The GeoDataFrame containing historical elements.
        historical_elements_gdf_sindex: Spatial Index
            The spatial index of the historical elements GeoDataFrame.
        score: str
            The name of the score column in the historical elements GeoDataFrame, by default None.

        Returns
        -------
        float
            The computed cultural score for the building.
        """
        possible_matches_index = list(historical_elements_gdf_sindex.intersection(building_geometry.bounds)) # looking for possible candidates in the external GDF
        possible_matches = historical_elements_gdf.iloc[possible_matches_index]
        matches = possible_matches[possible_matches.intersects(building_geometry)]

        if (score is None):
            cs = len(matches) # score only based on number of intersecting elements
        elif len(matches) == 0: 
            cs = 0
        else: 
            cs = matches[score].sum() # otherwise sum the scores of the intersecting elements
        return cs

    # spatial index
    sindex = historical_elements_gdf.sindex 
    buildings_gdf["cult"] = buildings_gdf.geometry.apply(lambda row: _cultural_score_building(row, historical_elements_gdf, sindex, score = historical_score))
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
        """
        Compute the pragmatic score for a single building based on its proximity to buildings with the same land use.

        Parameters
        ----------
        building_geometry: Polygon
            The geometry of the building.
        building_land_use: str
            The land use category of the building.
        buildings_gdf: Polygon GeoDataFrame
            The buildings GeoDataFrame for the case study area.
        buildings_gdf_sindex: Spatial Index
            The spatial index of the buildings GeoDataFrame.
        radius: float
            The radius to consider for proximity calculation.

        Returns
        -------
        float
            The computed pragmatic score for the building.
        """
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
    >>> import geopandas as gpd
    >>> buildings_gdf = gpd.GeoDataFrame(...)
    >>> global_indexes_weights = {"3dvis": 0.50, "fac": 0.30, "height": 0.20, "area": 0.30, "2dvis": 0.30, "neigh": 0.20, "road": 0.20}
    >>> global_components_weights = {"vScore": 0.50, "sScore": 0.30, "cScore": 0.20, "pScore": 0.10}
    >>> buildings_gdf_scores = compute_global_scores(buildings_gdf, global_indexes_weights, global_components_weights)
    >>> print(buildings_gdf_scores)
    """  
    # scaling
    col = ["3dvis", "fac", "height", "area","2dvis", "cult", "prag"]
    col_inverse = ["neigh", "road"]

    # keeping out visibility analysis if there is no information about buildings' elevation. Reassigning the weights to the other
    # components if any was given to the visibility
    if ("height" not in buildings_gdf.columns) or (("height" in buildings_gdf.columns) and (buildings_gdf.height.max() == 0.0)):
        buildings_gdf[['height','3dvis', 'fac']]  = 0.0
        if global_components_weights['vScore'] != 0.0:
            to_add = global_components_weights['vScore']/3
            global_components_weights['sScore'] += to_add
            global_components_weights['cScore'] += to_add
            global_components_weights['pScore'] += to_add
            global_components_weights['vScore'] = 0.0
    
    # rescaling values from 0 to 1
    for column in col + col_inverse:
        if buildings_gdf[column].max() == 0.0:
            buildings_gdf[column+"_sc"] = 0.0
        else:
            buildings_gdf[column+"_sc"] = scaling_columnDF(buildings_gdf[column], inverse = column in col_inverse)
  
    # computing component scores   
    vScore_terms = [buildings_gdf["fac_sc"] * global_indexes_weights["fac"],
                    buildings_gdf["height_sc"] * global_indexes_weights["height"],
                    buildings_gdf["3dvis_sc"] * global_indexes_weights["3dvis"]]
    buildings_gdf["vScore"] = sum(vScore_terms)

    sScore_terms = [buildings_gdf["area_sc"] * global_indexes_weights["area"],
                    buildings_gdf["neigh_sc"] * global_indexes_weights["neigh"],
                    buildings_gdf["2dvis_sc"] * global_indexes_weights["2dvis"],
                    buildings_gdf["road_sc"] * global_indexes_weights["road"]]    
    buildings_gdf["sScore"] = sum(sScore_terms)
    
    # rescaling them
    for column in ["vScore", "sScore"]: 
        if buildings_gdf[column].max() == 0.0: 
            buildings_gdf[column+"_sc"] = 0.0
        else: 
            buildings_gdf[column+"_sc"] = scaling_columnDF(buildings_gdf[column])
    
    buildings_gdf["cScore"] = buildings_gdf["cult_sc"]
    buildings_gdf["pScore"] = buildings_gdf["prag_sc"]
    # final global score
    buildings_gdf["gScore"] = (buildings_gdf["vScore_sc"]*global_components_weights["vScore"] + buildings_gdf["sScore_sc"]*global_components_weights["sScore"] + 
                               buildings_gdf["cScore"]*global_components_weights["cScore"] + buildings_gdf["pScore"]*global_components_weights["pScore"])

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
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf.index = buildings_gdf.buildingID
    buildings_gdf.index.name = None
    
    sindex = buildings_gdf.sindex # spatial index
    buildings_gdf["lScore"] = 0.0
    buildings_gdf["vScore_l"], buildings_gdf["sScore_l"] = 0.0, 0.0
    
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
                                                 
        col = ["3dvis", "fac", "height", "area","2dvis", "cult","prag"]
        col_inverse = ["neigh", "road"]
        
        buffer = building_geometry.buffer(radius)
        possible_matches_index = list(buildings_gdf_sindex.intersection(buffer.bounds))
        possible_matches = buildings_gdf.iloc[possible_matches_index].copy()
        matches = possible_matches[possible_matches.intersects(buffer)]
                    
        # rescaling the values 
        for column in col + col_inverse: 
            if matches[column].max() == 0.0: 
                matches[column+"_sc"] = 0.0
            else:
                matches[column+"_sc"] = scaling_columnDF(matches[column], inverse = column in col_inverse)
      
        # recomputing scores
        vScore_terms = [matches["fac_sc"] * local_indexes_weights["fac"],
                        matches["height_sc"] * local_indexes_weights["height"],
                        matches["3dvis"] * local_indexes_weights["3dvis"]]
        matches["vScore_l"] = sum(vScore_terms)

        sScore_terms = [matches["area_sc"] * local_indexes_weights["area"],
                        matches["neigh_sc"] * local_indexes_weights["neigh"],
                        matches["road_sc"] * local_indexes_weights["road"],
                        matches["2dvis_sc"] * local_indexes_weights["fac"]]
        matches["sScore_l"] = sum(sScore_terms)
       
        matches["cScore_l"] = matches["cult_sc"]
        matches["pScore_l"] = matches["prag_sc"]
        
        for column in ["vScore_l", "sScore_l"]: 
            if matches[column].max() == 0.0: 
                matches[column+"_sc"] = 0.0
            else:
                matches[column+"_sc"] = scaling_columnDF(matches[column])
        
        lScore_terms = [matches["vScore_l_sc"]*local_components_weights["vScore"],
                        matches["sScore_l_sc"]*local_components_weights["sScore"],
                        matches["cScore_l"]*local_components_weights["cScore"], 
                        matches["pScore_l"]*local_components_weights["pScore"]]
        matches["lScore"] = sum(lScore_terms)
        
        local_score = float("{0:.3f}".format(matches.loc[buildingID, "lScore"]))
        
        return local_score
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_scores = {executor.submit(_building_local_score, row["geometry"], row["buildingID"], buildings_gdf, sindex, local_components_weights, local_indexes_weights, 
                        rescaling_radius): row["buildingID"] for _, row in buildings_gdf.iterrows()}
        for future in concurrent.futures.as_completed(future_scores):
            buildingID = future_scores[future]
            try:
                score = future.result()
                buildings_gdf.loc[buildingID, "lScore"] = score
            except Exception as exc:
                print(f'{buildingID} generated an exception: {exc}')
    
    buildings_gdf["lScore_sc"] = scaling_columnDF(buildings_gdf["lScore"])
    
    return buildings_gdf