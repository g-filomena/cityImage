import osmnx as ox
import pandas as pd
import numpy as np
import geopandas as gpd
import pyvista as pv

from shapely.geometry import Point, LineString, Polygon, MultiPolygon, mapping, MultiLineString
from shapely.ops import cascaded_union, linemerge
from scipy.sparse import linalg
pd.set_option("display.precision", 3)

from .utilities import scaling_columnDF
from .angles import get_coord_angle

"""
This set of functions is designed for extracting the computational Image of The City.
Computational landmarks can be extracted employing the following functions.
"""
def downloadError(Exception):
    pass
 
def get_buildings_fromSHP(path: str, epsg: int, case_study_area = None, distance_from_center: float = 1000, height_field: str = None, base_field: 
    str = None, land_use_field: str = None) -> gpd.GeoDataFrame:
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
    
def get_buildings_fromOSM(place, download_method: str, epsg = None, distance = 1000) -> gpd.GeoDataFrame:
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

def simplify_footprints(buildings_gdf: gpd.GeoDataFrame, crs: int) -> gpd.GeoDataFrame:
    """    
    The function simplifies the building footprint geometries and creates a new GeoDataFrame for the area of interest.
    The function extracts the individual parts of the multi-part geometries and creates a new GeoDataFrame with single part geometries.
    It also assigns the original CRS to the new GeoDataFrame.
            
    Parameters
    ----------
    buildings_gdf: gpd.GeoDataFrame
        GeoDataFrame containing building footprint geometries
    crs: int
        CRS of the input GeoDataFrame
    
    Returns
    -------
    single_parts_gdf: Polygon GeoDataFrame
        the new GeoDataFrame with simplified single part building footprint geometries
    """  
    
    buildings_gdf = buildings_gdf.copy()
    single_parts = gpd.geoseries.GeoSeries([geom for geom in buildings_gdf.unary_union.geoms])
    single_parts_gdf = gpd.GeoDataFrame(geometry=single_parts, crs = crs)
    
    return single_parts_gdf

def attach_attributes(buildings_gdf: gpd.GeoDataFrame, attributes_gdf: gpd.GeoDataFrame, height_field: str, base_field: str, land_use_field: str) -> gpd.GeoDataFrame:
    """
    Attach attributes to buildings GeoDataFrame by intersecting it with another GeoDataFrame.
    
    Parameters
    ----------
    buildings_gdf : Polygon GeoDataFrame
        GeoDataFrame containing building footprint geometries
    attributes_gdf : Polygon GeoDataFrame
        GeoDataFrame containing attributes to be attached to the buildings
    height_field : string
        Column name of the height attribute in the attributes_gdf
    base_field : string
        Column name of the base attribute in the attributes_gdf
    land_use_field : string
        Column name of the land use attribute in the attributes_gdf
        
    Returns
    -------
    new_buildings_gdf : Polygon GeoDataFrame
        The modified GeoDataFrame containing the building footprints, as well as the heigh
     
    
    buildings_gdf = buildings_gdf.copy()
    attributes_gdf = attributes_gdf[attributes_gdf.geometry.area > 50].copy()
    attributes_gdf[land_use_field] = attributes_gdf[land_use_field].where(pd.notnull(attributes_gdf[land_use_field]), None)
    buildings_gdf = gpd.sjoin(buildings_gdf, attributes_gdf[['geometry', height_field, land_use_field]], how="left", op='intersects')
    """
    
    new_buildings_gdf = buildings_gdf.groupby("buildingID").agg({
                                                                'geometry': 'first',
                                                                height_field: 'max',
                                                                base_field: 'max',
                                                                land_use_field: lambda x: x.value_counts().idxmax()
                                                                }).reset_index()
                                                                
    new_buildings_gdf.rename(columns={height_field: "height", base_field: "base", land_use_field: "land_use_raw"}, inplace=True)
    new_buildings_gdf['area'] = new_buildings_gdf.geometry.area
    new_buildings_gdf.drop([height_field, base_field, land_use_field], axis=1, inplace=True)
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
    obstructions_gdf = buildings_gdf if obstructions_gdf is None else obstructions_gdf
    sindex = obstructions_gdf.sindex
    street_network = edges_gdf.geometry.unary_union

    buildings_gdf["road"] = buildings_gdf.geometry.distance(street_network)
    buildings_gdf["2dvis"] = buildings_gdf.geometry.apply(lambda row: _advance_visibility(row, obstructions_gdf, sindex, max_expansion_distance=
                                    max_expansion_distance, distance_along=distance_along))
    buildings_gdf["neigh"] = buildings_gdf.geometry.apply(lambda row: _number_neighbours(row, obstructions_gdf, sindex, radius=radius))

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
        ob = cascaded_union(obstacles.geometry)
        lines = [line.intersection(ob) for line in lines]
    lines = [LineString([origin, Point(x.coords[0])]) if type(x) == LineString else LineString([origin, Point(x[0].coords[0])]) for x in lines]

    # creating a polygon of visibility based on the lines and their progression, taking into account the origin Point too    
    poly = Polygon([[p.x, p.y] for p in [origin] + lines + [origin]])
    poly_vis = poly.difference(building_geometry)
    if poly_vis.is_empty:
        poly_vis = poly.buffer(0).difference(building_geometry) 
    
    return poly_vis.area

def visibility_graph(building_geometry, obstructions_gdf):
    # Create a visibility graph from the obstructions
    visibility_graph = vg.VisibilityGraph()
    visibility_graph.from_polygons(list(obstructions_gdf['geometry'].values))
    
    # Get the visibility polygon for the building
    visible_area = visibility_graph.visibility_polygon(building_geometry.centroid.coords[0])
    
    # Return the visible area
    return visible_area    
    
    

def compute_3d_sight_lines(nodes_gdf: gpd.GeoDataFrame, buildings_gdf: gpd.GeoDataFrame, distance_along: float, distance_min_observer_target: float ) -> gpd.GeoDataFrame:
    """
    Computes the 3D sight lines between observer nodes and target buildings, based on a given distance along the line of sight and a minimum distance between observer and target.
    
    Parameters
    ----------
    nodes_gdf (gpd.GeoDataFrame): A GeoDataFrame containing the observer nodes, with at least a Point geometry column
    buildings_gdf (gpd.GeoDataFrame): A GeoDataFrame containing the target buildings, with at least a Polygon geometry column
    distance_along (float): The distance along the line of sight to extend the sight lines
    distance_min_observer_target (float): The minimum distance between observer and target, to avoid self-intersections
    
    Returns:
    ----------
    gpd.GeoDataFrame: A GeoDataFrame containing the computed sight lines, with at least the columns 'observer' and 'target'
    
    """
    
    nodes_gdf = nodes_gdf.copy()
    buildings_gdf = buildings_gdf.copy()
    
    if 'base' not in buildings_gdf.columns:
        buildings_gdf["base"] = 0.0
    
    def add_height(exterior, base, height):
        coords = exterior.coords
        new_coords = [Point(x, y, height-base) for x, y in coords]
        return LineString(new_coords)

    building_exteriors = buildings_gdf.apply(lambda row: add_height(row.geometry.exterior, row.base, row.height), axis=1)
    num_intervals = [max(1, int(exterior.length / distance_along)) for exterior in building_exteriors]
    buildings_gdf['target'] = pd.Series([([(exterior.interpolate(min(exterior.length/2, distance_along)*i)) for i in range(x)]) 
              for exterior, x in zip(building_exteriors, num_intervals)], index=buildings_gdf.index)
    
    nodes_gdf['observer'] = nodes_gdf.apply(lambda row: Point(row.geometry.x, row.geometry.y, row.height), axis=1)
    tmp_buildings = buildings_gdf.explode('target')
    tmp_nodes = nodes_gdf[['nodeID', 'observer']].copy()
    tmp_buildings = tmp_buildings[['buildingID', 'target']].copy()
    
    sight_lines = pd.merge(tmp_nodes.assign(key=1), tmp_buildings.assign(key=1), on='key').drop('key', axis=1)
    sight_lines['distance']= [p1.distance(p2) for p1, p2 in zip(sight_lines.observer, sight_lines.target)]
    sight_lines = sight_lines[sight_lines['distance'] >= distance_min_observer_target]
    sight_lines['geometry'] = [LineString([p1, p2]) for p1, p2 in zip(sight_lines.observer, sight_lines.target)]
    sight_lines.reset_index(drop = True, inplace = True)

    buildings_gdf['building_3d'] = [polygon_2d_to_3d(geo, base, height) for geo,base, height in zip(buildings_gdf['geometry'], buildings_gdf['base'], buildings_gdf['height'])]
    sight_lines['start'] = [[observer.x, observer.y, observer.z] for observer in sight_lines.observer]
    sight_lines['stop'] =[[target.x, target.y, target.z] for target in sight_lines.target]
    buildings_sindex = buildings_gdf.sindex
    
    sight_lines['visible'] = sight_lines.apply(lambda row: intervisibility(row.geometry, row.buildingID, row.start, row.stop, buildings_gdf, buildings_sindex), axis =1)
    sight_lines = sight_lines[sight_lines['visibile'] == True]
    sight_lines.drop(['start', 'stop', 'observer', 'target', 'visible'], axis = 1, inplace = True)
    sight_lines = sight_lines.set_geometry('geometry)').set_crs(buildings_gdf.crs)
    
    return sight_lines
   
def intervisibility(line2d, buildingID, start, stop, buildings_gdf, buildings_sindex) -> bool:
    """
    Check if a line of sight between two points is obstructed by any buildings.
    
    Parameters:
        line2d (shapely.geometry.LineString): The 2D line of sight to check for obstruction.
        buildingID (int): The ID of the building that the line of sight originates from.
        start (Tuple[float, float, float]): The starting point of the line of sight in the form of (x, y, z).
        stop (Tuple[float, float, float]): The ending point of the line of sight in the form of (x, y, z).
    
    Returns:
        bool: True if the line of sight is not obstructed, False otherwise.
    """
     
    # first just check for 2d intersections
    possible_matches_index = list(buildings_sindex.intersection(line2d.buffer(5).bounds)) # looking for possible candidates in the external GDF
    possible_matches = buildings_gdf.loc[possible_matches_index]
    pm = possible_matches[possible_matches.intersects(line2d)]
    pm = pm[pm.buildingID != buildingID]
    if len(pm) == 0:
        return True
    
    # if there are, check for 3d ones
    visible = True
    for _, row_building in pm.iterrows():
        building_3d = row_building.building_3d.extract_surface().triangulate()
        points, intersections = building_3d.ray_trace(start, stop)
        if len(intersections) > 0:
            return False
            
    return visible 
    
def polygon_2d_to_3d(building_polygon, base, height):
               
    poly_points = building_polygon.exterior.coords
        
    def reorient_coords(xy):
        value = 0
        for i in range(len(xy)):
            x1, y1 = xy[i]
            x2, y2 = xy[(i+1)%len(xy)]
            value += (x2-x1)*(y2+y1)
        if value > 0:
            return xy
        else:
            return xy[::-1]

    xy = reorient_coords(poly_points)
    xyz_base = [(x,y,base) for x,y in xy]
    faces = [len(xyz_base), *range(len(xyz_base))]
    polygon = pv.PolyData(xyz_base, faces=faces)
    
    return polygon.extrude((0, 0, height - base), capping=True)
   
   
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
     
    # Create a buffer around the building of the specified radius
    buffer = building_geometry.buffer(radius)
    # Perform a spatial join between the buffer and the buildings_gdf, keeping only the buildings within the buffer
    neighbours = geopandas.sjoin(buildings_gdf, buildings_gdf.loc[buildings_gdf.intersects(buffer)], op='within')
    # Group the neighbours by land use and count the number of occurrences
    neigh_counts = neighbours.groupby(["land_use"])["nr"].count()
    # Compute the pragmatic meaning score
    Nj = neigh_counts.loc[building_land_use]
    Pj = 1-(Nj/neighbours["nr"].count())
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

    if ("height" not in buildings_gdf.columns) or (("height" in buildings_gdf.columns) and (buildings_gdf.height.max() == 0.0)):
        buildings_gdf[['height','3dvis', 'fac']]  = 0.0
        if g_cW['vScore'] != 0.0:
            to_add = g_cW['vScore']/3
            g_cW['sScore'] += to_add
            g_cW['cScore'] += to_add
            g_cW['pScore'] += to_add
            g_cW['vScore'] = 0.0

    for i in col + col_inverse:
        if buildings_gdf[i].max() == 0.0:
            buildings_gdf[i+"_sc"] = 0.0
        else:
            scaling_columnDF(buildings_gdf, i, inverse = i in col_inverse)
  
    # computing scores   
    buildings_gdf["vScore"] = buildings_gdf["fac_sc"]*g_iW["fac"] + buildings_gdf["height_sc"]*g_iW["height"] + buildings_gdf["3dvis_sc"]*g_iW["3dvis"]
    buildings_gdf["sScore"] = buildings_gdf["area_sc"]*g_iW["area"] + buildings_gdf["neigh_sc"]*g_iW["neigh"] + buildings_gdf["2dvis_sc"]*g_iW["2dvis"] + buildings_gdf["road_sc"]*g_iW["road"]
    
    col = ["vScore", "sScore"]
    buildings_gdf = buildings_gdf.assign(**{f'{i}_sc': scaling_columnDF(buildings_gdf, i) if buildings_gdf[i].max() != 0.0 else 0.0 for i in col})
    
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
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_scores = {executor.submit(_building_local_score, row["geometry"], row["buildingID"], buildings_gdf, sindex, l_cW, l_iW, radius): row["buildingID"] 
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
    pm = buildings_gdf.loc[possible_matches_index].copy()
    pm = pm[pm.intersects(buffer)]
    
    # rescaling the values
    pm[col_all + [col+"_sc" for col in col_all]] = (pm[col_all] != 0.0).select(
        lambda x: scaling_columnDF(pm, x, inverse = x in col_inverse), 0.0)
    # recomputing scores
    pm["vScore_l"] = pm["fac_sc"]*l_iW["fac"] + pm["height_sc"]*l_iW["height"] + pm["3dvis"]*l_iW["3dvis"]
    pm["sScore_l"] = pm["area_sc"]*l_iW["area"]+ pm["neigh_sc"]*l_iW["neigh"] + pm["road_sc"]*l_iW["road"] + pm["2dvis_sc"]*l_iW["fac"]
    pm["cScore_l"] = pm["cult_sc"]
    pm["pScore_l"] = pm["prag_sc"]
    pm[["vScore_l", "sScore_l"] + [col+"_sc" for col in ["vScore_l", "sScore_l"]]] = (pm[["vScore_l", "sScore_l"]] != 0.0).select(
        lambda x: scaling_columnDF(pm, x), 0.0)
        
    pm["lScore"] = pm["vScore_l_sc"]*l_cW["vScore"] + pm["sScore_l_sc"]*l_cW["sScore"] + pm["cScore_l"]*l_cW["cScore"] + pm["pScore_l"]*l_cW["pScore"]
    score = float("{0:.3f}".format(pm.loc[buildingID, "lScore"]))
    return score
    
class Error(Exception):
    """Base class for other exceptions"""

class downloadError(Error):
    """Raised when a wrong download method is provided"""