import osmnx as ox
import pandas as pd
import numpy as np
import geopandas as gpd

from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString
from shapely.ops import cascaded_union, linemerge, polygonize, polygonize_full, unary_union, nearest_points
from .utilities import gdf_from_geometries
pd.set_option("display.precision", 3)

def road_barriers(place, download_method, distance = 500.0, epsg = None, include_primary = False, include_secondary = False):
    """
    The function downloads major roads from OSM. These can be considered to be barrier to pedestrian movement or, at least, to structure people's cognitive Image of 
    the City. If 'include_primary' is true, the function considers also primary roads, beyond motorway and trunk roads.
    if 'include_secondary' is true, the function considers also secondary roads, beyond motorway and trunk roads.
    The user should assess based on the study area categorisation whether primary and secondary roads may indeed constitute barriers to pedestrian movement.
        
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
    distance: float
        Used when download_method == "distance from address" or == "distance from point".
    epsg: int
        Epsg of the area considered; if None OSMNx is used for the projection.
    include_primary: boolean
        When true, it includes primary roads as barriers.
    include_secondary: boolean
        When true, it includes primary roads as barriers.   
    
    Returns
    -------
    road_barriers: LineString GeoDataFrame
        The road barriers.
    """
    
    crs = 'EPSG:' + str(epsg)
    to_keep = ['trunk', 'motorway']
    if include_primary:
        to_keep.append('primary')
        if include_primary:
            to_keep.append('secondary')

    tags = {'highway': True}
    roads = _download_geometries(place, download_method, tags, crs, distance)
    roads = roads[roads.highway.isin(to_keep)]
    # exclude tunnels
    if "tunnel" in roads.columns:
        roads['tunnel'].fillna(0, inplace=True)
        roads = roads[roads.tunnel == 0]
        
    roads = roads.unary_union
    roads = _simplify_barrier(roads)
    df = pd.DataFrame({'geometry': roads, 'barrier_type': ['road'] * len(roads)})
    road_barriers = gpd.GeoDataFrame(df, geometry = df['geometry'], crs = crs)
       
    return road_barriers
    
def water_barriers(place, download_method, distance = 500.0, lakes_area = 1000, epsg = None):
    """
    The function downloads water bodies from OSM. Lakes, rivers and see coastlines can be considered structuring barriers.
        
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
    distance: float
        Used when download_method == "distance from address" or == "distance from point".
    lakes_area = float
        Minimum area for lakes to be considered.    
    epsg: int
        Epsg of the area considered; if None OSMNx is used for the projection.
        
    Returns
    -------
    water_barriers: LineString GeoDataFrame
        The water barriers GeoDataFrame.
    """
    
    crs = 'EPSG:' + str(epsg)
    tags =  {"waterway": True, "natural": "water", "natural":"coastline"}
    
    # rivers
    tags = {"waterway":True}  
    rivers = _download_geometries(place, download_method, tags, crs, distance)
    rivers = rivers[(rivers.waterway == 'river') | (rivers.waterway == 'canal')]
    rivers = rivers.unary_union
    rivers = _simplify_barrier(rivers)
    rivers = gdf_from_geometries(rivers, crs)
    
    # lakes   
    tags = {"natural":"water"}
    lakes = _download_geometries(place, download_method, tags, crs, distance)  
    to_remove = ['river', 'stream', 'canal', 'riverbank', 'reflecting_pool', 'reservoir', 'bay']
    lakes = lakes[~lakes.water.isin(to_remove)]
    lakes['area'] = lakes.geometry.area
    lakes = lakes[lakes.area > lakes_area]
    lakes = lakes.unary_union
    lakes = _simplify_barrier(lakes) 
    lakes = gdf_from_geometries(lakes, crs)
    lakes = lakes[lakes['length'] >=500]
    
    # sea   
    tags = {"natural":"coastline"}
    sea = _download_geometries(place, download_method, tags, crs, distance)
    sea = sea.unary_union      
    sea = _simplify_barrier(sea)
    sea = gdf_from_geometries(sea, crs)
    
    water = pd.concat([rivers, lakes, sea])
    water = water.unary_union
    water = _simplify_barrier(water)
        
    df = pd.DataFrame({'geometry': water, 'barrier_type': ['water'] * len(water)})
    water_barriers = gpd.GeoDataFrame(df, geometry = df['geometry'], crs = crs)

    return water_barriers
    
def railway_barriers(place, download_method, distance = 500.0, epsg = None, keep_light_rail = False):
    """
    The function downloads overground railway structures from OSM. Such structures can be considered barriers which shape the Image of the City and obstruct sight and 
    movement.
        
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
    distance: float
        Used when download_method == "distance from address" or == "distance from point".
    epsg: int
        Epsg of the area considered; if None OSMNx is used for the projection.
    keep_light_rail: boolean
        Includes light rail, like tramway.
        
    Returns
    -------
    railway_barriers: LineString GeoDataFrame
        The railway barriers GeoDataFrame.
    """    
    
    crs = 'EPSG:' + str(epsg)
    tags = {"railway":"rail"}
    railways = _download_geometries(place, download_method, tags, crs, distance)
    # removing light_rail, in case
    if not keep_light_rail:
        railways = railways[railways.railway != 'light_rail']
    if "tunnel" in railways.columns:
        railways['tunnel'].fillna(0, inplace=True)
        railways = railways[railways.tunnel == 0]
        
    r = railways.unary_union
    p = polygonize_full(r)
    railways = unary_union(p).buffer(10).boundary # to simpify a bit
    railways = _simplify_barrier(railways)
        
    df = pd.DataFrame({'geometry': railways, 'barrier_type': ['railway'] * len(railways)})
    railway_barriers = gpd.GeoDataFrame(df, geometry = df['geometry'], crs = crs)
    
    return railway_barriers
    
def park_barriers(place, download_method, distance = 500.0, epsg = None, min_area = 100000):
    """
    The function downloads parks areas with a certain extent and converts them to LineString features. Parks may break continuity in the urban structure, 
    besides being attractive areas for pedestrians.
        
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
    distance: float
        Used when download_method == "distance from address" or == "distance from point".
    epsg: int
        Epsg of the area considered; if None OSMNx is used for the projection.
    min_area: double
        Parks with an extension smaller that this parameter are disregarded.
      
    Returns
    -------
    park_barriers: LineString GeoDataFrame
        The park barriers GeoDataFrame.
    """

    crs = 'EPSG:' + str(epsg)
    tags = {"leisure": True}
    parks_poly = _download_geometries(place, download_method, tags, crs, distance)
    
    parks_poly = parks_poly[parks_poly.leisure == 'park']
    parks_poly = parks_poly[~parks_poly['geometry'].is_empty] 
    parks_poly['area'] = parks_poly.geometry.area
    parks_poly = parks_poly[parks_poly.area >= min_area]
 
    pp = parks_poly['geometry'].unary_union  
    pp = polygonize_full(pp)
    parks = unary_union(pp).buffer(10).boundary # to simpify a bit
    parks = _simplify_barrier(parks)

    df = pd.DataFrame({'geometry': parks, 'barrier_type': ['park'] * len(parks)})
    park_barriers = gpd.GeoDataFrame(df, geometry = df['geometry'], crs = crs)
    
    return park_barriers     

def _download_geometries(place, download_method, tags, crs, distance = 500.0):
    """
    The function downloads certain geometries from OSM, by means of OSMNX functions.
    It returns a GeoDataFrame, that could be empty when no geometries are found, with the provided tags.
    
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
    tag: dict 
        The desired OSMN tags.
    crs: str
        The coordinate system of the case study area.
    distance: float
        Used when download_method == "distance from address" or == "distance from point".
        
    Returns
    -------
    geometries_gdf: GeoDataFrame
        The resulting GeoDataFrame.
    """    
    download_options = {"distance_from_address", "distance_from_point", "OSMpolygon", "OSMplace"}
    if download_method not in download_options:
        raise downloadError('Provide a download method amongst {}'.format(download_options))
    
    download_method_dict = {
        'distance_from_address': ox.geometries_from_address,
        'distance_from_point': ox.geometries_from_point
        'OSMplace': ox.geometries_from_place,
        'polygon': ox.geometries_from_polygon
    }
    
    download_func = download_method_dict.get(download_method)
    if download_func:
        if download_method in ['distance_from_address', 'distance_from_point']
            geometries_gdf = download_func(place, tags = tags, dist = distance)
        else:
            geometries_gdf = download_func(place, tags = tags)

    geometries_gdf = geometries_gdf.to_crs(crs)
    return geometries_gdf
      
def along_water(edges_gdf, barriers_gdf):
    """
    The function assigns to each street segment in a GeoDataFrame the list of barrierIDs corresponding to waterbodies which lay along the street segment. 
    No obstructions between the street segment and the barriers are admitted.
        
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        The street segmentes GeoDataFrame .
    barriers_gdf: LineString GeoDataFrame
        The barriers GeoDataFrame.
        
    Returns
    -------
    edges_gdf: LineString GeoDataFrame
        The updated street segments GeoDataFrame.
    """
    
    sindex = edges_gdf.sindex
    tmp = barriers_gdf[barriers_gdf['barrier_type'].isin(['water'])]
    edges_gdf['ac_rivers'] = edges_gdf.apply(lambda row: barriers_along(row['edgeID'], edges_gdf, tmp, sindex, offset = 200), axis = 1)
    edges_gdf['c_rivers'] = edges_gdf.apply(lambda row: _crossing_barriers(row['geometry'], tmp), axis = 1)
    edges_gdf['bridge'] = edges_gdf.apply(lambda row: True if len(row['c_rivers']) > 0 else False, axis = 1)
    # excluding bridges
    edges_gdf['a_rivers'] = edges_gdf.apply(lambda row: list(set(row['ac_rivers'])-set(row['c_rivers'])), axis = 1)
    edges_gdf['a_rivers'] = edges_gdf.apply(lambda row: row['ac_rivers'] if not row['bridge'] else [], axis = 1)
    edges_gdf.drop(['ac_rivers', 'c_rivers'], axis = 1, inplace = True)
    
    return edges_gdf

def along_within_parks(edges_gdf, barriers_gdf):
    """
    The function assigns to each street segment in a GeoDataFrame the list of barrierIDs corresponding to parks which lie along the street segments.
    Also street segments within parks are considered.
    
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        The street segmentes GeoDataFrame .
    barriers_gdf: LineString GeoDataFrame
        The barriers GeoDataFrame.
        
    Returns
    -------
    edges_gdf: LineString GeoDataFrame
        The updated street segments GeoDataFrame.
    """
    
    sindex = edges_gdf.sindex
    tmp = barriers_gdf[barriers_gdf['barrier_type']=='park']
    
    # polygonize parks
    park_polygons = barriers_gdf[barriers_gdf['barrier_type']=='park'].copy()
    park_polygons['geometry'] = park_polygons.apply(lambda row: (polygonize_full(row['geometry']))[0], axis = 1)
    park_polygons = gpd.GeoDataFrame(park_polygons['barrierID'], geometry = park_polygons['geometry'], crs = edges_gdf.crs)
    
    edges_gdf['w_parks'] = edges_gdf.apply(lambda row: _within_parks(row['geometry'], park_polygons), axis = 1) #within

    return edges_gdf
    
def barriers_along(ix_line, edges_gdf, barriers_gdf, edges_gdf_sindex, offset = 100):
    """
    The function returns list of barrierIDs along the edgeID of a street segment, given a certain offset.
    Touching and intersecting barriers are ignored.
        
    Parameters
    ----------
    ix_line: int
        Index street segment
    edges_gdf: LineString GeoDataFrame
        The street segmentes GeoDataFrame.
    barriers_gdf: LineString GeoDataFrame
        The barriers GeoDataFrame.
    edges_gdf_sindex: Spatial Index
        Spatial index on edges_gdf.
    offset: int
        Offset along the street segment considered.
      
    Returns
    -------
    barriers_along: List
        A list of barriers along a given street segment.
    """
    
    buffer = edges_gdf.loc[ix_line].geometry.buffer(offset)
    intersecting_barriers = barriers_gdf[barriers_gdf.geometry.intersects(buffer) & ~barriers_gdf.geometry.touches(edges_gdf.loc[ix_line].geometry)]
    if intersecting_barriers.empty:
        return []
    possible_matches = edges_gdf.iloc[list(edges_gdf_sindex.intersection(buffer.bounds))].drop(ix_line)
    barriers_along = []
    for _, barrier in intersecting_barriers.iterrows():
        midpoint = edges_gdf.loc[ix_line].geometry.interpolate(0.5, normalized = True)
        line = LineString([midpoint, nearest_points(midpoint, barrier['geometry'])[1]])
        if not possible_matches[possible_matches.geometry.intersects(line)].empty:
            continue
        barriers_along.append(barrier['barrierID'])
    
    return barriers_along
        
def _within_parks(line_geometry, park_polygons):
    """
    The function supports the along_within_parks function. Returns a list containing the barrierID of possibly intersecting parks (should be one).
        
    Parameters
    ----------
    line_geometry: LineString 
        Street segment geometry.
    park_polygons: Polygon GeoDataFrame
        Parks GeoDataFrame.
      
    Returns
    -------
    within: List
        A list of street segments within a given park's polygon.
    """  
        
    park_sindex = park_polygons.sindex
    possible_matches_index = list(park_sindex.intersection(line_geometry.bounds))
    possible_matches = park_polygons.iloc[possible_matches_index]
    intersecting_parks = possible_matches[possible_matches.geometry.intersects(line_geometry)]
    touching_parks = possible_matches[possible_matches.geometry.touches(line_geometry)]
    if len(intersecting_parks) == 0: 
        return []
    intersecting_parks = intersecting_parks[~intersecting_parks.barrierID.isin(list(touching_parks.barrierID))]
    within = list(intersecting_parks.barrierID)
   
    return within
    
def assign_structuring_barriers(edges_gdf, barriers_gdf):
    """
    The function return a GeoDataFrame with an added boolean column field that indicates whether the street segment intersects a separating/structuring barrier.
    
    Parameters
    ----------
    edges_gdf: LineString GeoDataFrame
        The street segmentes GeoDataFrame.
    barriers_gdf: LineString GeoDataFrame
        The barriers GeoDataFrame.
        
    Returns
    -------
    edges_gdf: LineString GeoDataFrame
        The updated street segments GeoDataFrame.
    """
    
    barriers_gdf = barriers_gdf.copy()
    edges_gdf = edges_gdf.copy()
    exlcude = ['secondary_road', 'park'] # parks are disregarded
    tmp = barriers_gdf[~barriers_gdf['barrier_type'].isin(exlcude)].copy() 
    
    edges_gdf['c_barr'] = edges_gdf.apply(lambda row: _crossing_barriers(row['geometry'], tmp ), axis = 1)
    edges_gdf['sep_barr'] = edges_gdf.apply(lambda row: True if len(row['c_barr']) > 0 else False, axis = 1)
    edges_gdf.drop('c_barr', axis = 1, inplace = True)
    
    return edges_gdf

def _crossing_barriers(line_geometry, barriers_gdf):
    """
    The function supports the assign_structuring_barriers and along_water functions. It returns a list of intersecting barrierIDs.
    
    Parameters
    ----------
    line_geometry: LineString 
        Street segment geometry.
    barriers_gdf: LineString GeoDataFrame
        The barriers GeoDataFrame.
        
    Returns
    -------
    adjacent_barriers: List
        A list of adjacent barriers to a given street segment.
    """
    
    adjacent_barriers = []
    intersecting_barriers = barriers_gdf[barriers_gdf.geometry.intersects(line_geometry)]
    touching_barriers = barriers_gdf[barriers_gdf.geometry.touches(line_geometry)]
    if len(intersecting_barriers) == 0: 
        return adjacent_barriers
    intersecting_barriers = intersecting_barriers[~intersecting_barriers.barrierID.isin(list(touching_barriers.barrierID))]
    adjacent_barriers = list(intersecting_barriers.barrierID)
    return adjacent_barriers
    
def get_barriers(place, download_method, distance = 500.0, epsg = None, parks_min_area = 100000): 
    """
    The function returns all the barriers (water, park, railways, major roads) within a certain urban area.
    Certain parameter are set by default. For manipulating, use the barrier-type specific functions (see above).
    
    Parameters
    ----------
    place: str, tuple
        Name of cities or areas in OSM: 
        - when using "distance_from_point" please provide a (lat, lon) tuple to create the bounding box around it; 
        - when using "distance_from_address" provide an existing OSM address; 
        - when using "OSMplace" provide an OSM place name. The query must be geocodable and OSM must have polygon boundaries for the geocode result.  
        - when using "polygon" please provide the name of a relation in OSM as an argument of place;
    download_method: str, {"distance_from_address", "distance_from_point", "OSMplace", "polygon"}
        It indicates the method that should be used for downloading the data.
    distance: float
        Used when download_method == "distance from address" or == "distance from point".
    epsg: int
        Epsg of the area considered; if None OSMNx is used for the projection.
        
    Returns
    -------
    barriers_gdf: LineString GeoDataFrame
        The barriers GeoDataFrame.
    """
    
    rb = road_barriers(place, download_method, distance, epsg = epsg, include_primary = True)
    wb = water_barriers(place, download_method, distance, epsg = epsg)
    ryb = railway_barriers(place,download_method, distance, epsg = epsg)
    pb = park_barriers(place,download_method, distance, epsg = epsg, min_area = parks_min_area)
    barriers_gdf = pd.concat([rb, wb, ryb, pb])
    barriers_gdf.reset_index(inplace = True, drop = True)
    barriers_gdf['barrierID'] = barriers_gdf.index.astype(int)

    return barriers_gdf
   
   
def _simplify_barrier(geometries):
    """
    The function merges a list of geometries in a single geometry when possible; in any case it returns the resulting features within a list. 
    
    Parameters
    ----------
    geometry: Shapely Geometry
        The geommetric representation of a barrier.
        
    Returns
    -------
    features: list of LineString
        The list of actual geometries.
    """
  
    if type(geometries) is Polygon: 
        features = [geometries.boundary]
    elif type(geometries) is LineString: 
        features = [geometries]
    elif type(geometries) is MultiLineString:
        features = list(geometries.geoms)
    elif type(geometries) is MultiPolygon:
        features = list(geometries.boundary.geoms)
    else:
        return []
    
    return features
        
class Error(Exception):
    """Base class for other exceptions"""

class downloadError(Error):
    """Raised when a wrong download method is provided"""