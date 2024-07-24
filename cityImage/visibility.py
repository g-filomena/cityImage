import pandas as pd
import numpy as np
import geopandas as gpd
import pyvista as pv
from tqdm import tqdm

from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString
from shapely.ops import unary_union
pd.set_option("display.precision", 3)

import concurrent.futures
from .utilities import polygon_2d_to_3d 
from .angles import get_coord_angle

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
    
def compute_3d_sight_lines(nodes_gdf, buildings_gdf, distance_along = 200, distance_min_observer_target = 300, chunk_size = 1000):
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
    chunk_size: int
        It Computes sight lines in chunks.
    
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

    def add_height_to_line(exterior, base, height):
        # create a new list of Point objects with the z-coordinate set to (height - base)
        coords = exterior.coords
        new_coords = ((x, y, height - base) for x, y in coords)
        return LineString(new_coords)

    def process_geometry_to_linestring(geometry, base, height):
        if isinstance(geometry, Polygon):
            return add_height_to_line(geometry.exterior, base, height)
        elif isinstance(geometry, MultiPolygon):
            return LineString((point for poly in geometry.geoms for point in add_height_to_line(poly.exterior, base, height).coords))

    # create a Series with the height-extended LineString objects
    building_exteriors_roof = buildings_gdf.apply(lambda row: process_geometry_to_linestring(row.geometry, row.base, row.height), axis=1)
    # create a list with the number of intervals along the exterior LineString to divide the line into
    num_intervals = [max(1, int(exterior.length / distance_along)) for exterior in building_exteriors_roof]
    
    # create a new column with the list of targets along the exterior LineString
    def interpolate_targets(exterior, num_intervals):
        return (exterior.interpolate(min(exterior.length / 2, distance_along) * i) for i in range(num_intervals))
    buildings_gdf['target'] = [list(interpolate_targets(exterior, intervals)) for exterior, intervals in zip(building_exteriors_roof, num_intervals)]

    # create a new column in the nodes GeoDataFrame with the Point objects representing the observer positions
    nodes_gdf['observer'] = nodes_gdf.apply(lambda row: Point(row.geometry.x, row.geometry.y, row.z), axis=1)
    # create a temporary dataframe with building targets
    tmp_buildings = buildings_gdf.explode('target')
    tmp_nodes = nodes_gdf[['nodeID', 'observer']].copy()
    tmp_buildings = tmp_buildings[['buildingID', 'target']]

    # create pyvista 3d objects for buildings
    buildings_gdf['building_3d'] = [polygon_2d_to_3d(geo, base, height) for geo, base, height in zip(buildings_gdf['geometry'], buildings_gdf['base'], buildings_gdf['height'])]
    buildings_sindex = buildings_gdf.sindex

    sight_lines_chunks = []
    tmp_buildings_chunks = np.array_split(tmp_buildings, max(1, len(tmp_buildings) // chunk_size))
    
    for buildings_chunk in tqdm(tmp_buildings_chunks, desc="Processing buildings chunks"):
        # obtain the sight_lines dataframe by means of a cartesian product between GeoDataFrames
        sight_lines_chunk = pd.merge(tmp_nodes.assign(key=1), buildings_chunk.assign(key=1), on='key').drop('key', axis=1)
        # calculate the distance between observer and target and filter
        sight_lines_chunk['distance']= [p1.distance(p2) for p1, p2 in zip(sight_lines_chunk.observer, sight_lines_chunk.target)]
        sight_lines_chunk = sight_lines_chunk[sight_lines_chunk['distance'] >= distance_min_observer_target]
        # add geometry column with LineString connecting observer and target
        sight_lines_chunk['geometry'] = [LineString([p1, p2]) for p1, p2 in zip(sight_lines_chunk.observer, sight_lines_chunk.target)]

        # extract tuples for starting and target points of the sight lines
        sight_lines_chunk['start'] = [[observer.x, observer.y, observer.z] for observer in sight_lines_chunk.observer]
        sight_lines_chunk['stop'] =[[target.x, target.y, target.z] for target in sight_lines_chunk.target]

        # check intervisibility and filter out sight_lines that are obstructed
        sight_lines_chunk['visible'] = sight_lines_chunk.apply(lambda row: intervisibility(row.geometry, row.buildingID, row.start, row.stop, buildings_gdf, buildings_sindex), axis =1)
        sight_lines_chunk = sight_lines_chunk[sight_lines_chunk['visible'] == True]
        sight_lines_chunk.drop(['start', 'stop', 'observer', 'target', 'visible'], axis=1, inplace=True)
        sight_lines_chunks.append(sight_lines_chunk)
    
    sight_lines = pd.concat(sight_lines_chunks, ignore_index=True)
    sight_lines.reset_index(drop=True, inplace=True)
    sight_lines = sight_lines.set_geometry('geometry').set_crs(buildings_gdf.crs)

    return sight_lines
   
def intervisibility(line2d, buildingID, start, stop, obstructions_gdf, obstructions_sindex):
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
    possible_matches_index = list(obstructions_sindex.intersection(line2d.buffer(5).bounds)) # looking for possible candidates in the external GDF
    possible_matches = obstructions_gdf.iloc[possible_matches_index]
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