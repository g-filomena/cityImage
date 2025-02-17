import pandas as pd
import numpy as np
import geopandas as gpd
import pyvista as pv
from tqdm import tqdm
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor


from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString
from shapely.ops import unary_union
pd.set_option("display.precision", 3)

from .utilities import polygon_2d_to_3d 
from .angles import get_coord_angle
from .graph import graph_fromGDF
from .clean import consolidate_nodes

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
  
 
def compute_3d_sight_lines(nodes_gdf, buildings_gdf, distance_along = 200, distance_min_observer_target = 300, chunk_size=500, consolidate = 
                        False, edges_gdf = None, tolerance = 0.0):
    """
    Perform visibility check in parallel by processing in manageable chunks.

    Parameters:
        nodes_gdf: GeoDataFrame containing observer nodes.
        buildings_gdf: GeoDataFrame containing building geometries.
        distance_min_observer_target: Minimum observer-target distance.
        chunk_size: Size of chunks to process at a time.

    Returns:
        GeoDataFrame: Updated DataFrame with visibility results.
    """
    tmp_nodes, buildings_gdf = _prepare_3d_sight_lines(nodes_gdf, buildings_gdf, distance_along = 200, consolidate = consolidate, edges_gdf = edges_gdf, tolerance = tolerance)
    
    buildings_sindex = buildings_gdf.sindex
    sight_lines_chunks = []
    # Divide nodes into manageable chunks
    node_chunks = np.array_split(tmp_nodes, max(1, len(tmp_nodes) // chunk_size))
    
    
    def _compute_distances_filter(row, min_distance):
        """
        Compute the distance between observer and target points and check if it's above the minimum distance.
        """
        observer = row.observer
        target = row.target
        distance = observer.distance(target)
        return distance >= min_distance

    for node_chunk in tqdm(node_chunks, desc="Processing node chunks"):
        # Create a temporary cartesian product for the current node chunk
        potential_sight_lines = pd.merge(node_chunk.assign(key=1), buildings_gdf.assign(key=1), on="key").drop("key", axis=1)

        # Compute distances and filter
        potential_sight_lines = potential_sight_lines[
            potential_sight_lines.apply(
                lambda row: _compute_distances_filter(row, distance_min_observer_target),
                axis=1
            )
        ]

        # If filtered potential sight lines are empty, skip processing
        if potential_sight_lines.empty:
            continue

        potential_sight_lines['start'] = [[observer.x, observer.y, observer.z] for observer in potential_sight_lines.observer]
        potential_sight_lines['stop'] =[[target.x, target.y, target.z] for target in potential_sight_lines.target]

        # Perform parallelized intervisibility check
        max_workers = max(1, multiprocessing.cpu_count() // 2)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            tqdm.pandas(desc="Checking visibility (parallelized)")
            potential_sight_lines["visible"] = list(
                executor.map(
                    lambda row: intervisibility(row, buildings_gdf, buildings_sindex),
                    potential_sight_lines.itertuples(index=False)
                )
            )

        # Filter visible sight lines and append to the result
        potential_sight_lines = potential_sight_lines[potential_sight_lines["visible"] == True]
        sight_lines_chunks.append(potential_sight_lines)


    # Combine all chunks into a single GeoDataFrame
    sight_lines_tmp = pd.concat(sight_lines_chunks, ignore_index=True)
    sight_lines_tmp.reset_index(drop=True, inplace=True)
     
    nodes_gdf['geometry'] = [Point(geometry.x, geometry.y, z) for geometry, z in zip(nodes_gdf['geometry'], nodes_gdf['z'])]     
    if consolidate:
        sight_lines_tmp = sight_lines_tmp.explode(column= 'old_nodeIDs', ignore_index=True)
        sight_lines_tmp['nodeID'] = sight_lines_tmp['old_nodeIDs']
     
    sight_lines_tmp = sight_lines_tmp.drop(['geometry_x', 'geometry_y', 'start', 'stop', 'visible', 'base', 
                                    'height', 'building_3d', 'observer', 'z', 'old_nodeIDs'], errors = 'ignore', axis = 1)
    
    # Merge with `nodes` GeoDataFrame to get the geometry of each nodeID
    sight_lines_tmp = sight_lines_tmp.merge(
        nodes_gdf[['nodeID', 'geometry']],
        left_on='nodeID',
        right_on='nodeID',
        suffixes=('', '_node')
    )
    
    # Step 3: Update the `observer` column with the merged geometry
    sight_lines_tmp['geometry'] = [LineString([observer, target]) for observer, target in zip(sight_lines_tmp['geometry'], 
                                                                                          sight_lines_tmp['target'])]
    sight_lines = gpd.GeoDataFrame(sight_lines_tmp, geometry='geometry', crs = nodes_gdf.crs)
    sight_lines.drop(['target'], axis = 1, inplace = True)
    sight_lines['length'] = sight_lines.geometry.length
    sight_lines = sight_lines.sort_values(['buildingID', 'nodeID', 'length'], ascending=[False, False, False]).drop_duplicates(['buildingID', 'nodeID'], keep='first')
    sight_lines.reset_index(inplace = True, drop = True)
    
    return sight_lines 
    
def intervisibility(row, buildings_gdf, buildings_sindex):
    """
    Check if a 3D line of sight between two points is obstructed by any buildings.

    Parameters:
        row: 
            Row of the DataFrame containing observer and target points.
        buildings_gdf: 
            GeoDataFrame of buildings.
        buildings_sindex: 
            Spatial index of buildings GeoDataFrame.

    Returns:
        bool: True if the line is visible, False otherwise.
    """

    line2d = LineString([row.start, row.stop])
    possible_matches_index = list(buildings_sindex.intersection(line2d.buffer(5).bounds))
    possible_matches = buildings_gdf.iloc[possible_matches_index]
    matches = possible_matches[possible_matches.intersects(line2d)]
    matches = matches[matches.buildingID != row.buildingID]
    if len(matches) == 0:
        return True

    for _, row_building in matches.iterrows():
        building_3d = row_building.building_3d.extract_surface().triangulate()
        points, intersections = building_3d.ray_trace(row.start, row.stop)
        if len(intersections) > 0:
            return False
            
    return True
    
def _prepare_3d_sight_lines(nodes_gdf, buildings_gdf, distance_along = 200, consolidate = False, edges_gdf = None, tolerance = 0.0):

    nodes_gdf = nodes_gdf[['geometry', 'x', 'y', 'nodeID', 'z']].copy()
    nodes_gdf['geometry'] = nodes_gdf.apply(lambda row: Point(row['x'], row['y'], row['z']), axis=1)
    
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf['geometry'] = buildings_gdf['geometry'].apply(
        lambda geom: geom if isinstance(geom, Polygon) else geom.convex_hull if geom.is_valid else geom.buffer(0)
    )

    # add a 'base' column to the buildings GeoDataFrame with a default value of 1.0, if not provided
    if 'base' not in buildings_gdf.columns:
        buildings_gdf["base"] = 1.0
    buildings_gdf['base'] = buildings_gdf['base'].where(buildings_gdf['base'] < 1.0, 1.0) # minimum base
    
    buildings_gdf = buildings_gdf[['geometry', 'buildingID', 'base', 'height']].copy()

    def add_height_to_line(exterior, base, height):
        # create a new list of Point objects with the z-coordinate set to (height - base)
        return LineString([(x, y, height + base) for x, y in exterior.coords])

    def process_geometry_to_linestring(geometry, base, height):
        if isinstance(geometry, Polygon):
            return add_height_to_line(geometry.exterior, base, height)
        elif isinstance(geometry, MultiPolygon):
            return LineString([point for poly in geometry.geoms for point in add_height_to_line(poly.exterior, base, height).coords])

    # create a Series with the height-extended LineString objects
    building_exteriors_roof = buildings_gdf.apply(lambda row: process_geometry_to_linestring(row.geometry, row.base, row.height), axis=1)
    # create a list with the number of intervals along the exterior LineString to divide the line into
    num_intervals = [max(1, int(exterior.length / distance_along)) for exterior in building_exteriors_roof]
    
    # create a new column with the list of targets along the exterior LineString
    def interpolate_targets(exterior, num_intervals):
        return [exterior.interpolate(min(exterior.length / 2, distance_along) * i) for i in range(num_intervals)]
    
    buildings_gdf['target'] = [interpolate_targets(exterior, intervals) for exterior, intervals in zip(building_exteriors_roof, num_intervals)]

    if consolidate:
        consolidated_nodes = consolidate_nodes(nodes_gdf, edges_gdf, consolidate_edges_too = False, tolerance = tolerance)
    else:
        consolidated_nodes = nodes_gdf
        consolidated_nodes['geometry'] = [Point(geom.x, geom.y, z) for geom, z in zip(consolidated_nodes.geometry, consolidated_nodes['z'])]

    
    consolidated_nodes['observer'] = consolidated_nodes['geometry']
    # create a temporary dataframe with building targets
    buildings_gdf = buildings_gdf.explode('target')
    consolidated_nodes = consolidated_nodes.drop(["x", "y"], axis = 1)
    buildings_gdf = buildings_gdf[['buildingID', 'target', 'geometry', 'height', 'base']].copy()

    # create pyvista 3d objects for buildings
    buildings_gdf = buildings_gdf.copy()
    buildings_gdf['building_3d'] = [polygon_2d_to_3d(geo, base, height) for geo, base, height in 
                                    zip(buildings_gdf['geometry'], buildings_gdf['base'], buildings_gdf['height'])]

    return consolidated_nodes, buildings_gdf

