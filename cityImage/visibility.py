import os
import gc
import warnings
import psutil
import ast 

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point, LineString, Polygon, MultiPolygon, MultiLineString, MultiPoint
from shapely.ops import unary_union
from shapely.strtree import STRtree

import pyvista as pv
from tqdm import tqdm

# Parallel/Distributed
import dask
from dask import delayed, compute
from numba import njit, prange
from concurrent.futures import ThreadPoolExecutor

# Project modules
from .angles import get_coord_angle
from .graph_consolidate import consolidate_nodes
from .utilities import gdf_multipolygon_to_polygon

pd.set_option("display.precision", 3)

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
    
def compute_3d_sight_lines(nodes_gdf: gpd.GeoDataFrame, target_buildings_gdf: gpd.GeoDataFrame, obstructions_buildings_gdf: gpd.GeoDataFrame, 
                           simplified_target_buildings: gpd.GeoDataFrame, edges_gdf: gpd.GeoDataFrame, city_name: str,
                           distance_along: float = 200, min_observer_target_distance: float = 300, sight_lines_chunk_size: int = 500000, 
                           consolidate: bool = False, consolidate_tolerance: float = 0.0, 
                           with_pyvista = False, num_workers: int = 20):
    """
    Computes 3D sight lines between observer nodes and target buildings, accounting for 2D and 3D obstructions.

    The computation is performed in memory-efficient chunks. For each chunk:
    1. Filters potential sight lines by minimum distance.
    2. Restricts obstruction checks to those near each target.
    3. Checks for 2D (planimetric) obstructions.
    4. Optionally refines visibility with full 3D intersection analysis.
    5. Outputs are merged and saved to disk per chunk, then concatenated and finalized.

    Optionally, PyVista is used for the 3D step, and both observer/target coordinates and obstruction matches are exported.

    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        Observer locations (usually points) with 3D coordinates and unique node IDs.
    target_buildings_gdf : GeoDataFrame
        Target building polygons (must have 3D geometry).
    obstructions_buildings_gdf : GeoDataFrame
        All buildings considered as potential obstructions (must have 3D geometry).
    simplified_target_buildings : GeoDataFrame
        Simplified versions of target buildings (used for finalization and snapping).
    edges_gdf : GeoDataFrame
        Edge geometries for network structure (used for consolidation).
    city_name : str
        Name prefix for exported chunk files.
    distance_along : float, optional
        Sampling step along building edges (default: 200).
    min_observer_target_distance : float, optional
        Minimum allowed observer-to-target distance (default: 300).
    sight_lines_chunk_size : int, optional
        Maximum number of sight lines to process per chunk (default: 500000).
    consolidate : bool, optional
        Whether to spatially consolidate observer nodes and targets (default: False).
    consolidate_tolerance : float, optional
        Tolerance for consolidation (default: 0.0).
    with_pyvista : bool, optional
        If True, performs 3D obstruction checking using PyVista for extra accuracy (default: False).
    num_workers : int, optional
        Number of parallel workers to use for chunked and obstruction operations (default: 20).

    Returns
    -------
    sight_lines : GeoDataFrame
        Finalized GeoDataFrame of visible sight lines (3D LineString), with relevant attributes. If no lines are visible, returns empty GeoDataFrame.

    Notes
    -----
    - The function splits the computation into manageable chunks for memory efficiency.
    - Intermediate chunk files are written to disk and merged at the end.
    - Columns such as 'matchesIDs', 'observer_coords', and 'target_coords' are added temporarily and dropped in the final result.
    - If `with_pyvista` is False, a final 3D obstruction check is performed on all results.
    """
    
    # Step 0: Prepare data
    observers, targets, obstructions_gdf = _prepare_3d_sight_lines(nodes_gdf, target_buildings_gdf, obstructions_buildings_gdf,
                                                               distance_along = distance_along,
                                                               simplified_buildings = simplified_target_buildings,
                                                               consolidate=consolidate,
                                                               consolidate_tolerance=consolidate_tolerance,
                                                               edges_gdf=edges_gdf)
    
    num_observers = len(observers)
    num_targets = len(targets)
    projected_nr_sight_lines = num_observers * num_targets
    
    if projected_nr_sight_lines <= sight_lines_chunk_size:
        observer_chunks = [observers]
    else:
        observers_per_chunk = max(1, sight_lines_chunk_size // num_targets)
        num_chunks = int(np.ceil(num_observers / observers_per_chunk))
        observer_chunks = np.array_split(observers, num_chunks)

    out_prefix="chunk_sight_lines"
    out_files = []
    
    for n, chunk in enumerate(observer_chunks):

        print(f"Starting chunk {n+1} of {len(observer_chunks)}")
        visibles = []

        print(" 01 - Filtering by distance")
        potential_sight_lines = filter_distance(chunk, targets, min_observer_target_distance)
        if potential_sight_lines.empty:
            continue

        print(" 02 - Only checking obstructions around target building")
        potentially_visible = obstructions_around_target(potential_sight_lines, targets, obstructions_gdf, num_workers = num_workers)
        if potentially_visible.empty:
            continue

        print(" 03 - Checking 2d obstructions")
        visible_2d, obstructed_2d = obstructions_2d(potentially_visible, targets, obstructions_gdf, num_workers = num_workers)
        if not visible_2d.empty:
            visibles.append(visible_2d)
        
        print(" 04 - Checking 3d obstructions")
        visible_3d = None
        if not obstructed_2d.empty:
            if with_pyvista:
                visible_3d = obstructions_3d_with_pyvista(obstructed_2d, obstructions_gdf, 'matchesIDs', num_workers=num_workers)
            else:
                visible_3d = obstructions_3d(obstructed_2d, num_workers=num_workers)
            if not visible_3d.empty:
                visibles.append(visible_3d)
        
        # Finalize columns and export
        if visibles:             
            cols_to_drop = ['obstructions_xy','aroundIDs', 'obstructions_xy', 'obstructions_z', 'visible', 'z']
            
            for df in visibles:
                df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True, errors='ignore')
            
            chunk_sight_lines = pd.concat(visibles, ignore_index=True)
            chunk_sight_lines['matchesIDs'] = chunk_sight_lines['matchesIDs'].apply(lambda x: str(x))
            chunk_sight_lines = _finalize_sight_lines(chunk_sight_lines, nodes_gdf, consolidate, simplified_target_buildings)
            chunk_file = f"{city_name}_{out_prefix}_{n}.gpkg"
            chunk_sight_lines.to_file(chunk_file)
            out_files.append(chunk_file)

            del chunk_sight_lines, visibles
            gc.collect()
            print(f"Chunk {n+1} processed and exported: {chunk_file}")

        visibles = []
        del chunk, potentially_visible, potential_sight_lines, visible_2d, obstructed_2d, visible_3d
        gc.collect()

    print("All chunks processed.")
    sight_lines = gpd.GeoDataFrame
    if out_files:
        sight_lines = merge_gpkg_chunks_to_gdf(out_files)
        sight_lines['matchesIDs'] = sight_lines['matchesIDs'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])
        sight_lines['observer_coords'] = [ (line.coords[0][0], line.coords[0][1], line.coords[0][2]) for line in sight_lines['geometry']]
        sight_lines['target_coords'] = [(line.coords[-1][0], line.coords[-1][1], line.coords[-1][2]) for line in sight_lines['geometry']]

        # checks everything left
        if not with_pyvista:
            sight_lines = obstructions_3d_with_pyvista(sight_lines, obstructions_gdf, 'matchesIDs', num_workers)
        sight_lines.drop(['observer_coords', 'target_coords', 'matchesIDs', 'visible'], axis=1, errors = 'ignore', inplace=True)
    
    return sight_lines 

## Preparation ###########################
def _prepare_3d_sight_lines(nodes_gdf, target_buildings_gdf, obstructions_gdf, distance_along = 200, simplified_buildings = gpd.GeoDataFrame, 
        consolidate = False, consolidate_tolerance = 0.0, edges_gdf = None):
    """
    Prepares observer points, target points, and obstruction geometries for 3D sight line analysis.

    This function processes the input node and building GeoDataFrames to:
        - Set up observer points with 3D coordinates,
        - Prepare target building roof points at regular intervals,
        - Optionally apply simplified building outlines and spatial consolidation,
        - Annotate each target with nearby building IDs for efficient obstruction queries,
        - Create 3D representations for all obstruction polygons.

    Parameters
    ----------
    nodes_gdf : GeoDataFrame
        Observer locations with columns ['geometry', 'x', 'y', 'nodeID', 'z'].
    target_buildings_gdf : GeoDataFrame
        Target building polygons with 'height' and 'base' attributes.
    obstructions_gdf : GeoDataFrame
        Building polygons considered as obstructions (must have 'height' and 'base').
    distance_along : float, optional
        Interval in CRS units for sampling target building roofs (default: 200).
    simplified_buildings : GeoDataFrame, optional
        Simplified building outlines, used if not empty (default: empty).
    consolidate : bool, optional
        If True, spatially merges close observer points (default: False).
    consolidate_tolerance : float, optional
        Distance tolerance for observer consolidation (default: 0.0).
    edges_gdf : GeoDataFrame, optional
        Network edges for observer consolidation.

    Returns
    -------
    observer_points_gdf : GeoDataFrame
        GeoDataFrame of observer points with 3D geometry.
    target_points : GeoDataFrame
        Points along target building roofs with associated nearby building IDs.
    obstructions_gdf : GeoDataFrame
        Obstructions as 3D polygons with updated geometry.
    """

    nodes_gdf = nodes_gdf[['geometry', 'x', 'y', 'nodeID', 'z']].copy()
    nodes_gdf['geometry'] = nodes_gdf.apply(lambda row: Point(row['x'], row['y'], row['z']), axis=1)
    
    target_buildings_gdf = _prepare_buildings_gdf(target_buildings_gdf)
    obstructions_gdf = _prepare_buildings_gdf(obstructions_gdf)
    target_buildings_gdf = target_buildings_gdf[target_buildings_gdf.height > 5.0]
    target_points = _prepare_targets(target_buildings_gdf, distance_along)

    if simplified_buildings is not None and not simplified_buildings.empty:
        target_points, obstructions_gdf = _use_simplified_buildings(target_points, obstructions_gdf, simplified_buildings)
  
    if consolidate:
        observer_points_gdf = consolidate_nodes(nodes_gdf, edges_gdf, consolidate_edges_too = False, tolerance = consolidate_tolerance)
    else:
        observer_points_gdf = nodes_gdf.copy()
    
    observer_points_gdf['observer_geo'] = observer_points_gdf['geometry']
    observer_points_gdf = observer_points_gdf.drop(["x", "y", "z"], axis = 1)
    obstructions_gdf['building_3d'] = [polygon_2d_to_3d(geometry, base, height) for geometry, base, height in 
                                       zip(obstructions_gdf['geometry'], obstructions_gdf['base'], obstructions_gdf['height'])]

    def add_nearby_building_ids(target_points, buffer=50):
        tbuff = target_points.copy()
        tbuff['geometry'] = tbuff['target_geo'].buffer(buffer)
        join = gpd.sjoin(tbuff[['buildingID', 'geometry']], obstructions_gdf[['buildingID', 'geometry']],
                         how='left', predicate='intersects')
        join = join[join['buildingID_left'] != join['buildingID_right']]
        nearby = join.groupby('buildingID_left')['buildingID_right'].agg(list).rename('aroundIDs')
        return target_points.join(nearby, on='buildingID')

    target_points = add_nearby_building_ids(target_points, buffer=50)    
    
    # Replace NaN or empty lists in 'aroundIDs' column with an empty list; may include target's buildingID (dealt with later)  
    target_points['aroundIDs'] = target_points['aroundIDs'].apply(lambda x: [] if not isinstance(x, (list, np.ndarray)) else x)
    target_points['aroundIDs'] = target_points['aroundIDs'].apply(lambda x: [i for i in x if pd.notna(i)])
    obstructions_gdf['geometry'] = obstructions_gdf['geometry'].apply(lambda geom: geom.exterior)
    return observer_points_gdf, target_points, obstructions_gdf

def _prepare_buildings_gdf(buildings_gdf):
    """
    Converts MultiPolygons to Polygons, sets minimum base elevation, filters invalid heights, and standardizes columns.

    Parameters
    ----------
    buildings_gdf : GeoDataFrame
        Building footprints with 'geometry', 'buildingID', 'height', and (optionally) 'base' columns.

    Returns
    -------
    buildings_gdf : GeoDataFrame
        Cleaned building GeoDataFrame, indexed by 'buildingID', with non-null height and base >= 1.0.
    """   
    buildings_gdf = gdf_multipolygon_to_polygon(buildings_gdf)
    # add a 'base' column to the buildings GeoDataFrame with a default value of 1.0, if not provided
    if 'base' not in buildings_gdf.columns:
        buildings_gdf["base"] = 1.0
        
    buildings_gdf['base'] = buildings_gdf['base'].where(buildings_gdf['base'] > 1.0, 1.0) # minimum base
    buildings_gdf = buildings_gdf[buildings_gdf["height"].notna() & (buildings_gdf["height"] != None)]
    buildings_gdf = buildings_gdf[['buildingID', 'geometry', 'height', 'base']].copy()
    buildings_gdf.index = buildings_gdf.buildingID
    buildings_gdf.index.name = None
    return buildings_gdf
    
def _prepare_targets(target_buildings_gdf, distance_along):
    """
    Generates 3D target points along the roof perimeter of each building at specified intervals.

    Parameters
    ----------
    target_buildings_gdf : GeoDataFrame
        Target building polygons with 'geometry', 'height', and 'base' columns.
    distance_along : float
        Interval in CRS units for placing points along the roof edge.

    Returns
    -------
    target_points : GeoDataFrame
        Exploded GeoDataFrame with 3D points along each building's roof edge.
    """

    target_buildings = target_buildings_gdf.copy()
    target_buildings['geometry'] = target_buildings_gdf['geometry'].apply(
        lambda geom: geom if isinstance(geom, Polygon) else geom.convex_hull if geom.is_valid else geom.buffer(0)
    )
    
    def extrude_roof(geometry, base, height):
        if isinstance(geometry, Polygon):
            return add_height_to_line(geometry.exterior, base, height)
        elif isinstance(geometry, MultiPolygon):
            return LineString([point for poly in geometry.geoms for point in add_height_to_line(poly.exterior, base, height).coords])
    
    def add_height_to_line(exterior, base, height):
        # create a new list of Point objects with the z-coordinate set to (height - base)
        return LineString([(coord[0], coord[1], height + base) for coord in exterior.coords])
        
    # create the roof
    building_tops = target_buildings.apply(lambda row: extrude_roof(row.geometry, row.base, row.height), axis=1)
    # create a list with the number of intervals along the exterior LineString to divide the line into
    buildings_points = [max(1, int(building_top.length / distance_along)) for building_top in building_tops]
    
    # create a new column with the list of targets along the LineString representing the exterior of the top of the building
    def interpolate_targets(exterior, num_points):
        return [exterior.interpolate(min(exterior.length / 2, distance_along) * i) for i in range(num_points)]
    
    target_buildings['target_geo'] = [interpolate_targets(building_top, num_points) for building_top, num_points in zip(building_tops, buildings_points)]
    target_points = target_buildings.explode('target_geo')   
    return target_points
    
def _use_simplified_buildings(target_points, obstructions_gdf, simplified_buildings):
    """
    Associates each target point and obstruction polygon with a simplified building outline,
    then computes aggregated attributes for each simplified building.

    Parameters
    ----------
    target_points : GeoDataFrame
        3D points representing target locations on building roofs.
    obstructions_gdf : GeoDataFrame
        Building polygons considered as obstructions.
    simplified_buildings : GeoDataFrame
        Simplified building outlines.

    Returns
    -------
    simplified_target_points : GeoDataFrame
        Simplified buildings with aggregated target information and 3D centroid points.
    simplified_obstructions : GeoDataFrame
        Simplified buildings with aggregated obstruction attributes.
    """

    simplified_buildings = gdf_multipolygon_to_polygon(simplified_buildings)
    # Create a spatial index for the simplified buildings
    simplified_tree = STRtree(simplified_buildings.geometry)
    
    def assign_geometry_to_simplified_building(geometry):
        # First, check if the target is within or touches any simplified building using spatial index
        touching_buildings = simplified_tree.query(geometry)
        
        # Check for touch or within condition
        for simplified_building in touching_buildings:
            simplified_building_geo = simplified_buildings.loc[simplified_building].geometry
            if geometry.within(simplified_building_geo) or geometry.touches(simplified_building_geo) or geometry.intersects(simplified_building_geo):
                return simplified_building
        
        return None
    
    # Apply the optimized function to each target
    target_points['simplifiedID'] = target_points['target_geo'].apply(lambda x: assign_geometry_to_simplified_building(x))
    obstructions_gdf['simplifiedID'] = obstructions_gdf['geometry'].apply(lambda x: assign_geometry_to_simplified_building(x))

    # Function to get building IDs, average height, and average base of targets assigned to the same simplified building
    def get_attributes(simplifiedID, gdf):
        # Filter targets assigned to the simplified building
        detailed_buildings = gdf[gdf['simplifiedID'] == simplifiedID].copy()  
        # Extract building IDs of these targets
        buildingIDs = detailed_buildings['buildingID'].tolist() if not detailed_buildings.empty else []
    
        # Check if 'target_geo' exists and extract target geometries
        if 'target_geo' in detailed_buildings.columns:
            target_GEOs = detailed_buildings['target_geo'].tolist() if not detailed_buildings.empty else []
        else:
            target_GEOs = None
    
        # Calculate the average height and base of these targets
        height = detailed_buildings['height'].mean() if not detailed_buildings.empty else np.nan
        base = detailed_buildings['base'].mean() if not detailed_buildings.empty else np.nan
    
        return pd.Series([buildingIDs, target_GEOs, base, height])

    simplified_target_points = simplified_buildings.copy()
    simplified_obstructions = simplified_buildings.copy()
    # Apply the function to each simplified building to calculate building IDs, average height, and base
    simplified_target_points[['buildingIDs', 'target_GEOs', 'base', 'height']] = simplified_buildings.index.to_series().apply(
        get_attributes, gdf=target_points).apply(pd.Series)

    # Apply the function to each simplified building to calculate building IDs, average height, and base
    simplified_obstructions[['buildingIDs', 'target_GEOs', 'base', 'height']] = simplified_buildings.index.to_series().apply(
        get_attributes, gdf=obstructions_gdf).apply(pd.Series)

    simplified_target_points = simplified_target_points[simplified_target_points['buildingIDs'].apply(len) > 0]
    simplified_target_points = simplified_target_points[simplified_target_points['height'].notna()]
    simplified_target_points['buildingID'] = simplified_target_points.index
    
    simplified_obstructions.drop(['buildingIDs', 'target_GEOs'], axis = 1, inplace = True)    
    simplified_obstructions = simplified_obstructions[simplified_obstructions['height'].notna()]
    simplified_obstructions['buildingID'] = simplified_obstructions.index
    
    simplified_target_points['target_geo'] = simplified_target_points.apply(lambda row: Point(row['geometry'].centroid.x, 
                                                                                      row['geometry'].centroid.y, row['height'] + row['base']), axis=1)
    simplified_target_points.drop(['geometry', 'base', 'height', 'FID'], axis = 1, inplace = True, errors = 'ignore')
    return simplified_target_points, simplified_obstructions
    
def merge_gpkg_chunks_to_gdf(out_files):
    """
    Reads multiple GeoPackage chunk files and merges them into a single GeoDataFrame.

    Parameters
    ----------
    out_files : list of str
        List of file paths to the .gpkg files (one per chunk).

    Returns
    -------
    final_gdf : GeoDataFrame
        Combined GeoDataFrame containing all features from the input files.
    """

    # Read each chunk one at a time and append to a list, then concat all together
    gdfs = []
    for f in out_files:
        gdfs.append(gpd.read_file(f))
    final_gdf = pd.concat(gdfs, ignore_index=True)
    return final_gdf
    
## Step 0 ###########################
def filter_distance(chunk, targets, min_observer_target_distance):
    """
    Filters potential observer-target pairs by minimum distance and generates sight line geometries.

    Parameters
    ----------
    chunk : GeoDataFrame
        Observer points (usually a subset/chunk of all observers).
    targets : GeoDataFrame
        Target points to be paired with observers.
    min_observer_target_distance : float
        Minimum allowed 2D distance between observer and target (in CRS units).

    Returns
    -------
    potential_lines : GeoDataFrame
        GeoDataFrame of observer-target pairs meeting the distance threshold,
        with new 'geometry' (LineString) and coordinate columns.
    """

    # Prepare data
    potential_lines = _prepare_chunk_data(chunk, targets)
    # Assign 2D coordinates to new columns directly
    potential_lines['observer_coords'] = [(p.x, p.y) for p in potential_lines['observer_geo']]
    potential_lines['target_coords'] = [(p.x, p.y) for p in potential_lines['target_geo']]
   
    # Extract 2D coordinates and then distance filter
    observer_coords = np.array(potential_lines['observer_coords'].to_list())
    target_coords = np.array(potential_lines['target_coords'].to_list())
       
    # Query pairs with distance >= threshold
    distances = np.linalg.norm(observer_coords - target_coords, axis=1)
    mask = distances >= min_observer_target_distance
    potential_lines = potential_lines.loc[mask]

    # Extract 2D coordinates and then distance filter
    observer_coords = np.array(potential_lines['observer_coords'].to_list())
    target_coords = np.array(potential_lines['target_coords'].to_list())
    line_geometries = [LineString([start, stop]) for start, stop in zip(observer_coords, target_coords)]
    potential_lines['geometry'] = line_geometries

    return potential_lines
    
def _prepare_chunk_data(chunk: gpd.GeoDataFrame, targets: gpd.GeoDataFrame):
    """
    Computes the cartesian product of observers and targets for a chunk, omitting geometry columns for efficiency.

    Parameters
    ----------
    chunk : GeoDataFrame
        Observer points (one chunk of all observers).
    targets : GeoDataFrame
        Target points.

    Returns
    -------
    potential_lines : DataFrame
        DataFrame of all observer-target pairs (cartesian product) with all non-geometry attributes.
    """

    # Drop geometry column early to reduce memory usage
    chunk_data = chunk.drop('geometry', axis=1).assign(key=1)
    targets_data = targets.assign(key=1)
    
    # Compute cartesian product efficiently
    potential_lines = pd.merge(chunk_data, targets_data, on="key").drop("key", axis=1)
    print("num potential line", len(potential_lines), "num lines in chunk ", len(chunk_data), "num target", len(targets_data))
    return potential_lines

## Step 1 ###########################    
def obstructions_around_target(potential_sight_lines, targets, obstructions_gdf, num_workers=20):
    """
    Identifies potential sight lines that may be obstructed by nearby buildings using parallel 2D checks.

    For each batch of potential sight lines, checks for intersections with nearby (aroundIDs) obstructions.
    Performs a follow-up 3D obstruction check for any initially obstructed lines.

    Parameters
    ----------
    potential_sight_lines : GeoDataFrame
        Candidate sight lines with 'aroundIDs' referencing nearby obstructions.
    targets : GeoDataFrame
        Target points (not used directly, included for interface consistency).
    obstructions_gdf : GeoDataFrame
        Building polygons used as obstructions.
    num_workers : int, optional
        Number of parallel threads for chunked 2D obstruction checks (default: 20).

    Returns
    -------
    potentially_visible : GeoDataFrame
        Filtered sight lines that are not obstructed in 2D/3D or are cleared after a 3D check.
    """

    # Create LineString geometries directly using vectorized coordinates
    sub_chunks = _define_batches(potential_sight_lines)
    
    tasks = [_find_obstructions_2d(sub_chunk, obstructions_gdf, 'aroundIDs') for sub_chunk in sub_chunks]
    results = dask.compute(*tasks, scheduler="threads", num_workers = num_workers)
    potentially_visibles = [r[0] for r in results]
    obstructeds = [r[1] for r in results]
    potentially_visible = pd.concat(potentially_visibles, ignore_index=True)
    obstructed = pd.concat(obstructeds, ignore_index=True)
    
    if not obstructed.empty:
        tmp = obstructions_3d(obstructed, num_workers = num_workers)
        if not tmp.empty:
            potentially_visible = pd.concat([potentially_visible, tmp], ignore_index=True)
    
    return potentially_visible

def _define_batches(potential_sight_lines, min_size=50, max_size=2000, mem_fraction=0.1, row_weight_kb=10_000):
    """
    Splits a GeoDataFrame of sight lines into memory-aware batches for efficient parallel processing.

    Batch size is based on available system memory and estimated row weight.

    Parameters
    ----------
    potential_sight_lines : GeoDataFrame
        Data to be divided into batches.
    min_size : int, optional
        Minimum number of rows per batch (default: 50).
    max_size : int, optional
        Maximum number of rows per batch (default: 2000).
    mem_fraction : float, optional
        Fraction of available system memory to use for each batch (default: 0.1).
    row_weight_kb : int, optional
        Estimated memory usage per row, in kilobytes (default: 10,000).

    Yields
    ------
    GeoDataFrame
        The next batch of rows.
    """

    available_mem = psutil.virtual_memory().available
    rows_per_batch = int(available_mem * mem_fraction / row_weight_kb)
    rows_per_batch = max(min_size, min(rows_per_batch, max_size))
    n = len(potential_sight_lines)
    for start in range(0, n, rows_per_batch):
        yield potential_sight_lines.iloc[start: start + rows_per_batch]
        
## Step 2 ###########################  
def obstructions_2d(potential_sight_lines, targets, obstructions_gdf, num_workers=20):
    """
    Identifies 2D (planimetric) obstructions between each observer-target sight line and surrounding buildings.

    Uses spatial batch processing and parallel computation to:
      1. Find which obstructions could intersect each sight line (via spatial index).
      2. Check actual intersections and compute obstruction locations for each line.
      3. Split results into visible and potentially obstructed sight lines.

    Parameters
    ----------
    potential_sight_lines : GeoDataFrame
        Candidate sight lines (with geometry) between observers and targets.
    targets : GeoDataFrame
        Target points (not used directly in this function).
    obstructions_gdf : GeoDataFrame
        Building polygons (must include 'buildingID', 'geometry', 'height', and 'base').
    num_workers : int, optional
        Number of parallel threads for batch computation (default: 20).

    Returns
    -------
    visible : GeoDataFrame
        Sight lines with no 2D obstructions.
    potentially_obstructed : GeoDataFrame
        Sight lines that intersect at least one obstruction in 2D.
    """

    sub_chunks = _define_batches(potential_sight_lines)
    tasks = [_find_potential_obstructions_2d(sub_chunk, obstructions_gdf) for sub_chunk in sub_chunks]
    results = dask.compute(*tasks, scheduler="threads", num_workers = num_workers)
    potential_sight_lines = pd.concat(results, ignore_index=True) 
    
    sub_chunks = _define_batches(potential_sight_lines)
    tasks = [_find_obstructions_2d(sub_chunk, obstructions_gdf, 'matchesIDs') for sub_chunk in sub_chunks]
    results = dask.compute(*tasks, scheduler="threads", num_workers= num_workers)
    visibles = [r[0] for r in results]
    potentially_obstructeds = [r[1] for r in results]
    visible = pd.concat(visibles, ignore_index=True)
    potentially_obstructed = pd.concat(potentially_obstructeds, ignore_index=True)
    return visible, potentially_obstructed

@delayed
def _find_potential_obstructions_2d(sub_chunk, obstructions_gdf):
    """
    For a batch of sight lines, finds all nearby obstructions using a spatial index.

    Parameters
    ----------
    sub_chunk : GeoDataFrame
        Subset of potential sight lines.
    obstructions_gdf : GeoDataFrame
        Obstruction geometries (must include 'buildingID').

    Returns
    -------
    sub_chunk : GeoDataFrame
        Same as input, with an added 'matchesIDs' column listing nearby building IDs for each line.
    """

    obstruction_geoms = [(geom, idx) for idx, geom in zip(obstructions_gdf.buildingID, obstructions_gdf.geometry)]
    
    # Build STRtree with the modified geometries including the index
    strtree = STRtree([geom for geom, _ in obstruction_geoms])   
    line_geoms = list(sub_chunk['geometry'].values)

    # Query STRtree for all potentially intersecting obstructions for each line geometry
    matches = [
        [obstruction_geoms[i][1] for i in strtree.query(line_geom)]  # Retrieve actual obstruction.index as it may be different from 0,1,2, strtree
        for line_geom in line_geoms
    ]
    
    # sub_chunk matches directly to the matchesIDs column, may include target's buildingID (dealt with later)
    sub_chunk['matchesIDs'] = matches
    
    strtree, matches, line_geoms = None, None, None
    return sub_chunk     
    
@delayed
def _find_obstructions_2d(sub_chunk, obstructions_gdf, columnIDs):
    """
    For each sight line in the batch, checks for actual 2D intersections with obstructions.

    Parameters
    ----------
    sub_chunk : GeoDataFrame
        Batch of candidate sight lines (with column given by `columnIDs` containing building IDs).
    obstructions_gdf : GeoDataFrame
        Obstruction polygons with 'buildingID', 'geometry', 'height', and 'base'.
    columnIDs : str
        Column name in `sub_chunk` listing potential obstruction building IDs.

    Returns
    -------
    visible : GeoDataFrame
        Sub-chunk of sight lines without any 2D obstruction.
    obstructed : GeoDataFrame
        Sub-chunk of sight lines obstructed in 2D, with obstruction coordinates/z-values.
    """
  
    # Build a dictionary for fast lookup of obstructions by buildingID
    buildingID_to_geom = obstructions_gdf.set_index('buildingID')['geometry'].to_dict()

    # Vectorized approach to generate the obstructions_xy column
    obstructions_xy = []
    obstructions_z = []
    
    for idx, row in sub_chunk.iterrows():
        # Get the list of intersecting buildingIDs
        potential_obstructionIDs = row[columnIDs]
        buildingID = row.buildingID
        
        # Remove the current buildingID from the list of potential obstructions
        potential_obstructionIDs = [bid for bid in potential_obstructionIDs if bid != buildingID]
        if not potential_obstructionIDs:
            obstructions_xy.append([])
            obstructions_z.append([])
            continue
            
        # Fast check if intersections exist for this sightline
        sight_line_geometry = row['geometry']
        intersecting_buildings = [
            buildingID for buildingID in potential_obstructionIDs
            if sight_line_geometry.intersects(buildingID_to_geom[buildingID])
        ]
        
        flat_coords, z_intersections = [], []
        
        # If there are intersections, compute the intersections
        if intersecting_buildings:
            flat_coords = []
            z_intersections = []
            for buildingID in intersecting_buildings:
                intersection_geom = sight_line_geometry.intersection(buildingID_to_geom[buildingID])
                for pt in (
                    intersection_geom.geoms if isinstance(intersection_geom, MultiPoint)
                    else intersection_geom.coords if isinstance(intersection_geom, (LineString, MultiLineString))
                    else [intersection_geom] if isinstance(intersection_geom, Point)
                    else []
                ):
                    flat_coords.append((pt.x, pt.y))  # Coordinates
                    z_intersections.append(obstructions_gdf.loc[buildingID, 'height'] + obstructions_gdf.loc[buildingID, 'base'])  # Z values

        obstructions_xy.append(flat_coords)
        obstructions_z.append(z_intersections)

    # Assign the list of obstructions to the 'obstructions_xy' column
    sub_chunk['obstructions_xy'] = obstructions_xy
    sub_chunk['obstructions_z'] = obstructions_z
    
    # Split into visible and obstructed lines based on the result of obstructions_xy
    mask_obstructed = sub_chunk['obstructions_xy'].map(len) > 0
    visible = sub_chunk.loc[~mask_obstructed].reset_index(drop=True)
    obstructed = sub_chunk.loc[mask_obstructed].reset_index(drop=True)

    # Clean up and return the results
    sub_chunk, building_id_to_geom, obstructions_xy, obstructions_z = None, None, None, None
    return visible, obstructed  
        
## Step 3 ###########################
def obstructions_3d(potential_sight_lines, num_workers=20):
    """
    Checks 3D visibility for potentially obstructed sight lines, using batched and parallelized processing.

    Each sight line is tested against obstruction heights along its path; only lines unobstructed in 3D are returned.

    Parameters
    ----------
    potential_sight_lines : GeoDataFrame
        Candidate sight lines to check in 3D (should already be filtered by 2D checks).
    num_workers : int, optional
        Number of parallel workers (default: 20).

    Returns
    -------
    visible : GeoDataFrame
        GeoDataFrame of sight lines unobstructed in 3D.
    """
   
    sub_chunks = _define_batches(potential_sight_lines)
    tasks = [_find_obstructions_3d(sub_chunk) for sub_chunk in sub_chunks]
    results = dask.compute(*tasks, scheduler="threads", num_workers= num_workers)
    visible = pd.concat(results, ignore_index=True)
    return visible
    
def obstructions_3d_with_pyvista(potential_sight_lines, obstructions_gdf, potential_obstructions_column, num_workers):
    """
    Checks 3D visibility of each sight line using detailed ray tracing with PyVista.

    Each sight line is tested for intersection with potential obstructions (buildings) using mesh ray tracing.
    Parallelized for speed using ThreadPoolExecutor.

    Parameters
    ----------
    potential_sight_lines : GeoDataFrame
        Candidate sight lines with observer/target coordinates and a column listing potential obstruction IDs.
    obstructions_gdf : GeoDataFrame
        Buildings with 'building_3d' mesh geometry.
    potential_obstructions_column : str
        Name of the column in `potential_sight_lines` with a list of candidate obstruction building IDs.
    num_workers : int
        Number of parallel threads to use for the visibility checks.

    Returns
    -------
    GeoDataFrame
        Only sight lines which are unobstructed in 3D, as determined by PyVista ray tracing.
    """
    
     # Remove the current buildingID from the list of potential obstructions for each sight line
    potential_sight_lines[potential_obstructions_column] = [
        [bid for bid in bids if bid != row.buildingID]
        for bids, row in zip(potential_sight_lines[potential_obstructions_column], potential_sight_lines.itertuples())
    ]
    
    # Perform parallelized intervisibility check
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(lambda row: _intervisibility_with_pyvista(row, obstructions_gdf, 
                                                                   potential_obstructions_column), potential_sight_lines.itertuples(index=False)),
            total=len(potential_sight_lines),
            desc="Checking intervisibility"
        ))
        
    potential_sight_lines["visible"] = results
    # Filter visible sight lines and append to the result
    potential_sight_lines = potential_sight_lines[potential_sight_lines["visible"] == True]
    return potential_sight_lines

@delayed
def _find_obstructions_3d(sub_chunk):
    """
    Performs fast 3D line-of-sight visibility checks for a batch of sight lines against obstruction heights.

    Interpolates the Z coordinate along each sight line and compares to the Z values of all obstruction points;
    returns only those sight lines that are unobstructed in 3D.

    Parameters
    ----------
    sub_chunk : GeoDataFrame
        Batch of potentially obstructed sight lines, with columns for observer/target coordinates,
        obstruction XY coordinates, and obstruction Z values.

    Returns
    -------
    GeoDataFrame
        Only sight lines in this batch which are visible in 3D.
    """

    # Build arrays for observer and target coordinates
    # Create columns for 2D coordinates
    observer_coords = np.array([(p.x, p.y, p.z) for p in sub_chunk['observer_geo']], dtype=np.float64)
    target_coords = np.array([(p.x, p.y, p.z) for p in sub_chunk['target_geo']], dtype=np.float64)

    # Extract obstructions coordinates
    obstructions_xy = sub_chunk['obstructions_xy'].to_numpy()
    obstructions_z = sub_chunk['obstructions_z'].to_numpy()
    
    # Create a 3D matrix for XY coordinates
    n_rows = len(obstructions_xy)
    max_len_obstructions = max(len(x) for x in obstructions_xy) # same length as z
    xy_array = np.full((n_rows, max_len_obstructions, 2), np.nan)  # (n_rows, max_len, 2 for X, Y)
    z_array = np.full((n_rows, max_len_obstructions), np.nan)  # (n_rows, max_len) for Z values
    
    # Fill in the XY values for obstructions and Z values
    for n, obstruction_xy in enumerate(obstructions_xy):
        if len(obstruction_xy):  # Ensure there are obstructions
            new_obstruction_xy = np.array(obstruction_xy)[:, :2]  # Only X and Y (no Z)
            new_obstruction_z = np.array(obstructions_z[n])  # Z values
            xy_array[n, :new_obstruction_xy.shape[0], :] = new_obstruction_xy
            z_array[n, :new_obstruction_z.shape[0]] = new_obstruction_z  # Correct placement of Zs
            
    # Observer and target coordinates
    x0 = observer_coords[:, 0][:, None]
    y0 = observer_coords[:, 1][:, None]
    z_observer = observer_coords[:, 2][:, None]
    x1 = target_coords[:, 0][:, None]
    y1 = target_coords[:, 1][:, None]
    z_target = target_coords[:, 2][:, None]

    # Calculate XY distance for each obstruction point
    dx = xy_array[..., 0] - x0
    dy = xy_array[..., 1] - y0
    distance_to_intersection = np.hypot(dx, dy)  # XY distance between points
    distance_to_target = np.hypot(x1 - x0, y1 - y0)  # Total XY distance from observer to target
    
    # Interpolate Z values along the line using the ratio of distances
    with np.errstate(invalid='ignore', divide='ignore'):
        z_intersections = z_observer + (distance_to_intersection / distance_to_target) * (z_target - z_observer)  # Z interpolation
    
    # Set NaN values where fictious intersection was inserted
    z_intersections[np.isnan(xy_array[..., 0])] = np.nan  
    # Perform the visibility check
    is_visible = _intervisibility(z_intersections, z_array)

    # Store visibility in the sub_chunk dataframe
    sub_chunk['visible'] = is_visible
    sub_chunk = sub_chunk[sub_chunk.visible == True] 
    return sub_chunk
    
@njit(parallel=True)
def _intervisibility(z_intersections, z_array):
    """
    Vectorized visibility check for many sight lines in parallel, comparing interpolated Z values
    along each line to the Z (height) values of intersected obstructions.

    Parameters
    ----------
    z_intersections : np.ndarray
        Interpolated Z values for each sight line and obstruction intersection.
        Shape: (n_lines, max_num_obstructions)
    z_array : np.ndarray
        Obstruction Z (height) values, same shape as z_intersections.

    Returns
    -------
    np.ndarray
        Boolean array (n_lines,) indicating which sight lines are visible (True = visible).
    """
    n_rows = z_intersections.shape[0]
    is_visible = np.zeros(n_rows, dtype=np.bool_)
    
    for i in prange(n_rows):
        # Compare z_intersections with the corresponding obstructions_z, ignoring NaNs
        z_intersection_row = z_intersections[i]
        obstructions_z_row = z_array[i]
        
        # Check visibility for the pairwise comparison
        visible_points = (z_intersection_row > obstructions_z_row) & ~np.isnan(z_intersection_row) & ~np.isnan(obstructions_z_row)
        
        # If all comparison points are True, set the row as visible
        is_visible[i] = np.all(visible_points)
    
    return is_visible    
    
def _intervisibility_with_pyvista(row, obstructions_gdf, potential_obstructions_column):
    """
    Checks if a 3D line of sight is unobstructed by any 3D building mesh, using PyVista ray tracing.

    Parameters
    ----------
    row : pandas.Series or namedtuple
        A single row from the potential sight lines DataFrame, containing observer and target coordinates,
        and the list of potential obstruction building IDs.
    obstructions_gdf : GeoDataFrame
        Building geometries, with 'building_3d' as a PyVista-compatible mesh.
    potential_obstructions_column : str
        Name of the column in `row` containing candidate obstruction building IDs.

    Returns
    -------
    bool
        True if no intersections are found (line is visible), False if any obstruction blocks the line.
    """
    
    potential_obstruction_ids = getattr(row, potential_obstructions_column)
    for buildingID in potential_obstruction_ids:
        building_3d = obstructions_gdf.loc[buildingID].building_3d.extract_surface().triangulate()
        points, intersections = building_3d.ray_trace(row.observer_coords, row.target_coords)
        if len(intersections) > 0:
            return False
    return True

def _finalize_sight_lines(sight_lines_tmp, nodes_gdf, consolidate, simplified_buildings):
    """
    Finalizes, cleans, and merges sight lines after visibility analysis.

    This function handles the final step in sight line construction by:
      - Exploding node pairs if node consolidation was performed.
      - (If using simplified buildings) Exploding all building-target pairs from simplified outlines.
      - Rebuilding correct LineString geometries for each observer-target pair.
      - Dropping intermediate columns used in previous steps.
      - Sorting and deduplicating the result so each (buildingID, nodeID) pair appears only once, keeping the longest line.

    Parameters
    ----------
    sight_lines_tmp : DataFrame or GeoDataFrame
        Temporary results table with observer/target columns, node/building references, and possibly intermediate attributes.
    nodes_gdf : GeoDataFrame
        Reference nodes with 3D coordinates, including 'nodeID', 'geometry', and (optionally) 'z'.
    consolidate : bool
        Whether node consolidation (merging/clustered nodes) was performed.
    simplified_buildings : GeoDataFrame
        If non-empty, provides simplified outlines for additional sight line explosion and assignment.

    Returns
    -------
    sight_lines : GeoDataFrame
        Cleaned, deduplicated GeoDataFrame of final sight lines, with correct geometry, observer and target references, and length.
    """
    nodes_gdf['geometry'] = [Point(geometry.x, geometry.y, z) for geometry, z in zip(nodes_gdf['geometry'], nodes_gdf['z'])]     

    if consolidate:
        sight_lines_tmp = sight_lines_tmp.explode(column='old_nodeIDs', ignore_index=True)
        sight_lines_tmp['nodeID'] = sight_lines_tmp['old_nodeIDs']
        geom_map = nodes_gdf.set_index('nodeID')['geometry']
        sight_lines_tmp['observer_geo'] = sight_lines_tmp['nodeID'].map(geom_map)

    if simplified_buildings is not None and not simplified_buildings.empty:
        sight_lines_tmp['pair'] = sight_lines_tmp.apply(
            lambda row: list(zip(row['buildingIDs'], row['target_GEOs'])), axis=1
        )
        sight_lines_tmp_exploded = sight_lines_tmp.explode('pair', ignore_index=True)
        sight_lines_tmp_exploded['buildingID'] = sight_lines_tmp_exploded['pair'].apply(lambda x: x[0])
        sight_lines_tmp_exploded['target_geo'] = sight_lines_tmp_exploded['pair'].apply(lambda x: x[1])
        sight_lines_tmp = sight_lines_tmp_exploded.copy()
    
    sight_lines_tmp['geometry'] = [LineString([observer, target]) for observer, target in zip(sight_lines_tmp['observer_geo'], 
                                                                                              sight_lines_tmp['target_geo'])]
      
    sight_lines_tmp = sight_lines_tmp.drop(['old_nodeIDs', 'buildingIDs', 'target_GEOs', 'pair', 'observer_geo', 'target_geo', 'observer_coords',
                                           'target_coords'], errors='ignore', axis=1)
    sight_lines = gpd.GeoDataFrame(sight_lines_tmp, geometry='geometry', crs=nodes_gdf.crs)
    sight_lines['length'] = sight_lines.geometry.length

    sight_lines = sight_lines.sort_values(['buildingID', 'nodeID', 'length'], ascending=[False, False, False]).drop_duplicates(
        ['buildingID', 'nodeID'], keep='first')
    sight_lines.reset_index(inplace=True, drop=True)

    return sight_lines
  
def polygon_2d_to_3d(building_polygon, base, height, extrude_from_zero = True, height_from_base = True):
    """
    Convert a 2D polygon (building) to a 3D geometry by extruding it with a specified height and base.
    
    Parameters
    ----------
    building_polygon : shapely.geometry.Polygon
        2D polygon representing the building footprint to be extruded.
    
    base : float
        The base height of the 3D polygon (the height from which extrusion starts).
        
    height : float
        The total height of the 3D polygon after extrusion.
        
    extrude_from_zero : bool, optional, default=True
        If True, the extrusion starts from the zero height (i.e., sea level). If False, the extrusion starts from the `base` height.
        
    height_from_base : bool, optional, default=True
        If True, the `height` is considered to be above the `base`. If False, it is considered above sea level (i.e., height is reduced by `base`).
        
    Returns
    -------
    pyvista.PolyData
        The resulting 3D polygon (building) after extrusion.
    """
    
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
    
    poly_points = building_polygon.exterior.coords
       
    # Reorient the coordinates of the polygon
    xy = reorient_coords(poly_points)
    # Create 3D coordinates with the base height or "fictious" ground
    # Determine the starting height and adjust the extrusion height based on the flags
    if extrude_from_zero:
        # If extruding from zero (e.g., ground level), set the base height to 0
        building_base = 0.0
        if height_from_base:
            # If height is computed from base (e.g., above ground level), add base to height
            height = height + base
    else:
        # If extruding from the given base, set the base height to the given base value
        building_base = base
        if not height_from_base:
            # If height is computed from sea level (not the base), subtract the base from height
            height = height - base
   
    xyz_base = [(x, y, building_base) for x, y in xy]
    # Create faces of the polygon
    faces = [len(xyz_base), *range(len(xyz_base))]
    # Create the 3D polygon using pyvista
    polygon = pv.PolyData(xyz_base, faces=faces)
    
    # Extrude the 3D polygon to the specified height 
    return polygon.extrude((0, 0, height), capping=True)
  
  
  
  
  
  
  
  
  